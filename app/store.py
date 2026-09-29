"""简历存储：上传的简历在这里暂存，供后续匹配与批量排序使用。

两种模式
--------
- **落盘模式**（``persist=True``，默认）：每份简历存成 ``data/resumes/<id>.json``。
  好处是服务重启后上传的简历还在，不用重复粘贴。
  ``data/`` 已被 ``.gitignore`` 排除 —— 简历属于个人隐私数据，
  绝不可以进版本库，也不应该被打进容器镜像（``.dockerignore`` 同样排除了它）。
- **内存模式**（``persist=False``）：只放内存，进程退出即消失。
  测试用这个模式，避免在仓库里留下测试残留文件。

为什么不用数据库？
    这个项目的定位是"可读、可跑、可验证的 AI 应用范例"。
    引入 SQLite 或 Postgres 会让读者需要额外理解一层存储细节，
    而实际需求只是"存几十份简历文本"。用 JSON 文件反而更透明 ——
    用户可以直接打开文件看到自己上传了什么。
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

#: 单份简历允许的最大字符数（防止有人粘贴一整本书进来）
MAX_RESUME_CHARS = 60_000

#: 简历 id 的格式：12 位十六进制
_ID_RE = re.compile(r"^[0-9a-f]{12}$")


@dataclass
class ResumeRecord:
    """一份已上传的简历。"""

    resume_id: str
    text: str
    name: str = ""
    char_count: int = 0
    uploaded_at: str = ""
    source: str = "paste"

    def to_dict(self) -> dict[str, object]:
        return {
            "resume_id": self.resume_id,
            "name": self.name,
            "text": self.text,
            "char_count": self.char_count,
            "uploaded_at": self.uploaded_at,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ResumeRecord":
        """从字典恢复（对脏数据做容错，单条损坏不应该让整个存储加载失败）。"""
        text = str(data.get("text") or "")
        return cls(
            resume_id=str(data.get("resume_id") or ""),
            text=text,
            name=str(data.get("name") or ""),
            char_count=int(data.get("char_count") or len(text)),
            uploaded_at=str(data.get("uploaded_at") or ""),
            source=str(data.get("source") or "paste"),
        )


def _now_iso() -> str:
    """当前时间的 ISO 字符串（秒精度，够用了）。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _guess_name(text: str) -> str:
    """从简历文本里猜一个显示名。

    只用于列表展示，猜不到就退回"未命名简历"。
    真正的姓名抽取由 parser 负责（那个更严谨）。
    """
    from .parser import find_name
    from .sections import split_sections
    from .textutil import normalize_text

    clean = normalize_text(text)
    index = split_sections(clean)
    guessed = find_name(index.preamble, clean)
    return guessed or "未命名简历"


class ResumeStore:
    """简历仓库（内存索引 + 可选落盘）。"""

    def __init__(self, data_dir: Path | str = "data", *, persist: bool = True) -> None:
        self.dir = Path(data_dir) / "resumes"
        self.persist = persist
        self._items: dict[str, ResumeRecord] = {}
        self._order: list[str] = []

        if self.persist:
            self.dir.mkdir(parents=True, exist_ok=True)
            self._load()

    # -- 内部 --------------------------------------------------------------

    def _load(self) -> None:
        """启动时把磁盘上的简历读进内存。

        单份文件损坏只跳过那一份并告警，不让整个服务起不来 ——
        存储层不该成为可用性的单点。
        """
        if not self.dir.exists():
            return
        loaded = 0
        for path in sorted(self.dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("跳过损坏的简历文件 %s：%s", path.name, exc)
                continue
            record = ResumeRecord.from_dict(data)
            if not record.resume_id or not record.text:
                logger.warning("跳过字段不完整的简历文件：%s", path.name)
                continue
            self._items[record.resume_id] = record
            self._order.append(record.resume_id)
            loaded += 1
        if loaded:
            logger.info("已从 %s 载入 %d 份简历", self.dir, loaded)

    def _write(self, record: ResumeRecord) -> None:
        if not self.persist:
            return
        path = self.dir / f"{record.resume_id}.json"
        try:
            path.write_text(
                json.dumps(record.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            # 写盘失败不影响内存里的数据，服务继续可用
            logger.warning("简历落盘失败（不影响本次使用）：%s", exc)

    # -- 对外 --------------------------------------------------------------

    def add(self, text: str, *, name: str = "", source: str = "paste") -> ResumeRecord:
        """新增一份简历。

        Args:
            text: 简历原文。
            name: 显示名；为空时自动从文本里猜。
            source: 来源标记（paste / upload）。

        Returns:
            新建的记录。

        Raises:
            ValueError: 文本为空或超长。
        """
        body = (text or "").strip()
        if not body:
            raise ValueError("简历内容为空")
        if len(body) > MAX_RESUME_CHARS:
            raise ValueError(f"简历内容过长（{len(body)} 字符，上限 {MAX_RESUME_CHARS}）")

        record = ResumeRecord(
            resume_id=uuid.uuid4().hex[:12],
            text=body,
            name=(name or "").strip() or _guess_name(body),
            char_count=len(body),
            uploaded_at=_now_iso(),
            source=source,
        )
        self._items[record.resume_id] = record
        self._order.append(record.resume_id)
        self._write(record)
        return record

    def get(self, resume_id: str) -> Optional[ResumeRecord]:
        """按 id 取简历。id 格式非法时直接返回 None（不做磁盘查找）。"""
        if not resume_id or not _ID_RE.match(resume_id):
            return None
        return self._items.get(resume_id)

    def require(self, resume_id: str) -> ResumeRecord:
        """按 id 取简历，取不到抛 KeyError（便于路由层统一转 404）。"""
        record = self.get(resume_id)
        if record is None:
            raise KeyError(f"简历不存在：{resume_id}")
        return record

    def list(self) -> list[ResumeRecord]:
        """按上传时间倒序列出全部简历。"""
        return [self._items[rid] for rid in reversed(self._order) if rid in self._items]

    def remove(self, resume_id: str) -> bool:
        """删除一份简历。返回是否真的删掉了东西。"""
        if resume_id not in self._items:
            return False
        del self._items[resume_id]
        if resume_id in self._order:
            self._order.remove(resume_id)
        if self.persist:
            path = self.dir / f"{resume_id}.json"
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("删除简历文件失败：%s", exc)
        return True

    def clear(self) -> int:
        """清空全部简历，返回删除数量。"""
        count = len(self._items)
        for rid in list(self._items):
            self.remove(rid)
        return count

    @property
    def count(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[ResumeRecord]:
        return iter(self.list())


__all__ = ["ResumeStore", "ResumeRecord", "MAX_RESUME_CHARS"]
