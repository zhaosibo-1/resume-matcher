"""简历存储层测试。

存储里有两处特别值得用测试守住的点：

1. **id 合法性前置校验**：``get()`` 在 id 格式不对时直接返回 None，
   不去磁盘上找。否则任何外部输入（URL 路径参数）都能变成一个文件系统探测点。
2. **单份文件损坏不能拖垮整个存储**：启动加载时跳过坏文件 + 告警，
   而不是让服务起不来。存储层不该成为可用性的单点。
"""

from __future__ import annotations

import json

import pytest

from app.store import MAX_RESUME_CHARS, ResumeRecord, ResumeStore

RESUME_TEXT = """\
张三
电话：138-0000-0000

专业技能
- Python、FastAPI
"""


@pytest.fixture()
def store(tmp_path) -> ResumeStore:
    """落盘模式的存储（指向临时目录，不污染仓库）。"""
    return ResumeStore(tmp_path / "data", persist=True)


@pytest.fixture()
def memory_store() -> ResumeStore:
    return ResumeStore("unused", persist=False)


# ===========================================================================
# 新增
# ===========================================================================


class TestAdd:
    def test_add_returns_record_with_generated_id(self, memory_store: ResumeStore) -> None:
        record = memory_store.add(RESUME_TEXT)
        assert len(record.resume_id) == 12
        assert all(c in "0123456789abcdef" for c in record.resume_id)
        assert record.char_count == len(RESUME_TEXT.strip())
        assert record.uploaded_at

    def test_name_is_guessed_from_text(self, memory_store: ResumeStore) -> None:
        """没显式给名字时，从正文里猜一个用于列表展示。"""
        assert memory_store.add(RESUME_TEXT).name == "张三"

    def test_explicit_name_wins_over_guess(self, memory_store: ResumeStore) -> None:
        assert memory_store.add(RESUME_TEXT, name="我的简历").name == "我的简历"

    def test_blank_explicit_name_falls_back_to_guess(self, memory_store: ResumeStore) -> None:
        assert memory_store.add(RESUME_TEXT, name="   ").name == "张三"

    def test_unguessable_name_uses_placeholder(self, memory_store: ResumeStore) -> None:
        assert memory_store.add("1234567890").name == "未命名简历"

    def test_empty_text_rejected(self, memory_store: ResumeStore) -> None:
        with pytest.raises(ValueError, match="为空"):
            memory_store.add("   \n  ")
        assert memory_store.count == 0

    def test_oversized_text_rejected(self, memory_store: ResumeStore) -> None:
        with pytest.raises(ValueError, match="过长"):
            memory_store.add("x" * (MAX_RESUME_CHARS + 1))

    def test_text_is_stripped(self, memory_store: ResumeStore) -> None:
        record = memory_store.add("\n\n" + RESUME_TEXT + "\n\n\n")
        assert not record.text.startswith("\n")
        assert not record.text.endswith("\n")

    def test_ids_are_unique(self, memory_store: ResumeStore) -> None:
        ids = {memory_store.add(RESUME_TEXT).resume_id for _ in range(30)}
        assert len(ids) == 30

    def test_source_recorded(self, memory_store: ResumeStore) -> None:
        assert memory_store.add(RESUME_TEXT, source="upload").source == "upload"
        assert memory_store.add(RESUME_TEXT).source == "paste"


# ===========================================================================
# 查询
# ===========================================================================


class TestGet:
    def test_get_existing(self, memory_store: ResumeStore) -> None:
        record = memory_store.add(RESUME_TEXT)
        assert memory_store.get(record.resume_id) is record

    def test_get_missing_returns_none(self, memory_store: ResumeStore) -> None:
        assert memory_store.get("aaaaaaaaaaaa") is None

    @pytest.mark.parametrize(
        "bad_id",
        [
            "",
            "short",
            "ZZZZZZZZZZZZ",       # 非十六进制
            "aaaaaaaaaaaaa",      # 13 位
            "../etc/passwd",      # 路径穿越
            "aaaaaaaaaaaa/../x",
            "../../data/resumes/aaaaaaaaaaaa",
        ],
    )
    def test_invalid_id_returns_none_without_touching_disk(self, memory_store: ResumeStore, bad_id: str) -> None:
        """非法 id 必须直接判死 —— 这是路径穿越的第一道闸门。"""
        assert memory_store.get(bad_id) is None

    def test_require_raises_keyerror(self, memory_store: ResumeStore) -> None:
        with pytest.raises(KeyError):
            memory_store.require("aaaaaaaaaaaa")

    def test_require_returns_record(self, memory_store: ResumeStore) -> None:
        record = memory_store.add(RESUME_TEXT)
        assert memory_store.require(record.resume_id).resume_id == record.resume_id


class TestList:
    def test_list_is_newest_first(self, memory_store: ResumeStore) -> None:
        first = memory_store.add(RESUME_TEXT, name="第一份")
        second = memory_store.add(RESUME_TEXT, name="第二份")
        assert [r.name for r in memory_store.list()] == ["第二份", "第一份"]
        assert [r.name for r in memory_store] == ["第二份", "第一份"]
        assert first.resume_id != second.resume_id

    def test_list_empty(self, memory_store: ResumeStore) -> None:
        assert memory_store.list() == []
        assert memory_store.count == 0


# ===========================================================================
# 删除
# ===========================================================================


class TestRemove:
    def test_remove_existing(self, memory_store: ResumeStore) -> None:
        record = memory_store.add(RESUME_TEXT)
        assert memory_store.remove(record.resume_id) is True
        assert memory_store.get(record.resume_id) is None
        assert memory_store.count == 0

    def test_remove_missing_returns_false(self, memory_store: ResumeStore) -> None:
        assert memory_store.remove("aaaaaaaaaaaa") is False

    def test_remove_twice_is_false_second_time(self, memory_store: ResumeStore) -> None:
        record = memory_store.add(RESUME_TEXT)
        assert memory_store.remove(record.resume_id) is True
        assert memory_store.remove(record.resume_id) is False

    def test_clear(self, memory_store: ResumeStore) -> None:
        for _ in range(4):
            memory_store.add(RESUME_TEXT)
        assert memory_store.clear() == 4
        assert memory_store.count == 0


# ===========================================================================
# 落盘
# ===========================================================================


class TestPersistence:
    def test_writes_json_file(self, tmp_path) -> None:
        store = ResumeStore(tmp_path / "data", persist=True)
        record = store.add(RESUME_TEXT)
        path = tmp_path / "data" / "resumes" / f"{record.resume_id}.json"
        assert path.is_file()

        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved["resume_id"] == record.resume_id
        assert saved["text"] == record.text
        # 中文不能变成 \uXXXX，用户要能直接打开文件看懂
        assert "张三" in path.read_text(encoding="utf-8")

    def test_reload_from_disk(self, tmp_path) -> None:
        first = ResumeStore(tmp_path / "data", persist=True)
        record = first.add(RESUME_TEXT)

        second = ResumeStore(tmp_path / "data", persist=True)
        assert second.count == 1
        assert second.get(record.resume_id).text == record.text

    def test_remove_deletes_file(self, tmp_path) -> None:
        store = ResumeStore(tmp_path / "data", persist=True)
        record = store.add(RESUME_TEXT)
        path = tmp_path / "data" / "resumes" / f"{record.resume_id}.json"
        store.remove(record.resume_id)
        assert not path.exists()

    def test_memory_mode_writes_nothing(self, tmp_path) -> None:
        store = ResumeStore(tmp_path / "data", persist=False)
        store.add(RESUME_TEXT)
        assert not (tmp_path / "data" / "resumes").exists()

    def test_corrupted_file_is_skipped_not_fatal(self, tmp_path) -> None:
        """损坏的单份文件只跳过并告警，不能让整个服务起不来。"""
        good = ResumeStore(tmp_path / "data", persist=True)
        record = good.add(RESUME_TEXT)

        (tmp_path / "data" / "resumes" / "bad000000000.json").write_text("{ 这不是 JSON", encoding="utf-8")
        (tmp_path / "data" / "resumes" / "incomplete00.json").write_text(
            json.dumps({"resume_id": "incomplete00", "text": ""}), encoding="utf-8"
        )

        reloaded = ResumeStore(tmp_path / "data", persist=True)
        assert reloaded.count == 1
        assert reloaded.get(record.resume_id) is not None

    def test_missing_directory_is_not_an_error(self, tmp_path) -> None:
        store = ResumeStore(tmp_path / "does-not-exist", persist=False)
        assert store.count == 0


# ===========================================================================
# 记录对象的序列化
# ===========================================================================


class TestResumeRecord:
    def test_roundtrip(self) -> None:
        original = ResumeRecord(
            resume_id="aaaaaaaaaaaa",
            text="内容",
            name="张三",
            char_count=2,
            uploaded_at="2026-01-01 00:00:00",
            source="upload",
        )
        restored = ResumeRecord.from_dict(original.to_dict())
        assert restored == original

    def test_from_dict_tolerates_dirty_data(self) -> None:
        """字段缺失/类型不对时用兜底值，而不是抛异常。"""
        record = ResumeRecord.from_dict({"resume_id": "aaaaaaaaaaaa", "text": "abc"})
        assert record.name == ""
        assert record.char_count == 3          # 缺失时按正文长度补
        assert record.source == "paste"
        assert record.uploaded_at == ""

    def test_from_dict_ignores_unknown_fields(self) -> None:
        record = ResumeRecord.from_dict(
            {"resume_id": "aaaaaaaaaaaa", "text": "abc", "奇怪字段": {"nested": 1}}
        )
        assert record.resume_id == "aaaaaaaaaaaa"

    def test_from_dict_handles_none_values(self) -> None:
        """JSON 里的 null 要变成空串 / 0，而不是字符串 "None"。"""
        record = ResumeRecord.from_dict({"resume_id": None, "text": None, "name": None, "char_count": None})
        assert record.resume_id == ""
        assert record.text == ""
        assert record.name == ""
        assert record.char_count == 0
