"""大模型抽取层：只做一件规则做不好的事 —— 从自然语言里"看懂"内容。

分工原则（整个项目的核心设计）
------------------------------
    规则（parser/skills）负责：确定性的事 —— 归一化、定位、分级、打分
    模型（本模块）负责：理解性的事 —— 从口语化描述里识别出技能与要求

举例说明为什么需要模型：
    JD 里写「有过向量检索相关实践，了解过一些大模型的应用场景」。
    规则词典能认出"向量检索"、"大模型"，但认不出这句话的**强度**
    （"了解过一些" 其实是 nice 而不是 must）。

模型只**补漏**，不推翻规则：
    - 规则已经抽到的技能，模型再提一次只会把可信度标记升级为 dict+llm；
    - 规则没抽到的，模型提了才加进来；
    - 模型的任何输出都要经过类型校验和清洗，脏数据一律丢弃。

为什么不让模型直接打分？
    因为不可复现。同一份简历跑两次得到不同分数，在招聘场景里是致命的。
    分数必须来自确定性算法，模型只能影响"输入了哪些技能"。
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from .config import Settings
from .llm import LLMClient, LLMError

logger = logging.getLogger(__name__)

#: 单次抽取允许的最大输入字符数。超过就截断 —— 简历/JD 再长也不会超过这个量级，
#: 而无上限的输入会把 token 成本推高，还可能触发模型的上下文长度限制。
MAX_INPUT_CHARS = 12000


JD_SYSTEM_PROMPT = """你是一位资深的招聘信息分析专家，擅长把口语化的职位描述拆解成结构化的信息。

你的任务是阅读一段职位描述（JD），输出一个 JSON 对象。要求：

1. **只输出 JSON**，不要任何解释、寒暄或 Markdown 代码块标记。
2. 严格按下面的字段定义输出，不要增加或删减字段：

{
  "title": "岗位名称（字符串，识别不到给空串）",
  "company": "公司名称（字符串，识别不到给空串）",
  "skills": ["技术或技能名词（数组，只放真正的硬技能）"],
  "bonus_skills": ["加分项技能（数组，只放明确写着'加分/优先/了解'的）"],
  "soft_skills": ["软性能力（数组，如 沟通能力、团队协作）"],
  "education": "学历要求（字符串，如 '本科及以上'；没有则空串）",
  "min_years": 经验年限要求（数字，如 3；'应届/不限'给 0；没写给 null）,
  "domains": ["行业方向（数组，如 '金融科技'、'电商'、'智能客服'；没有则空数组）"],
  "seniority": "招聘层级（字符串：实习/校招/初级/中级/高级/专家；判断不出给空串）"
}

3. skills 里请提取**具体的技术名词**（如 Python、FastAPI、向量数据库、RAG），
   不要把整句话塞进来，也不要提取"编程能力"这种笼统说法。
4. 同一个技能不要重复出现。中文和英文写法请统一为**业内最常见的写法**
   （例如统一写 "Kubernetes" 而不是 "k8s"）。
5. 如果某个字段在 JD 里完全没有线索，用空串、空数组或 null，**不要编造**。"""


RESUME_SYSTEM_PROMPT = """你是一位资深的简历解析专家，擅长从简历文本里提取结构化信息。

你的任务是阅读一份简历，输出一个 JSON 对象。要求：

1. **只输出 JSON**，不要任何解释、寒暄或 Markdown 代码块标记。
2. 严格按下面的字段定义输出，不要增加或删减字段：

{
  "name": "姓名（字符串，识别不到给空串）",
  "education": "最高学历（字符串，如 '本科'、'硕士'；没有则空串）",
  "school": "学校名称（字符串）",
  "major": "专业名称（字符串）",
  "graduation_year": 毕业年份（数字，如 2027；不确定给 null）,
  "skills": ["技能名词（数组，只放简历里真正出现过的）"],
  "total_years": 累计相关经验年限（数字，含实习；应届生按实习与项目折算，给一位小数）,
  "highlights": ["最能体现能力的 3-6 条原句（数组，优先挑带具体数字的）"]
}

3. skills 必须是简历里**真实出现过**的技术名词，不要根据"看起来像这个岗位"
   而推测补充。宁可少写，也不要编造。
4. 中文和英文写法请统一为**业内最常见的写法**（例如统一写 "PyTorch"）。
5. highlights 请直接引用简历原句（可以截取，但不要改写），
   优先选择含具体数字、指标、成果的句子。"""


# ---------------------------------------------------------------------------
# 清洗与校验
# ---------------------------------------------------------------------------


def _clean_str(value: Any, *, limit: int = 200) -> str:
    """把任意值清洗成一个安全的短字符串。

    模型有时候会返回嵌套对象或者超长文本，直接塞进 schema 会让
    接口响应体积失控，甚至引发前端渲染问题。这里统一收口。
    """
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        text = str(value)
    elif isinstance(value, str):
        text = value
    else:
        return ""
    text = text.strip().strip('"').strip("'")
    text = re.sub(r"\s+", " ", text)
    return text[:limit]


#: ``_clean_str`` 的"不截断"上限。用它先做一次完整清洗，
#: 才有办法判断**原始条目**是否超长（见 ``_clean_str_list``）。
_UNCAPPED = 1 << 20


def _clean_str_list(value: Any, *, limit: int = 60, max_items: int = 80, item_limit: int = 60) -> list[str]:
    """把任意值清洗成字符串列表（去重、去空、限长、限量）。

    模型返回 ``"skills": "Python, Java"`` 这种用字符串代替数组的情况
    非常常见，所以这里对字符串做了按分隔符切分的兼容处理。

    关于超长条目：**丢弃，而不是截断**。
    超长条目几乎总是模型把一整句话塞进了列表（"熟悉并能够独立完成基于 RAG 的
    问答链路搭建"）。截断会留下半句话，它会以"技能"的身份进入词典匹配与打分，
    既匹配不上任何东西、又在报告里显眼地碍事 —— 比缺一项更糟。
    """
    if value is None:
        return []

    if isinstance(value, str):
        parts = re.split(r"[,，、;；\n|]", value)
    elif isinstance(value, (list, tuple, set)):
        parts = list(value)
    else:
        return []

    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        # 先做一次不截断的清洗：这样才拿得到"条目真实长度"。
        # （直接在截断后的文本上比较长度是永远不成立的死代码。）
        full = _clean_str(part, limit=_UNCAPPED)
        if not full:
            continue
        if len(full) > limit:
            logger.debug("丢弃超长条目（%d 字符）：%s…", len(full), full[:40])
            continue
        text = full[:item_limit]
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= max_items:
            break
    return out


def _clean_number(value: Any, *, minimum: float = 0.0, maximum: float = 60.0) -> Optional[float]:
    """把任意值清洗成一个合理范围内的数字（超出范围视为脏数据，返回 None）。"""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        m = re.search(r"-?\d+(?:\.\d+)?", value)
        if not m:
            return None
        value = m.group(0)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not (minimum <= number <= maximum):
        return None
    return number


def _clean_year(value: Any) -> Optional[int]:
    """清洗年份（1990 ~ 当前年 + 10）。"""
    from .parser import _now_ym

    number = _clean_number(value, minimum=1990, maximum=_now_ym()[0] + 10)
    return int(number) if number is not None else None


def sanitize_jd_payload(data: dict[str, Any]) -> dict[str, Any]:
    """清洗模型返回的 JD 抽取结果。未知字段一律丢弃。"""
    skills = _clean_str_list(data.get("skills"))
    bonus = _clean_str_list(data.get("bonus_skills"))

    # 同一个技能同时出现在 skills 和 bonus_skills 时，按硬性处理（更强的那个）
    bonus = [name for name in bonus if name.lower() not in {s.lower() for s in skills}]

    min_years_raw = data.get("min_years")
    min_years = _clean_number(min_years_raw, minimum=0.0, maximum=30.0)

    return {
        "title": _clean_str(data.get("title"), limit=60),
        "company": _clean_str(data.get("company"), limit=60),
        "skills": skills,
        "bonus_skills": bonus,
        "soft_skills": _clean_str_list(data.get("soft_skills"), max_items=20, item_limit=30),
        "education": _clean_str(data.get("education"), limit=30),
        "min_years": min_years,
        "domains": _clean_str_list(data.get("domains"), max_items=12, item_limit=20),
        "seniority": _clean_str(data.get("seniority"), limit=12),
    }


def sanitize_resume_payload(data: dict[str, Any]) -> dict[str, Any]:
    """清洗模型返回的简历抽取结果。"""
    return {
        "name": _clean_str(data.get("name"), limit=40),
        "education": _clean_str(data.get("education"), limit=30),
        "school": _clean_str(data.get("school"), limit=60),
        "major": _clean_str(data.get("major"), limit=40),
        "graduation_year": _clean_year(data.get("graduation_year")),
        "skills": _clean_str_list(data.get("skills")),
        "total_years": _clean_number(data.get("total_years"), minimum=0.0, maximum=30.0),
        "highlights": _clean_str_list(data.get("highlights"), max_items=10, item_limit=160),
    }


# ---------------------------------------------------------------------------
# 抽取器
# ---------------------------------------------------------------------------


class Extractor:
    """把「调用模型 + 清洗 + 降级」三件事收口成一个类。

    对外只暴露两个方法，且**都不抛异常** —— 抽取失败一律返回 None，
    由调用方无缝降级到规则引擎。这是"零配置可跑"的技术基础：
    没配 Key 或者 Key 欠费了，系统依然完整可用，只是抽取精度略降。
    """

    def __init__(self, settings: Settings, client: LLMClient) -> None:
        self.settings = settings
        self.client = client
        #: 最近一次失败原因，供 /api/config 展示"为什么走了规则引擎"
        self.last_error: str = ""
        self.last_error_kind: str = ""

    def _should_use_llm(self) -> bool:
        """当前配置是否应该调用模型。"""
        if self.settings.engine_mode == "rule":
            return False
        if self.settings.engine_mode == "llm" and not self.settings.llm_enabled:
            # 强制 llm 但没配 Key：不是静默降级，而是明确记录原因
            self.last_error = "PARSER_ENGINE=llm 但未配置 LLM_API_KEY"
            self.last_error_kind = "config"
            return False
        return self.settings.llm_enabled

    async def extract_jd(self, text: str) -> Optional[dict[str, Any]]:
        """抽取 JD 的结构化信息。

        Returns:
            清洗后的字典；未启用模型或调用失败时返回 None。
        """
        if not self._should_use_llm():
            return None

        payload = text[:MAX_INPUT_CHARS]
        try:
            raw = await self.client.chat_json(
                JD_SYSTEM_PROMPT,
                f"请解析下面这份职位描述：\n\n{payload}",
            )
        except LLMError as exc:
            self.last_error = str(exc)
            self.last_error_kind = exc.kind
            logger.warning("JD 抽取失败（将降级为规则引擎）：%s", exc)
            return None

        cleaned = sanitize_jd_payload(raw)
        self.last_error = ""
        self.last_error_kind = ""
        logger.info(
            "JD 抽取完成：技能 %d 项 / 加分 %d 项 / 年限 %s",
            len(cleaned["skills"]),
            len(cleaned["bonus_skills"]),
            cleaned["min_years"],
        )
        return cleaned

    async def extract_resume(self, text: str) -> Optional[dict[str, Any]]:
        """抽取简历的结构化信息。"""
        if not self._should_use_llm():
            return None

        payload = text[:MAX_INPUT_CHARS]
        try:
            raw = await self.client.chat_json(
                RESUME_SYSTEM_PROMPT,
                f"请解析下面这份简历：\n\n{payload}",
            )
        except LLMError as exc:
            self.last_error = str(exc)
            self.last_error_kind = exc.kind
            logger.warning("简历抽取失败（将降级为规则引擎）：%s", exc)
            return None

        cleaned = sanitize_resume_payload(raw)
        self.last_error = ""
        self.last_error_kind = ""
        logger.info(
            "简历抽取完成：姓名 %r / 技能 %d 项 / 年限 %s",
            cleaned["name"],
            len(cleaned["skills"]),
            cleaned["total_years"],
        )
        return cleaned

    def describe(self) -> dict[str, Any]:
        """当前抽取器状态（用于 /api/config）。"""
        return {
            "engine_mode": self.settings.engine_mode,
            "active_engine": self.settings.active_engine,
            "llm_enabled": self.settings.llm_enabled,
            "last_error": self.last_error,
            "last_error_kind": self.last_error_kind,
        }


__all__ = [
    "Extractor",
    "sanitize_jd_payload",
    "sanitize_resume_payload",
    "JD_SYSTEM_PROMPT",
    "RESUME_SYSTEM_PROMPT",
    "MAX_INPUT_CHARS",
]
