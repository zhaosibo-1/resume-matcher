"""结构化解析器：把 JD / 简历的自然语言文本变成结构化的对象。

这是整个系统的"翻译层"。它要回答的问题是：
    「这段人写的文字里，到底要求了什么 / 具备什么？」

三层协作关系
------------
本模块刻意把工作拆给两层，各司其职：

    词典层（skills.py）  → 归一化与定位："js" 和 "JavaScript" 是同一个东西
    本模块（parser.py）  → 结构与判断："这条写在『加分项』下面，算 nice"

大模型（可选）只在**词典覆盖不到**的时候补位 —— 例如 JD 里写了
"有过向量检索相关实践"，词典里没有"向量检索"这个词条（实际上有，
但假设没有），这时让模型来判断它是否等价于某个已知技能。

合并规则是**规则优先、模型补空缺**，理由：规则的结果稳定可复现，
模型的结果每次都可能不一样。匹配系统里结果不稳定是致命的。

年限计算的特别说明
------------------
简历里多段经历的时间**经常重叠**（一边实习一边做项目、两段实习间有gap）。
简单地把各段时长相加会显著虚高。所以这里用了区间合并：
把所有经历视为时间段，合并重叠部分后再求总长。
这个细节决定了「3 段实习各 3 个月」是算 9 个月还是 9 个月里实际只有 5 个月。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from . import sections as sec
from .schemas import (
    ExperienceItem,
    ParsedJD,
    ParsedResume,
    Requirement,
    SkillItem,
)
from .skills import (
    CAT_DOMAIN,
    DEFAULT_NORMALIZER,
    EDUCATION_LEVELS,
    EXPLICIT_NICE_WORDS,
    GENERAL_NICE_WORDS,
    MUST_MARKERS,
    NICE_MARKERS,
    SOFT_MUST_WORDS,
    STRONG_MUST_WORDS,
    SkillMention,
    SkillNormalizer,
)
from .textutil import (
    DateRange,
    context_phrase,
    evidence_around,
    find_date_range,
    find_single_date,
    find_years,
    has_quantified,
    is_bullet,
    normalize_text,
    shorten,
    split_blocks,
    strip_bullet,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# 一、通用检测工具
# ===========================================================================

#: 招聘层级判定规则（按顺序取第一个命中）
_SENIORITY_RULES: tuple[tuple[str, str], ...] = (
    (r"实习|intern|日常实习|暑期实习|寒假实习", "实习"),
    (r"校招|校园招聘|应届|毕业生|20\d{2}\s*届", "校招"),
    (r"专家|架构师|architect|技术专家", "专家"),
    (r"资深|高级|senior|sr\.|sr ", "高级"),
    (r"中级|intermediate", "中级"),
    (r"初级|junior|entry", "初级"),
)

#: 常见专业关键词（用于从"计算机相关专业"这类描述里提取）
_MAJOR_KEYWORDS: tuple[str, ...] = (
    "计算机科学与技术",
    "计算机",
    "软件工程",
    "人工智能",
    "数据科学",
    "大数据",
    "信息安全",
    "网络工程",
    "物联网",
    "电子信息",
    "通信工程",
    "自动化",
    "控制工程",
    "电子工程",
    "微电子",
    "数学与应用数学",
    "统计学",
    "应用数学",
    "计算数学",
    "模式识别",
    "智能科学与技术",
    "机器人工程",
    "光电",
    "物理",
    "机械",
)

#: 学校名识别
_SCHOOL_RE = re.compile(
    r"([\u4e00-\u9fff]{2,16}(?:大学|学院|学校|职业技术学院|高等专科学校))"
    r"|([A-Z][A-Za-z\.\s]{2,40}(?:University|College|Institute|School))"
)

#: 姓名标签
_NAME_LABEL_RE = re.compile(r"(?:姓名|名字|Name)\s*[：:]\s*([^\s，,。;；|]{2,20})", re.IGNORECASE)

#: 中文姓名：2~4 个汉字，且不含这些明显不是名字的字
_NAME_CN_RE = re.compile(r"^[\u4e00-\u9fff]{2,4}$")
_NAME_STOPWORDS = ("简历", "个人", "求职", "应聘", "基本", "信息", "联系", "方式", "电话", "邮箱", "地址", "意向")

#: 机构后缀：命中这些后缀的"2~4 字中文"是单位名，不是人名。
#: 典型误判：简历头部第一行只写了「某某大学」，不拦就会把学校当成姓名。
_NAME_ORG_SUFFIXES = (
    "大学",
    "学院",
    "学校",
    "校区",
    "中学",
    "高中",
    "初中",
    "小学",
    "职业学院",
    "职业技术学院",
    "公司",
    "集团",
    "科技",
    "有限",
    "研究所",
    "研究院",
    "实验室",
)

#: 英文姓名
_NAME_EN_RE = re.compile(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}$")

#: 否定/非特指语境词。出现在学校名里说明这句话不是"某个具体学校"，
#: 例如「没有学校信息」会被 ``_SCHOOL_RE`` 匹配出「没有学校」。
#:
#: 注意这里只放**多字词组**，不放单字（如"无"）——
#: 「无锡职业技术学院」是真实学校名，按单字拦会误杀。
_SCHOOL_STOPWORDS = (
    "没有",
    "不限",
    "任何",
    "什么",
    "哪个",
    "该校",
    "本校",
    "贵校",
    "学校信息",
    "大学信息",
    "学院信息",
)

#: 年份要求：「3年以上」「3-5年」「1~2年经验」
_YEARS_REQ_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(\d{1,2})\s*(?:[-~至到])\s*(\d{1,2})\s*年"),
    re.compile(r"(\d{1,2})\s*年(?:以上|及以上)"),
    re.compile(r"(?:工作|开发|相关|从业)(?:经验|经历)\s*(?:不少于|至少|满)?\s*(\d{1,2})\s*年"),
    re.compile(r"(\d{1,2})\s*年(?:以上)?(?:工作|开发|相关)?(?:经验|经历)"),
)

#: 「接受应届」「经验不限」
_NO_EXPERIENCE_RE = re.compile(r"应届|无经验|经验不限|不限经验|欢迎应届|在校生")

#: 毕业年份：「2027年毕业」「2027届」「预计2026年6月毕业」
_GRAD_RE = re.compile(r"(?:预计)?((?:19|20)\d{2})\s*年?\s*(?:6月|七月|七月|毕业|届)")

#: 「专业：xxx」
_MAJOR_LABEL_RE = re.compile(r"(?:专业|所学专业|主修)\s*[：:]\s*([^\s，,。;；|]{2,24})")


def find_majors(text: str, *, limit: int = 6) -> list[str]:
    """从文本里提取专业关键词。

    `_MAJOR_KEYWORDS` 里存在包含关系（"计算机" 是 "计算机科学与技术" 的子串），
    直接用 ``in`` 判断会导致「计算机科学与技术」被同时记成两个专业。
    所以这里用**长词优先 + 区间占用**的方式，与技能扫描保持同一套思路。
    """
    if not text:
        return []

    occupied = bytearray(len(text))
    found: list[tuple[int, str]] = []

    for keyword in sorted(_MAJOR_KEYWORDS, key=len, reverse=True):
        idx = text.find(keyword)
        while idx >= 0:
            end = idx + len(keyword)
            if not any(occupied[idx:end]):
                for i in range(idx, end):
                    occupied[i] = 1
                found.append((idx, keyword))
            idx = text.find(keyword, idx + 1)

    found.sort(key=lambda item: item[0])
    return [keyword for _, keyword in found][:limit]


def detect_seniority(text: str) -> str:
    """识别招聘层级（实习 / 校招 / 高级 …）。识别不到返回空串。"""
    sample = text[:400]  # 层级信息基本都在开头
    for pattern, label in _SENIORITY_RULES:
        if re.search(pattern, sample, re.IGNORECASE):
            return label
    return ""


def _strip_list_prefix(text: str) -> str:
    """剥掉短语开头的列表编号（``1. `` / ``2、`` / ``（3）``）。

    场景：学历短语从 "1. 本科及以上学历，计算机相关专业" 里截出来时，
    会把行首的编号一起带进来。展示成「1. 本科及以上学历」略显多余。
    """
    return re.sub(r"^\(?\d{1,2}\)?\s*[.、)）]?\s*", "", text or "").strip()


def detect_education_min(text: str) -> tuple[str, int]:
    """JD 视角：识别**最低**学历要求。

    为什么要取最小？因为 JD 常写「本科及以上」，包含"本科"与"硕士"两个
    可识别词。要求的是**本科**（硕士当然也满足），取最小才是对的答案。
    """
    if not text:
        return "", 0

    # 「学历不限」优先判定，避免被后面的其他词带偏
    if "学历不限" in text or "不限学历" in text:
        return "不限", 0

    found: list[tuple[str, int]] = []
    for word, level in EDUCATION_LEVELS.items():
        idx = text.find(word)
        if idx >= 0:
            found.append((word, level))

    if not found:
        return "", 0

    # 取最低等级；同等级取更早出现的那个词，保证结果稳定
    found.sort(key=lambda item: (item[1], text.find(item[0])))
    word, level = found[0]
    idx = text.find(word)
    # 只截短语级上下文（"本科及以上学历"），而不是整行，并剥掉行首编号
    return _strip_list_prefix(context_phrase(text, idx, len(word))), level


def detect_education_max(text: str) -> tuple[str, int]:
    """简历视角：识别**最高**学历。

    简历里「本科在读，硕士已录取」这类写法不常见，但「本科」+「硕士」
    同时出现是有的（写了本科经历和硕士经历）。简历要取最高。
    """
    if not text:
        return "", 0

    found: list[tuple[str, int]] = []
    for word, level in EDUCATION_LEVELS.items():
        idx = text.find(word)
        if idx >= 0:
            found.append((word, level))

    if not found:
        return "", 0

    found.sort(key=lambda item: (-item[1], text.find(item[0])))
    word, level = found[0]
    idx = text.find(word)
    return _strip_list_prefix(context_phrase(text, idx, len(word))), level


def detect_years_requirement(text: str) -> tuple[Optional[float], Optional[float]]:
    """JD 视角：识别经验年限要求。

    Returns:
        ``(最低年限, 最高年限)``。识别不到时最低年限为 None；
        明确写了"接受应届"时返回 ``(0.0, None)``。
    """
    if not text:
        return None, None

    # 先看有没有明确的区间
    for pattern in _YEARS_REQ_PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        groups = [g for g in m.groups() if g is not None]
        if len(groups) == 2:
            lo, hi = float(groups[0]), float(groups[1])
            if lo > hi:
                lo, hi = hi, lo
            return lo, hi
        if len(groups) == 1:
            return float(groups[0]), None

    if _NO_EXPERIENCE_RE.search(text):
        return 0.0, None

    return None, None


def find_school(text: str) -> str:
    """从文本里提取学校名。识别不到返回空串。

    难点是**否定语境**：``_SCHOOL_RE`` 只认「XX大学/XX学院」这种后缀，
    遇到「没有学校信息」会老老实实抓出「没有学校」——因为"没有"恰好是
    两个汉字，能填满 ``[\\u4e00-\\u9fff]{2,16}``。
    所以匹配之后还要过一道语义检查（``_SCHOOL_STOPWORDS``）。
    """
    for m in _SCHOOL_RE.finditer(text or ""):
        name = (m.group(1) or m.group(2) or "").strip()
        if not name:
            continue
        if any(word in name for word in _SCHOOL_STOPWORDS):
            continue
        return name
    return ""


def find_major(text: str) -> str:
    """从文本里提取专业名（优先取显式标注的，其次从专业词库里找）。"""
    if not text:
        return ""
    m = _MAJOR_LABEL_RE.search(text)
    if m:
        return m.group(1).strip()

    for keyword in _MAJOR_KEYWORDS:
        if keyword in text:
            return keyword
    return ""


def find_name(preamble: str, full_text: str = "") -> str:
    """从"头部信息区"提取姓名。

    简历的姓名有两个位置：显式标注（``姓名：张三``）和裸放在第一行。
    后者的误判风险很高（第一行可能是"个人简历"四个字、也可能是
    ``张三 | 男 | 1999.05`` 这种同行信息），所以做了三层防护：

    1. 优先找 ``姓名：`` 这样的显式标签；
    2. 对头部首行做"取第一个分隔片段"处理（``张三 | 男`` → ``张三``）；
       **英文姓名按整体匹配**（``John Smith`` 是一个名字，不能按空格切开）；
    3. 片段必须严格匹配"2~4 个汉字"或"英文姓名"，不含停用词，
       也不能以机构后缀结尾（``某某大学`` 是学校，不是人名）。

    只要遇到第一个"看起来像头部信息但不是姓名"的有效行就停止 ——
    继续往下看会把学校名、公司名也当成姓名。
    """
    for source in (preamble, full_text):
        if not source:
            continue
        m = _NAME_LABEL_RE.search(source)
        if m:
            return m.group(1).strip()

    if not preamble:
        return ""

    for line in preamble.split("\n"):
        candidate = line.strip()
        if not candidate:
            continue

        # 先按**显式分隔符**切出第一段："张三 | 男 | 1999" -> "张三"。
        # 注意这里**不能**把空白也算作分隔符，否则英文姓名
        # "John Smith" 会被切成 "John"（丢掉了姓）。
        first = re.split(r"[|｜·,，]+", candidate)[0].strip()
        if not first:
            continue
        if any(word in first for word in _NAME_STOPWORDS):
            continue

        # 英文姓名允许含空格，整体匹配
        if _NAME_EN_RE.match(first):
            return first

        # 中文姓名不含空格，这时才按空白再切一刀
        cn_first = re.split(r"\s+", first)[0].strip()
        if not cn_first:
            continue
        if any(word in cn_first for word in _NAME_STOPWORDS):
            continue
        # 机构名不该被当成人名："某某大学" 长度虽然只有 4 个字，
        # 但后缀已经说明它是单位，而不是某个人。
        if any(cn_first.endswith(suffix) for suffix in _NAME_ORG_SUFFIXES):
            break
        if _NAME_CN_RE.match(cn_first):
            return cn_first

        # 首个有效行不是姓名，说明这份简历没有裸写的姓名，不必再往下找
        if len(candidate) < 40:
            break

    return ""


# ===========================================================================
# 二、技能定位与合并
# ===========================================================================

#: 技能出现位置 -> 可信度说明（会写进报告，让用户知道我们为什么信这一条）
_ORIGIN_LABEL: dict[str, str] = {
    "skill_section": "技能清单",
    "experience": "经历描述",
    "education": "教育背景",
    "summary": "自我评价",
    "personal": "基本信息",
    "other": "其他位置",
}


def origin_at(index: sec.SectionIndex, offset: int) -> str:
    """判断某个字符偏移落在哪类章节里。"""
    for section in index.sections:
        if section.start_offset <= offset < section.end_offset:
            if section.kind == sec.KIND_SKILL:
                return "skill_section"
            if section.kind in sec.EXPERIENCE_KINDS:
                return "experience"
            if section.kind == sec.KIND_EDUCATION:
                return "education"
            if section.kind == sec.KIND_SUMMARY:
                return "summary"
            if section.kind == sec.KIND_PERSONAL:
                return "personal"
            return "other"
    # 落在任何章节之前 => 头部信息区
    first_start = index.sections[0].start_offset if index.sections else 10**9
    return "personal" if offset < first_start else "other"


def _line_at(text: str, offset: int) -> str:
    """取某个偏移所在的整行文本（已去掉首尾空白）。"""
    if not text:
        return ""
    start = text.rfind("\n", 0, offset) + 1
    end = text.find("\n", offset)
    if end < 0:
        end = len(text)
    return text[start:end].strip()


#: 子句分隔符。注意**故意不含顿号 `、`** ——
#: 顿号表示并列（"精通 Python、PyTorch"），并列项的强度应当继承前面的修饰词，
#: 按顿号切会把 "PyTorch" 变成孤立片段，反而丢掉"精通"这个信息。
_CLAUSE_SEP = "，,；;。！？!?：:"


def _clause_at(text: str, offset: int) -> str:
    """取某个偏移所在的**最小子句**。

    这是 demand 判定的关键改进。原先是按"整行"判定，会出现这种问题::

        熟悉 RAG 技术栈，了解向量数据库（FAISS / Milvus）的使用

    整行含有"熟悉"，于是整行的技能（含 FAISS、Milvus）全被判成 must。
    但这句话的真实意思是：RAG 是硬性要求，向量数据库是"了解"（次要）。

    按子句切分之后，"了解向量数据库（FAISS / Milvus）的使用" 独立成句，
    没有硬性标记词，正确落入 nice。
    """
    if not text:
        return ""

    line_start = text.rfind("\n", 0, offset) + 1
    line_end = text.find("\n", offset)
    if line_end < 0:
        line_end = len(text)

    lo = line_start
    for i in range(offset - 1, line_start - 1, -1):
        if text[i] in _CLAUSE_SEP:
            lo = i + 1
            break

    hi = line_end
    for i in range(offset, line_end):
        if text[i] in _CLAUSE_SEP:
            hi = i
            break

    return text[lo:hi].strip()


def judge_demand(line: str, section_kind: str) -> str:
    """判断一个子句里提到的技能属于「硬性(must)」还是「加分(nice)」。

    判定按四档优先级，**取第一个命中的那一档**（详见 skills 里的分层说明）：

      1. 强硬性词 ``STRONG_MUST_WORDS``（必须/必需/要求/精通/深入/扎实）→ must
      2. 显式加分词 ``EXPLICIT_NICE_WORDS``（加分/优先/更佳）→ nice
      3. 普通硬性词 ``SOFT_MUST_WORDS``（熟悉/掌握/熟练/具备/负责）→ must
      4. 普通加分词 ``GENERAL_NICE_WORDS``（了解/有所了解/…）→ nice
      5. 都没命中 → 看所在章节：加分项/自我评价/福利 → nice，其余 → must

    为什么第 2 档要压过第 3 档？
    因为「者优先」「加分项」这类词表达的是**整句的性质**，而"熟悉"只是它的
    搭配动词。把「熟悉 Kubernetes 者优先」判成硬要求，会让匹配分虚低、
    劝退本可以一试的岗位 —— 这比"高估要求"更伤用户。

    为什么第 1 档又压过第 2 档？
    「必须熟悉 Python，熟练掌握者优先」这种句子里，"必须"限定的对象是整个
    技术栈，优先只是补充说明。看到它就说明这是硬门槛，宁可高估也别低估。
    """
    if line:
        if any(word in line for word in STRONG_MUST_WORDS):
            return "must"
        if any(word in line for word in EXPLICIT_NICE_WORDS):
            return "nice"
        if any(word in line for word in SOFT_MUST_WORDS):
            return "must"
        if any(word in line for word in GENERAL_NICE_WORDS):
            return "nice"

    if section_kind == sec.KIND_BONUS:
        return "nice"
    if section_kind in (sec.KIND_SUMMARY, sec.KIND_BENEFIT):
        return "nice"
    return "must"


def locate_skills(
    text: str,
    index: sec.SectionIndex,
    normalizer: SkillNormalizer,
    *,
    with_demand: bool = False,
    with_level: bool = False,
) -> list[SkillItem]:
    """在文本里定位全部技能命中，并补齐上下文信息。

    Args:
        text: 规范化后的全文。
        index: 章节索引（用于判定 origin 与 demand）。
        normalizer: 技能归一化器。
        with_demand: 是否判定 JD 侧的 must / nice。
        with_level: 是否判定简历侧的掌握程度。

    Returns:
        按出现位置排序的 SkillItem 列表（同一技能的多次出现都会保留，
        由调用方决定是去重还是保留全部证据）。
    """
    items: list[SkillItem] = []
    for mention in normalizer.scan(text):
        origin = origin_at(index, mention.start)
        section = next(
            (s for s in index.sections if s.start_offset <= mention.start < s.end_offset),
            None,
        )
        section_kind = section.kind if section else sec.KIND_OTHER
        # 用"最小子句"而不是"整行"来做强度判定，避免一句话里的并列项被一刀切
        clause = _clause_at(text, mention.start)

        level_word: Optional[str] = None
        level_score: Optional[int] = None
        if with_level:
            level_word, level_score = normalizer.detect_level(text, mention.start, mention.end)

        item = SkillItem(
            canonical=mention.canonical,
            raw=mention.raw,
            category=mention.category,
            evidence=evidence_around(text, mention.start, mention.end),
            start=mention.start,
            end=mention.end,
            source=mention.source,
            origin=origin,
        )
        if with_level:
            item.level = level_word
            item.level_score = level_score
        if with_demand:
            item.demand = judge_demand(clause, section_kind)
        items.append(item)

    return items


def _as_name_list(value: Any) -> list[str]:
    """把模型返回的"技能名列表"统一成 ``list[str]``。

    模型返回 ``"skills": "Python, Java"``（字符串而非数组）是最常见的偏差之一，
    而且两个解析入口（JD / 简历）都要用，所以收口成一个函数，避免两处各写一遍
    再各自写错。
    """
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[,，、;；\n|]", value)
    elif isinstance(value, (list, tuple, set)):
        parts = [str(part) for part in value]
    else:
        return []
    return [part.strip() for part in parts if part.strip()]


#: 认可的招聘层级。模型可能编出"中级偏上"这类词，只接受白名单内的取值。
SENIORITY_LEVELS: tuple[str, ...] = ("实习", "校招", "初级", "中级", "高级", "专家")


def list_span_position(text: str, needle: str, *, start: int = 0) -> tuple[int, int] | None:
    """在文本里定位一个字符串（用于给模型抽取的结果补 offset）。

    大小写不敏感；定位不到时返回 None（调用方应把 offset 留空，
    而不是编一个假的 —— 证据位置错了比没有位置更糟）。
    """
    if not text or not needle:
        return None
    idx = text.lower().find(needle.lower(), start)
    if idx < 0:
        return None
    return idx, idx + len(needle)


def merge_llm_skills(
    existing: list[SkillItem],
    llm_skills: Iterable[str],
    text: str,
    normalizer: SkillNormalizer,
    *,
    with_demand: bool = False,
    with_level: bool = False,
    default_demand: Optional[str] = None,
) -> list[SkillItem]:
    """把大模型抽取出的技能合并进规则抽取的结果。

    合并规则：
    - 先用词典归一化（词典认识就用标准名，不认识就保留模型给的原名）；
    - 已经存在的技能不重复添加，但会把 ``source`` 从 dict 升级为 ``dict+llm``
      ——这表示"两边都认为它是技能"，可信度更高；
    - 新技能尝试在原文里定位，定位到就补上 offset 与证据，定位不到就留空。

    Args:
        default_demand: 给**新增的**模型技能指定的强度（``must`` / ``nice``）。
            为 None 时按原文所在行判定。需要这个参数是因为模型的
            ``bonus_skills`` 是它"读懂了语义"才分出来的类别 ——
            这些词往往压根不在 JD 原文里（比如 JD 写"有云原生经验"，
            模型补出 "Kubernetes"），按原文行判定只会落到默认的 must，
            把加分项错算成硬门槛。
    """
    by_canonical: dict[str, SkillItem] = {}
    order: list[str] = []
    for item in existing:
        if item.canonical not in by_canonical:
            by_canonical[item.canonical] = item
            order.append(item.canonical)
        else:
            # 同一技能多次出现：只把首次命中留作主记录，但来源标记为双重确认
            master = by_canonical[item.canonical]
            if master.source == "dict":
                master.source = "dict+llm"

    for raw_name in llm_skills:
        name = (raw_name or "").strip()
        if not name:
            continue
        canonical = normalizer.canonical_of(name) or name
        if canonical in by_canonical:
            master = by_canonical[canonical]
            if master.source == "dict":
                master.source = "dict+llm"
            continue

        span = list_span_position(text, name)
        item = SkillItem(
            canonical=canonical,
            raw=name,
            category=normalizer.category_of(canonical),
            source="llm",
            origin="other",
        )
        if span:
            item.start, item.end = span
            item.evidence = evidence_around(text, span[0], span[1])
            item.origin = "other"
            if with_level:
                item.level, item.level_score = normalizer.detect_level(text, span[0], span[1])
        if with_demand:
            if default_demand is not None:
                item.demand = default_demand
            else:
                item.demand = judge_demand(_line_at(text, item.start or 0), sec.KIND_REQUIREMENT)

        by_canonical[canonical] = item
        order.append(canonical)

    return [by_canonical[key] for key in order]


def dedupe_skill_names(items: Iterable[SkillItem]) -> list[str]:
    """取技能的标准名去重列表（保持出现顺序）。"""
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item.canonical not in seen:
            seen.add(item.canonical)
            out.append(item.canonical)
    return out


# ===========================================================================
# 三、JD 解析
# ===========================================================================

#: 岗位名称标签
_TITLE_LABEL_RE = re.compile(
    r"(?:岗位|职位|招聘岗位|应聘职位|职位名称|岗位名称)\s*[：:]\s*([^\n，,。;；|]{2,40})"
)
#: 公司名标签
_COMPANY_LABEL_RE = re.compile(r"(?:公司|企业|公司名称|所属公司)\s*[：:]\s*([^\n，,。;；|]{2,40})")

#: 职责条目里常见的软性描述（这些不算"技能要求"，但值得单列）
_SOFT_SKILL_RE = re.compile(
    r"沟通|团队|协作|主动|责任心|抗压|学习能力|表达|逻辑|自驱|owner|推动|好奇心|钻研"
)


def _detect_jd_title(text: str, index: sec.SectionIndex) -> str:
    """识别岗位名称。"""
    m = _TITLE_LABEL_RE.search(text)
    if m:
        return m.group(1).strip()

    # 没有显式标签时，看第一段非空文本
    for line in (index.preamble or text).split("\n"):
        line = line.strip()
        if not line or len(line) > 40:
            continue
        if line.isdigit():
            continue
        if any(ch in line for ch in "。！？；"):
            continue
        # 「XX公司招聘AI应用开发工程师」这类常见开头
        line = re.sub(r"^.*?公司\s*(?:招聘|诚招|急招)\s*", "", line).strip()
        return line
    return ""


def _classify_requirement(text: str, skills: list[str]) -> str:
    """给一条要求分类，供前端分组展示。"""
    if re.search(r"学历|本科|硕士|博士|大专|统招|应届|毕业", text):
        return "学历"
    # 经验类：`\d{1,2}\s*年` 与「经验/经历」之间的字数是**不定**的，
    # 有「3 年以上后端开发经验」（中间 6 个字），也有「3 年经验」。
    # 原先写死 ``.{0,4}`` 会让前者漏判成"其他"，所以放宽到十几个字，
    # 并用 ``[^\n，,。;；]`` 限制不跨子句 —— 否则会把整段里
    # 不相干的"年"和"经验"凑成一对。
    if re.search(
        r"\d{1,2}\s*年\s*(?:以上|及以上|以内|左右|多)?\s*[^\n，,。;；]{0,12}?(?:经验|经历)"
        r"|(?:工作|从业|开发|测试|运维|算法|项目|相关)\s*(?:经验|经历)"
        r"|工作经验|经验丰富",
        text,
    ):
        return "经验"
    if skills:
        return "技能"
    if _SOFT_SKILL_RE.search(text):
        return "软技能"
    if re.search(r"项目|系统|平台|产品|工程", text):
        return "项目"
    return "其他"


def build_requirements(
    index: sec.SectionIndex,
    normalizer: SkillNormalizer,
    fallback_text: str,
) -> list[Requirement]:
    """把「任职要求 / 加分项」章节拆成条目，并判定每条的性质。

    实现要点：**逐块处理**而不是逐句处理，因为一条要求经常写两行，
    按句切会把「熟悉 Python；有良好的编码习惯」拆成两条，其中一条
    没有技能信息，列表会变得又长又碎。

    Args:
        index: 章节索引。
        normalizer: 技能归一化器。
        fallback_text: 当 JD 完全没有"任职要求/加分项"小标题时，
            退化为在这段文本里按块筛出"看起来像要求"的条目。
    """
    requirements: list[Requirement] = []

    for kind in (sec.KIND_REQUIREMENT, sec.KIND_BONUS):
        for section in index.get(kind):
            for block in split_blocks(section.body):
                matched = normalizer.scan_canonicals(block)
                requirements.append(
                    Requirement(
                        text=shorten(block, 160),
                        kind=judge_demand(block, section.kind),
                        category=_classify_requirement(block, matched),
                        skills=matched,
                    )
                )

    # 有些 JD 压根不写"任职要求"标题，整篇就是一段一段的。这时退化处理：
    # 从全文的块里挑出"含技能或含要求标记词"的，作为要求条目。
    if not requirements:
        for block in split_blocks(fallback_text):
            matched = normalizer.scan_canonicals(block)
            if not matched and not any(w in block for w in MUST_MARKERS + NICE_MARKERS):
                continue
            requirements.append(
                Requirement(
                    text=shorten(block, 160),
                    kind=judge_demand(block, sec.KIND_REQUIREMENT),
                    category=_classify_requirement(block, matched),
                    skills=matched,
                )
            )

    return requirements


def _merge_soft_requirements(
    requirements: list[Requirement],
    llm_data: dict[str, Any] | None,
) -> list[Requirement]:
    """把模型抽出的「软技能」作为要求条目追加进去。

    为什么单独抽一个函数？因为软技能和硬技能的性质不同：
    它们**不能进 ``skills``**（那会污染技能匹配与 must/nice 分桶 ——
    "沟通能力强"和 "Python" 不是一回事），但它们确实是 JD 提出的一条要求，
    应该出现在"要求清单"里让用户看到。所以只进 ``requirements``。

    强度一律记为 ``nice``：软技能几乎不会成为简历筛选的硬门槛，
    记为 must 会让报告看起来比实际严格。
    """
    if not llm_data:
        return requirements

    existing_texts = {req.text for req in requirements}
    extra: list[Requirement] = []
    for name in _as_name_list(llm_data.get("soft_skills")):
        if name in existing_texts:
            continue
        existing_texts.add(name)
        extra.append(Requirement(text=name, kind="nice", category="软技能", skills=[]))

    return requirements + extra


def parse_jd(
    text: str,
    *,
    normalizer: SkillNormalizer | None = None,
    llm_data: dict[str, Any] | None = None,
) -> ParsedJD:
    """把 JD 原文解析成结构化对象。

    Args:
        text: JD 原文（可以是任意格式，会先做规范化）。
        normalizer: 技能归一化器，默认用全局词典。
        llm_data: 大模型抽取结果（可选）。会被消费的键与用途：

            ================  ==========================================
            ``title``         岗位名，规则抽不到时补空
            ``company``       公司名，规则抽不到时补空
            ``skills``        技能，**并集**合并进技能列表（判为 must）
            ``bonus_skills``  加分技能，**并集**合并（判为 nice）
            ``soft_skills``   软技能，只进 ``requirements``，不进技能列表
            ``education``     学历，规则抽不到时补空
            ``min_years``     年限，规则抽不到时补空
            ``domains``       行业方向，**并集**合并
            ``seniority``     招聘层级，规则抽不到时补空（白名单校验）
            ================  ==========================================

    Returns:
        ParsedJD。
    """
    nz = normalizer or DEFAULT_NORMALIZER
    clean = normalize_text(text or "")
    index = sec.split_sections(clean)

    # ---- 技能：全文扫描（JD 的技能在职责段落里也会出现）----
    skills = locate_skills(clean, index, nz, with_demand=True)

    if llm_data:
        skills = merge_llm_skills(
            skills, _as_name_list(llm_data.get("skills")), clean, nz, with_demand=True
        )
        # 加分项**单独合并一次**，并显式指定强度为 nice。
        #
        # 这里曾经漏过：提示词要求模型输出 bonus_skills、清洗层也认真处理了它，
        # 但解析层只消费了 skills，模型给出的加分项被整段丢掉 ——
        # 典型的三层接缝里"最后一公里没人接"。
        # 也不能把它混进 skills 一起合并：那样新技能会按原文行判定强度，
        # 而 bonus_skills 里的词常常压根不在 JD 原文中（模型是"读懂"才补出来的），
        # 定位不到就落到默认的 must，把加分项错算成硬门槛。
        skills = merge_llm_skills(
            skills,
            _as_name_list(llm_data.get("bonus_skills")),
            clean,
            nz,
            with_demand=True,
            default_demand="nice",
        )

    # ---- must / nice 分桶 ----
    # 同一个技能可能在 JD 里出现多次（职责里提一次、要求里提一次、
    # 加分项里又提一次）。这时要**取更强的那个判定**：
    # 只要有一处把它当作硬性要求，它就是硬性 —— 不能因为加分项里也提了就降级。
    must_set: dict[str, int] = {}
    nice_set: dict[str, int] = {}
    for pos, item in enumerate(skills):
        name = item.canonical
        if item.demand == "nice":
            if name not in must_set:
                nice_set.setdefault(name, pos)
        else:
            must_set.setdefault(name, pos)
            nice_set.pop(name, None)  # 升级为 must，从加分桶里摘掉

    must_have = sorted(must_set, key=lambda name: must_set[name])
    nice_to_have = sorted(nice_set, key=lambda name: nice_set[name])

    # ---- 基础字段 ----
    title = _detect_jd_title(clean, index)
    company_m = _COMPANY_LABEL_RE.search(clean)
    company = company_m.group(1).strip() if company_m else ""

    if llm_data:
        title = title or str(llm_data.get("title") or "").strip()
        company = company or str(llm_data.get("company") or "").strip()

    min_years, max_years = detect_years_requirement(clean)
    education, edu_level = detect_education_min(clean)
    # 层级优先从岗位名里看（最准），看不出来再退回全文开头
    seniority = detect_seniority(title) or detect_seniority(clean[:500])

    # ---- 模型补空：规则没抽到的标量字段，才用模型的结果 ----
    # 统一遵循「规则优先、模型补空」：规则能抽出来的一定更可信（它有原文位置），
    # 模型的强项是**规则抽不到**的表达（"有云原生经验""接受应届"这类没有关键词的写法）。
    if llm_data:
        if min_years is None:
            model_years = llm_data.get("min_years")
            # bool 是 int 的子类，必须显式排除 —— 模型返回 true 不能当成"1 年"
            if isinstance(model_years, (int, float)) and not isinstance(model_years, bool):
                min_years = float(model_years)

        if not education:
            model_education = str(llm_data.get("education") or "").strip()
            if model_education:
                # 复用同一套识别器把描述映射成等级，避免两套口径打架
                phrase, level = detect_education_min(model_education)
                education = phrase or model_education
                edu_level = level

        if not seniority:
            model_seniority = str(llm_data.get("seniority") or "").strip()
            # 只接受白名单内的层级，防止模型编出一个奇怪的词直接写进报告
            if model_seniority in SENIORITY_LEVELS:
                seniority = model_seniority

    # ---- 领域关键词：复用技能词典里的"领域方向"大类 ----
    domains = dedupe_skill_names([item for item in skills if item.category == CAT_DOMAIN])
    if llm_data:
        # 领域方向用**并集**：词典认识的方向保留，模型补充的追加在后面。
        # 这里不能再"规则优先"，因为两份来源的口径不同 ——
        # 词典只能认出已有词条，模型能读到"面向保险行业的智能核保"这种新方向。
        for extra in _as_name_list(llm_data.get("domains")):
            if extra not in domains:
                domains.append(extra)
        domains = domains[:12]

    # ---- 职责与要求条目 ----
    responsibilities = []
    for section in index.get(sec.KIND_DUTY):
        responsibilities.extend(split_blocks(section.body))
    responsibilities = [shorten(block, 160) for block in responsibilities][:30]

    # 要求章节的文本 = 任职要求 + 加分项 + 岗位职责（用于兜底）
    requirements_text = "\n".join(
        s.body for s in index.get(*sec.JD_REQUIREMENT_KINDS) if s.body
    )
    if not requirements_text:
        requirements_text = "\n".join(s.body for s in index.get(sec.KIND_DUTY) if s.body)

    requirements = build_requirements(index, nz, requirements_text or clean)
    requirements = _merge_soft_requirements(requirements, llm_data)

    return ParsedJD(
        title=title,
        company=company,
        seniority=seniority,
        min_years=min_years,
        max_years=max_years,
        education=education,
        education_level=edu_level,
        majors=find_majors(clean),
        must_have=must_have,
        nice_to_have=nice_to_have,
        skills=skills,
        domains=domains,
        requirements=requirements,
        responsibilities=responsibilities,
        raw_text=clean,
        char_count=len(clean),
    )


# ===========================================================================
# 四、简历解析
# ===========================================================================


@dataclass
class _RawEntry:
    """经历切分的中间产物（还没解析出公司/职位）。"""

    header: str = ""
    bullets: list[str] = field(default_factory=list)


#: 经历条目标题里的常见职位词（用来判断某个片段是不是"职位"）
_TITLE_HINTS = (
    "实习生",
    "工程师",
    "开发",
    "算法",
    "研究员",
    "负责人",
    "主管",
    "经理",
    "组长",
    "架构师",
    "设计师",
    "运营",
    "产品",
    "测试",
    "分析师",
    "intern",
    "engineer",
    "developer",
    "leader",
)

#: 条目标题的分隔符：| ｜ · / 以及连续 2 个以上空格或 Tab
_HEADER_SPLIT_RE = re.compile(r"\s*[|｜·•]\s*|\t+|\s{2,}")


def split_experience_entries(body_lines: list[str]) -> list[_RawEntry]:
    """把一段经历章节的正文切成若干条目。

    判定"这一行是条目标题"的依据（按可靠性从高到低）：

    1. 不是列表项，且含有日期区间 —— 这是最强的信号，
       简历里只有条目头会写时间；
    2. 不是列表项，且较短（<= 40 字）—— 次强的信号，
       对应「智能问答系统 后端开发」这种不带时间的写法；
    3. 其余情况视为条目的正文内容。

    一个容易踩的坑：有些简历的条目头写成两行（第一行项目名、第二行角色+时间）。
    这里用"标题续行合并"处理 —— 如果当前已经把标题行收下了、还没收到任何
    正文内容，就认为新来的非列表行是标题的续行，拼在一起。拼接到 80 字为止；
    超过 80 字说明这行其实是正文，直接归入条目内容。
    """
    entries: list[_RawEntry] = []
    current: _RawEntry | None = None

    for raw in body_lines:
        line = raw.strip()
        if not line:
            continue

        bullet = is_bullet(raw)
        has_date = find_date_range(line) is not None
        looks_like_header = (not bullet) and (has_date or len(line) <= 40)

        if looks_like_header:
            title_open = current is not None and bool(current.header) and not current.bullets
            if title_open:
                if len(current.header) < 80:
                    current.header = f"{current.header} {line}".strip()
                    continue
                # 标题已经够长了，这一行只能是正文
                current.bullets.append(line)
                continue
            if current is not None:
                entries.append(current)
            current = _RawEntry(header=line)
            continue

        if current is None:
            # 章节以正文开头（没有标题行），开一个无名条目
            current = _RawEntry(header="")

        current.bullets.append(strip_bullet(line) if bullet else line)

    if current is not None:
        entries.append(current)

    # 丢掉完全空的条目
    return [e for e in entries if e.header or e.bullets]


def parse_entry_header(header: str) -> dict[str, Any]:
    """从条目标题里解析出「组织 / 角色 / 时间」。

    支持的常见格式::

        腾讯科技有限公司 | 后端开发实习生 | 2024.07-2024.10
        2023.09 - 2024.03  智能问答系统  后端负责人
        某某科技  算法工程师  2022.06-2024.08
        智能问答系统（后端开发）
    """
    result: dict[str, Any] = {"org": "", "title": "", "period": "", "date": None}
    if not header:
        return result

    date = find_date_range(header)
    rest = header
    if date:
        result["period"] = date.label
        result["date"] = date
        # 把日期片段从标题里摘掉，剩下的才是组织与角色
        if date.raw:
            rest = rest.replace(date.raw, " ")

    parts = [p.strip(" -—·|") for p in _HEADER_SPLIT_RE.split(rest) if p.strip()]
    parts = [p for p in parts if p]
    if not parts:
        return result

    # 只有一个片段时，试着把括号里的内容当作角色
    if len(parts) == 1:
        only = parts[0]
        m = re.match(r"^(.+?)\s*[（(](.+?)[）)]$", only)
        if m:
            result["org"] = m.group(1).strip()
            result["title"] = m.group(2).strip()
        else:
            result["org"] = only
        return result

    # 多个片段：用职位关键词判断谁是角色
    title_part = ""
    org_part = ""
    for part in parts:
        if not title_part and any(hint in part.lower() for hint in _TITLE_HINTS):
            title_part = part
        elif not org_part:
            org_part = part

    if not org_part:
        org_part = parts[0]
    if not title_part:
        # 没找到关键词时，按位置猜：第一个是组织、第二个是角色
        title_part = parts[1] if len(parts) > 1 and parts[1] != org_part else ""

    result["org"] = org_part
    result["title"] = title_part
    return result


def merge_month_ranges(ranges: list[tuple[int, int]]) -> int:
    """合并重叠的时间区间，返回总月数。

    参数中每个元素是 ``(起始月序号, 结束月序号)``，月序号用
    ``年份 * 12 + 月份`` 表示（便于直接做差值）。

    为什么要合并？因为「2024.01-2024.06 实习 A」和「2024.03-2024.09 实习 B」
    如果直接相加是 12 个月，但实际只覆盖 2024.01 到 2024.09 共 9 个月。
    简历上把并行经历都写上很常见，不合并会显著虚高年限。

    另外会**过滤掉异常区间**（结束早于开始、或长度超过 30 年的），
    避免简历里写错的日期把总年限算飞。
    """
    if not ranges:
        return 0

    valid: list[tuple[int, int]] = []
    for start, end in ranges:
        if end < start:
            continue
        if end - start > 12 * 30:
            continue
        valid.append((start, end))

    if not valid:
        return 0

    valid.sort()
    merged = 0
    cur_start, cur_end = valid[0]
    for start, end in valid[1:]:
        if start <= cur_end + 1:
            cur_end = max(cur_end, end)
        else:
            merged += cur_end - cur_start + 1
            cur_start, cur_end = start, end
    merged += cur_end - cur_start + 1
    return merged


def _to_month_index(year: int, month: int | None) -> int:
    """把「年 + 月」转成连续月序号（便于做区间运算）。"""
    return year * 12 + (month or 1)


def build_experience_items(
    index: sec.SectionIndex,
    normalizer: SkillNormalizer,
) -> list[ExperienceItem]:
    """把全部经历类章节解析成 ExperienceItem 列表。"""
    items: list[ExperienceItem] = []
    order = 0

    kind_map = {
        sec.KIND_WORK: "work",
        sec.KIND_INTERNSHIP: "internship",
        sec.KIND_PROJECT: "project",
        sec.KIND_RESEARCH: "project",
        sec.KIND_CAMPUS: "campus",
    }

    for section in index.sections:
        if section.kind not in sec.EXPERIENCE_KINDS:
            continue
        kind = kind_map[section.kind]
        for raw_entry in split_experience_entries(section.body_lines):
            parsed = parse_entry_header(raw_entry.header)
            bullets = [b for b in raw_entry.bullets if b]
            full = "\n".join([raw_entry.header, *bullets])

            date: DateRange | None = parsed["date"]
            duration: Optional[int] = None
            start_year = end_year = None
            start_month = end_month = None
            ongoing = False
            if date:
                start_year, start_month = date.start_year, date.start_month
                end_year, end_month = date.end_year, date.end_month
                duration = date.months
                ongoing = date.ongoing

            items.append(
                ExperienceItem(
                    kind=kind,  # type: ignore[arg-type]
                    org=parsed["org"],
                    title=parsed["title"],
                    period=parsed["period"],
                    start_year=start_year,
                    start_month=start_month,
                    end_year=end_year,
                    end_month=end_month,
                    ongoing=ongoing,
                    duration_months=duration,
                    bullets=bullets,
                    skills=normalizer.scan_canonicals(full),
                    quantified=sum(1 for b in bullets if has_quantified(b)),
                    index=order,
                )
            )
            order += 1

    return items


def parse_resume(
    text: str,
    *,
    normalizer: SkillNormalizer | None = None,
    llm_data: dict[str, Any] | None = None,
) -> ParsedResume:
    """把简历原文解析成结构化对象。

    年限的口径说明
    -------------
    - ``work_years``：正式工作经历（kind=work）合并后的长度；
    - ``internship_months``：实习经历（kind=internship）合并后的长度；
    - ``total_years``：**全部经历**（工作 + 实习 + 项目 + 科研 + 校园）合并后的长度，
      并额外把实习按 1:1 计入（不折算）——因为校招场景下实习价值与工作相当，
      而 JD 的年限要求对校招生本来也是宽松解释的。

    这个口径是显式的、写在文档里的，而不是藏在代码里的魔法系数 ——
    匹配系统里"分数怎么来的"必须能被解释。

    Args:
        llm_data: 大模型抽取结果（可选）。会被消费的键与用途：

            ===================  ==========================================
            ``name``            姓名，**优先取模型值**（见下方注释）
            ``education``       学历，规则抽不到时补空
            ``school``          学校，规则抽不到时补空
            ``major``           专业，规则抽不到时补空
            ``graduation_year`` 毕业年份，规则抽不到时补空
            ``skills``          技能，**并集**合并（不覆盖规则结果）
            ``highlights``      亮点句，去重后追加
            ``total_years``     **极窄兜底**：仅当规则层零条带日期经历时采用
            ===================  ==========================================
    """
    nz = normalizer or DEFAULT_NORMALIZER
    clean = normalize_text(text or "")
    index = sec.split_sections(clean)

    # ---- 技能：全文扫描，并在技能清单等位置做程度分级 ----
    all_skills = locate_skills(clean, index, nz, with_level=True)

    if llm_data:
        all_skills = merge_llm_skills(
            all_skills, _as_name_list(llm_data.get("skills")), clean, nz, with_level=True
        )

    skill_names = dedupe_skill_names(all_skills)
    skill_groups: dict[str, list[str]] = {}
    for name in skill_names:
        skill_groups.setdefault(nz.category_of(name), []).append(name)

    # ---- 基本信息 ----
    education_section = index.first(sec.KIND_EDUCATION)
    edu_source = (education_section.text if education_section else "") or clean
    education, edu_level = detect_education_max(edu_source)
    if not education:
        education, edu_level = detect_education_max(clean)

    school = find_school(index.text_of(sec.KIND_EDUCATION)) or find_school(index.preamble) or find_school(clean)

    name = find_name(index.preamble, clean)
    if llm_data:
        # 姓名这一项**优先信任模型**，与其它字段的"规则优先、模型补空"相反。
        # 原因：规则层的依据只有头部几行文本，遇到「赵六」这种被截断/缩写
        # 的写法无能为力；而模型读的是全文，能给出更完整的结果（如「赵六六」）。
        # 姓名是单值字段，取模型值不会像列表字段那样引入噪声。
        # 模型没给值（空串/None）时，保留规则层的结果。
        name = str(llm_data.get("name") or "").strip() or name

    major = find_major(index.text_of(sec.KIND_EDUCATION)) or find_major(index.preamble) or find_major(clean)

    # ---- 模型补空：规则抽不到的标量字段才用模型结果 ----
    # （姓名是唯一的例外，见上面的说明。）
    if llm_data:
        if not major:
            major = str(llm_data.get("major") or "").strip()
        if not school:
            # 扫描版 PDF / 图片简历里，学校往往整段识别不出来，这时模型是唯一线索
            school = str(llm_data.get("school") or "").strip()
        if not education:
            model_education = str(llm_data.get("education") or "").strip()
            if model_education:
                phrase, level = detect_education_max(model_education)
                education = phrase or model_education
                edu_level = level

    # ---- 毕业年份 ----
    graduation_year: Optional[int] = None
    grad_m = _GRAD_RE.search(clean)
    if grad_m:
        graduation_year = int(grad_m.group(1))
    else:
        years = find_years(edu_source)
        if years:
            graduation_year = max(years)
        else:
            single = find_single_date(edu_source)
            if single:
                graduation_year = single[0]

    if graduation_year is None and llm_data:
        model_year = llm_data.get("graduation_year")
        if isinstance(model_year, int) and not isinstance(model_year, bool):
            graduation_year = model_year

    # ---- 经历 ----
    experiences = build_experience_items(index, nz)

    # ---- 年限：分类型合并重叠区间 ----
    now_index = _to_month_index(*_now_ym())
    work_ranges: list[tuple[int, int]] = []
    intern_ranges: list[tuple[int, int]] = []
    all_ranges: list[tuple[int, int]] = []

    for exp in experiences:
        if exp.start_year is None:
            continue
        start_idx = _to_month_index(exp.start_year, exp.start_month)
        end_idx = now_index if exp.ongoing else _to_month_index(exp.end_year or exp.start_year, exp.end_month)
        if end_idx < start_idx:
            continue
        span = (start_idx, end_idx)
        all_ranges.append(span)
        if exp.kind == "work":
            work_ranges.append(span)
        elif exp.kind == "internship":
            intern_ranges.append(span)

    work_months = merge_month_ranges(work_ranges)
    intern_months = merge_month_ranges(intern_ranges)
    total_months = merge_month_ranges(all_ranges)

    total_years = round(total_months / 12, 1)

    # 年限的兜底：**只有规则层一条带日期的经历都没找到时**，才接受模型的整体估计。
    # 这是一种很窄的兜底，但它解决一个真实问题 ——
    # 有些简历正文完全不带时间（或时间以图片形式存在），规则层会得出"0 年"，
    # 那是个明确的错误答案，会直接把这个维度的分数打到 0。
    # 常规情况下年限完全由确定性算法给出 ——
    # "同一份简历跑两次必须得到同一个分数"是这个项目的硬约束，不能被模型破坏。
    if total_months == 0 and llm_data:
        model_years = llm_data.get("total_years")
        if isinstance(model_years, (int, float)) and not isinstance(model_years, bool) and model_years > 0:
            total_years = round(float(model_years), 1)
            logger.info("规则层未识别到任何带日期的经历，年限采用模型估计值 %s 年", total_years)

    # ---- 应届判定 ----
    current_year = _now_ym()[0]
    degree_expected = bool(
        re.search(r"应届|在读|在校|预计.*毕业|20\d{2}\s*届", clean[:1200])
        or (graduation_year is not None and graduation_year >= current_year)
    )

    # ---- 亮点句：含量化成果的条目优先 ----
    highlights: list[str] = []
    for exp in experiences:
        for bullet in exp.bullets:
            if has_quantified(bullet):
                highlights.append(shorten(bullet, 120))
    if len(highlights) < 5:
        for section in index.get(sec.KIND_SUMMARY):
            for block in split_blocks(section.body):
                if has_quantified(block) or len(block) >= 12:
                    highlights.append(shorten(block, 120))
    highlights = highlights[:10]

    if llm_data:
        # 模型给的亮点句是**照抄原文**的（提示词里明确要求不要改写），
        # 所以可以安全地并入。按原句截断后去重再追加到末尾 ——
        # 规则层挑出来的是"带数字的条目"，模型补的往往是"不带数字但含金量高"的句子，
        # 两者互补。
        for item in _as_name_list(llm_data.get("highlights")):
            text_item = shorten(item, 120)
            if text_item and text_item not in highlights:
                highlights.append(text_item)
        highlights = highlights[:10]

    domains = dedupe_skill_names([item for item in all_skills if item.category == CAT_DOMAIN])

    return ParsedResume(
        name=name,
        education=education,
        education_level=edu_level,
        school=school,
        major=major,
        graduation_year=graduation_year,
        degree_expected=degree_expected,
        total_years=total_years,
        work_years=round(work_months / 12, 1),
        internship_months=round(intern_months, 1),
        skills=all_skills,
        skill_names=skill_names,
        skill_groups=skill_groups,
        experiences=experiences,
        highlights=highlights,
        domains=domains,
        raw_text=clean,
        char_count=len(clean),
    )


def _now_ym() -> tuple[int, int]:
    """当前年月的便捷封装（抽出来便于测试替换）。"""
    from datetime import datetime

    now = datetime.now()
    return now.year, now.month


__all__ = [
    "parse_jd",
    "parse_resume",
    "locate_skills",
    "merge_llm_skills",
    "judge_demand",
    "SENIORITY_LEVELS",
    "detect_seniority",
    "detect_education_min",
    "detect_education_max",
    "detect_years_requirement",
    "find_school",
    "find_major",
    "find_name",
    "split_experience_entries",
    "parse_entry_header",
    "merge_month_ranges",
    "build_experience_items",
    "build_requirements",
    "dedupe_skill_names",
    "origin_at",
]
