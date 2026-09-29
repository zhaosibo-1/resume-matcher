"""章节识别：把一份简历 / JD 切成有语义的小节。

为什么这一层单独拆出来？
----------------------
因为后面的所有判断都依赖"这段话属于哪个板块"：

- 「技能」板块里的 "Python" 是**自述技能**，可以直接采信；
- 「自我评价」里的 "熟悉 Python" 就要打折看，因为那是自我描述而非罗列；
- 「工作经历」里的 "Python" 说明是**用过的**，含金量高于技能清单。

如果只做全文关键词匹配，这三者的权重会完全一样，匹配报告就失去了分辨力。

识别策略（纯规则，可解释）
--------------------------
一行被认为是章节标题，需要同时满足：

1. 独占一行（或形如 ``教育背景：沈阳大学`` 的"标题+内容"同行形式）；
2. 长度较短（清理装饰后不超过 18 个字符）；
3. 不含句末标点（``。！？；``）与逗号 —— 带这些的多半是正文；
4. 命中已知的章节模式表。

第 4 条用的是**全匹配**而不是包含匹配，这是刻意的：
中文里"工作"两个字出现在正文里的概率极高（"负责工作流的搭建"），
只要用包含匹配就一定会误判。宁可漏识别，不可误切分。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .textutil import clean_heading

# ---------------------------------------------------------------------------
# 章节类型
# ---------------------------------------------------------------------------

KIND_PERSONAL = "personal"
KIND_INTENT = "intent"
KIND_EDUCATION = "education"
KIND_WORK = "work"
KIND_INTERNSHIP = "internship"
KIND_PROJECT = "project"
KIND_RESEARCH = "research"
KIND_CAMPUS = "campus"
KIND_SKILL = "skill"
KIND_AWARD = "award"
KIND_CERT = "cert"
KIND_SUMMARY = "summary"
KIND_DUTY = "duty"
KIND_REQUIREMENT = "requirement"
KIND_BONUS = "bonus"
KIND_BENEFIT = "benefit"
KIND_OTHER = "other"

#: 这些类型的章节标题是"经历类"的 —— 解析器会按经历条目去切分它们
EXPERIENCE_KINDS: tuple[str, ...] = (
    KIND_WORK,
    KIND_INTERNSHIP,
    KIND_PROJECT,
    KIND_RESEARCH,
    KIND_CAMPUS,
)

#: 这些类型是 JD 侧的"要求类"章节 —— 决定技能算 must 还是 nice
JD_REQUIREMENT_KINDS: tuple[str, ...] = (KIND_REQUIREMENT, KIND_BONUS)


# ---------------------------------------------------------------------------
# 章节模式表
# ---------------------------------------------------------------------------

#: (正则, 章节类型, 标准标题)
#:
#: 顺序有讲究：**具体的规则必须排在宽泛的规则前面**。
#: 例如「实习经历」必须排在「工作经历」之前判断（虽然两者的正则不重叠，
#: 但保持这个习惯能避免以后加规则时踩坑）。
_SECTION_RULES: tuple[tuple[str, str, str], ...] = (
    # ---------------- 简历侧 ----------------
    (r"(个人)?(基本信息|个人信息|基本资料|个人资料|个人信息表)", KIND_PERSONAL, "基本信息"),
    (r"(联系方式|联络方式|联系方式及邮箱)", KIND_PERSONAL, "联系方式"),
    (r"(求职|应聘)(意向|目标|方向|岗位)", KIND_INTENT, "求职意向"),
    (r"(教育|学历)(背景|经历|信息|情况|履历)", KIND_EDUCATION, "教育背景"),
    (r"(教育|学历)背景及经历", KIND_EDUCATION, "教育背景"),
    (r"(实习|实践)(经历|经验|背景)", KIND_INTERNSHIP, "实习经历"),
    (r"(实习|实践)经历及项目", KIND_INTERNSHIP, "实习经历"),
    (r"(工作|职业)(经历|经验|履历|背景)", KIND_WORK, "工作经历"),
    (r"(项目)(经历|经验|实践|介绍|描述|成果)", KIND_PROJECT, "项目经历"),
    (r"(科研|学术|研究)(经历|成果|项目)", KIND_RESEARCH, "科研经历"),
    (r"(校园|社团|学生)(经历|工作|活动|实践)", KIND_CAMPUS, "校园经历"),
    (r"(专业|技术|IT|计算机|软件)?(技能|特长|能力|技能栈|技术栈)(清单|特长|描述|列表|掌握情况)?", KIND_SKILL, "专业技能"),
    # 注意：「荣誉奖项」这种"两个近义词拼起来"的标题必须显式列出。
    # 因为匹配用的是 fullmatch，「(荣誉|获奖|奖项)」只能吃掉前两个字，
    # 剩下的「奖项」没处安放，整条规则就失配了 —— 这种"部分匹配"漏识别
    # 不会报错，只会让该章节被并入上一节，属于很隐蔽的一类问题。
    (r"(荣誉奖项|荣誉|获奖情况|获奖经历|获奖|奖项|奖励|所获荣誉)", KIND_AWARD, "荣誉奖项"),
    (r"(证书|资格证书|技能证书|认证|证书情况|证书列表)", KIND_CERT, "证书"),
    (r"(自我评价|个人评价|自我描述|个人总结|自我介绍|个人简介|工作总结|个人优势|自我认知)", KIND_SUMMARY, "自我评价"),
    # ---------------- JD 侧 ----------------
    (r"(岗位|职位|工作)(职责|描述|内容|内容描述|说明)", KIND_DUTY, "岗位职责"),
    (r"(任职|岗位|职位)(要求|资格|条件)", KIND_REQUIREMENT, "任职要求"),
    (r"(我们|公司)?(希望|期待|寻找|要求)(你|您)?(有|具备|是|拥有)?", KIND_REQUIREMENT, "任职要求"),
    (r"(能力|素质)(要求|模型)", KIND_REQUIREMENT, "任职要求"),
    (r"(加分|优先)(项|条件|项及要求|加分点)", KIND_BONUS, "加分项"),
    (r"(福利待遇|薪酬福利|薪资福利|福利|待遇|薪酬|薪资|我们提供|你将获得|团队介绍|公司介绍|关于我们)", KIND_BENEFIT, "福利待遇"),
)

_COMPILED_RULES: tuple[tuple[re.Pattern[str], str, str], ...] = tuple(
    (re.compile(rf"^(?:{pattern})$"), kind, title) for pattern, kind, title in _SECTION_RULES
)

#: 章节标题的长度上限（清理装饰之后）
MAX_TITLE_LEN = 18

#: 章节标题行本身不能太长（避免把一整段正文误当成"标题+内容"同行形式）
MAX_HEADING_LINE_LEN = 40

#: 去掉标题尾部的括号补充，如「任职要求（急招）」「技能清单(选填)」
_TRAILING_PAREN_RE = re.compile(r"[（(][^）)]{0,14}[）)]\s*$")


def _strip_trailing_paren(title: str) -> str:
    """反复剥掉标题尾部的括号补充。

    需要循环，因为像「任职要求（杭州）（急招）」这种有多个括号。
    """
    for _ in range(3):
        stripped = _TRAILING_PAREN_RE.sub("", title).strip()
        if stripped == title:
            break
        title = stripped
    return title


def match_heading(line: str) -> tuple[str, str] | None:
    """判断一行是否是章节标题。

    Args:
        line: 原始行（可以带 Markdown 井号、列表符号等装饰）。

    Returns:
        ``(章节类型, 标准标题)``；不是标题则返回 None。
    """
    raw = (line or "").strip()
    if not raw or len(raw) > MAX_HEADING_LINE_LEN:
        return None

    # 章节标题前面一般不会有列表符号；带了说明是正文条目
    stripped_marker = raw.lstrip("-*•·▪◦●○◆◇■□ ")
    if stripped_marker != raw and not raw.startswith("#"):
        return None

    title = _strip_trailing_paren(clean_heading(raw))
    if not title or len(title) > MAX_TITLE_LEN:
        return None

    # 句末标点/逗号出现即视为正文
    if any(ch in title for ch in "。！？；!?;，,"):
        return None
    # 纯数字或纯标点不是标题
    if not re.search(r"[\u4e00-\u9fffA-Za-z]", title):
        return None

    for pattern, kind, canonical in _COMPILED_RULES:
        if pattern.match(title):
            return kind, canonical
    return None


# ---------------------------------------------------------------------------
# Section 数据结构
# ---------------------------------------------------------------------------


@dataclass
class Section:
    """一个章节。"""

    kind: str
    title: str
    start_line: int
    end_line: int
    start_offset: int
    end_offset: int
    heading_line: str = ""
    lines: list[str] = field(default_factory=list)

    @property
    def body_lines(self) -> list[str]:
        """去掉标题行之后的正文行。"""
        if self.heading_line and self.lines and self.lines[0] is self.heading_line:
            return self.lines[1:]
        if self.lines and self.lines[0].strip() == self.heading_line.strip():
            return self.lines[1:]
        return list(self.lines)

    @property
    def text(self) -> str:
        """章节全文（含标题行）。"""
        return "\n".join(self.lines).strip()

    @property
    def body(self) -> str:
        """章节正文（不含标题行）。"""
        return "\n".join(self.body_lines).strip()

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "title": self.title,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "char_count": len(self.text),
        }


@dataclass
class SectionIndex:
    """一份文本的章节索引。"""

    sections: list[Section] = field(default_factory=list)
    preamble: str = ""
    preamble_lines: list[str] = field(default_factory=list)
    total_lines: int = 0

    def kinds(self) -> list[str]:
        """出现过的章节类型（去重，保持顺序）。"""
        out: list[str] = []
        for section in self.sections:
            if section.kind not in out:
                out.append(section.kind)
        return out

    def get(self, *kinds: str) -> list[Section]:
        """取指定类型的所有章节。"""
        return [s for s in self.sections if s.kind in kinds]

    def first(self, *kinds: str) -> Section | None:
        """取指定类型的第一个章节。"""
        for section in self.sections:
            if section.kind in kinds:
                return section
        return None

    def text_of(self, *kinds: str) -> str:
        """把指定类型章节的正文拼起来。"""
        return "\n".join(s.body for s in self.get(*kinds) if s.body).strip()

    def has(self, *kinds: str) -> bool:
        return any(s.kind in kinds for s in self.sections)

    def to_dict(self) -> dict[str, object]:
        return {
            "count": len(self.sections),
            "kinds": self.kinds(),
            "sections": [s.to_dict() for s in self.sections],
        }


# ---------------------------------------------------------------------------
# 切分
# ---------------------------------------------------------------------------

#: "标题：内容" 同行形式的分隔符
_INLINE_SEP_RE = re.compile(r"^([^：:]{2,20})\s*[：:]\s*(.+)$")


def _try_inline_heading(line: str) -> tuple[str, str, str] | None:
    """处理「教育背景：沈阳大学 人工智能 本科」这种同行形式。

    Returns:
        ``(章节类型, 标准标题, 剩余内容)``；不成立则返回 None。
    """
    m = _INLINE_SEP_RE.match(line.strip())
    if not m:
        return None
    head, rest = m.group(1).strip(), m.group(2).strip()
    matched = match_heading(head)
    if not matched:
        return None
    kind, title = matched
    if not rest:
        return None
    return kind, title, rest


def split_sections(text: str) -> SectionIndex:
    """把文本切分成章节。

    切分规则：
    - 逐行扫描，遇到标题行就开启新章节；
    - 标题行之前的行归入 ``preamble``（简历的姓名/联系方式常在这里，
      因为很多简历不写"个人信息"这个小标题）；
    - 标题行后面、下一个标题之前的行归入当前章节。

    Args:
        text: 已经过 :func:`~app.textutil.normalize_text` 的文本。

    Returns:
        SectionIndex。
    """
    lines = (text or "").split("\n")
    index = SectionIndex(total_lines=len(lines))

    current: Section | None = None
    # 逐行累积 offset：注意行与行之间那个被 split 吃掉的 "\n" 也要算进去
    offset = 0

    for lineno, line in enumerate(lines):
        line_len = len(line)
        line_start = offset
        line_end = offset + line_len
        offset = line_end + 1  # +1 对应换行符

        inline = _try_inline_heading(line) if current is None or True else None
        if inline is not None:
            kind, title, rest = inline
            if current is not None:
                current.end_line = lineno - 1
                current.end_offset = line_start
                index.sections.append(current)
            current = Section(
                kind=kind,
                title=title,
                start_line=lineno,
                end_line=lineno,
                start_offset=line_start,
                end_offset=line_end,
                heading_line=line,
                lines=[line, rest],
            )
            continue

        matched = match_heading(line)
        if matched is not None:
            kind, title = matched
            if current is not None:
                current.end_line = lineno - 1
                current.end_offset = line_start
                index.sections.append(current)
            current = Section(
                kind=kind,
                title=title,
                start_line=lineno,
                end_line=lineno,
                start_offset=line_start,
                end_offset=line_end,
                heading_line=line,
                lines=[line],
            )
            continue

        if current is None:
            index.preamble_lines.append(line)
        else:
            current.lines.append(line)
            current.end_line = lineno
            current.end_offset = line_end

    if current is not None:
        current.end_line = len(lines) - 1
        index.sections.append(current)

    index.preamble = "\n".join(index.preamble_lines).strip()
    return index


def strip_sections_for_skills(index: SectionIndex) -> str:
    """取"适合做技能扫描"的文本。

    排除掉自我评价与福利待遇这类噪声章节 —— 自我评价里出现的
    "有良好的沟通能力"很容易把无关词扫成技能，干扰判断。
    """
    parts: list[str] = []
    if index.preamble:
        parts.append(index.preamble)
    for section in index.sections:
        if section.kind in (KIND_SUMMARY, KIND_BENEFIT):
            continue
        parts.append(section.body)
    return "\n".join(p for p in parts if p).strip()


__all__ = [
    "Section",
    "SectionIndex",
    "split_sections",
    "match_heading",
    "strip_sections_for_skills",
    "EXPERIENCE_KINDS",
    "JD_REQUIREMENT_KINDS",
    "KIND_PERSONAL",
    "KIND_INTENT",
    "KIND_EDUCATION",
    "KIND_WORK",
    "KIND_INTERNSHIP",
    "KIND_PROJECT",
    "KIND_RESEARCH",
    "KIND_CAMPUS",
    "KIND_SKILL",
    "KIND_AWARD",
    "KIND_CERT",
    "KIND_SUMMARY",
    "KIND_DUTY",
    "KIND_REQUIREMENT",
    "KIND_BONUS",
    "KIND_BENEFIT",
    "KIND_OTHER",
]
