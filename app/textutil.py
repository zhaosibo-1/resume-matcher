"""文本处理工具：规范化、句子切分、证据提取、日期与量化识别。

这一层的存在意义是**把所有"脏活"收口**，让解析器（parser）能专注于业务判断。

关于字符偏移量的一个重要约定
----------------------------
本模块的 :func:`normalize_text` **会改变文本长度**（例如把 ``\\r\\n`` 压成 ``\\n``、
删掉零宽字符）。为了让「证据高亮」这件事不出错，项目里统一约定：

    凡是要暴露给外部的 ``start`` / ``end`` 偏移量，
    **一律基于 normalize_text 之后的那份文本**。

因此解析产物里的 ``raw_text`` 存的也是规范化后的文本，前端渲染时
直接用这份文本即可，不需要做任何偏移换算。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime

# ---------------------------------------------------------------------------
# 文本规范化
# ---------------------------------------------------------------------------

#: 零宽与不可见字符。简历从 PDF/网页复制过来时经常夹带这些，会破坏技能匹配。
_INVISIBLE_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff\u00ad]")

#: 需要统一的各类"破折号/连字符"。统一成 ASCII 连字符便于后续处理。
_DASHES = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\uff0d"
_DASH_MAP = {ord(ch): "-" for ch in _DASHES}

#: 各种引号统一成 ASCII 引号（保留中文引号语义的场合不多，统一更好处理）
_QUOTE_MAP = {
    ord("\u2018"): "'",
    ord("\u2019"): "'",
    ord("\u201c"): '"',
    ord("\u201d"): '"',
    ord("\u3000"): " ",  # 全角空格
}


#: 需要转成半角的"技术符号"全角形式。
#:
#: 为什么不直接按 FF01-FF5E 整段转？因为那会把中文标点也一起转掉
#: （中文逗号 ``,``、括号 ``（）``、冒号 ``：``），前端展示出来就不像用户原文了。
#: 但技术名词里的符号必须转 —— 中文输入法下很容易打出全角的 `.`、`+`、`#`，
#: 不转的话 `Node．js`、`C＋＋`、`C＃` 这些全都匹配不上。
_TECH_SYMBOLS = {
    "\uff0e": ".",  # ．
    "\uff0b": "+",  # ＋
    "\uff03": "#",  # ＃
    "\uff0d": "-",  # －
    "\uff3f": "_",  # ＿
    "\uff0f": "/",  # ／
    "\uff0a": "*",  # ＊
    "\uff20": "@",  # ＠
}
# 注意：全角括号 （） 与全角冒号 ： 刻意**不转** ——
# 它们在中文语境里（"专业技能（选填）"、"任职要求："）是正常标点，
# 转成半角反而会让原文看起来被改过。


def _to_halfwidth(text: str) -> str:
    """把全角**字母、数字与技术符号**转成半角，中文标点原样保留。

    用途：有些简历用了全角写法（``Ｐｙｔｈｏｎ``、``Ｃ＋＋``），
    不转换的话技能词典一条都匹配不上。

    保留的字符：中文逗号 ``,``、中文句号 ``。``、中文冒号 ``：``、
    中文问号感叹号、以及各种中文引号 —— 这些是"用户原文的样子"，
    改掉会让"证据溯源"失去说服力（用户会认出"这不是我写的"）。
    """
    chars = []
    for ch in text:
        code = ord(ch)
        if 0xFF10 <= code <= 0xFF19 or 0xFF21 <= code <= 0xFF3A or 0xFF41 <= code <= 0xFF5A:
            # 全角数字 ０-９ / 大写 Ａ-Ｚ / 小写 ａ-ｚ
            chars.append(chr(code - 0xFEE0))
        elif ch in _TECH_SYMBOLS:
            chars.append(_TECH_SYMBOLS[ch])
        else:
            chars.append(ch)
    return "".join(chars)


def normalize_text(text: str) -> str:
    """规范化原始文本。

    具体做四件事：

    1. 统一换行符（``\\r\\n`` / ``\\r`` → ``\\n``）；
    2. 删除零宽与不可见字符（PDF 复制常见）；
    3. 全角 ASCII 转半角（提升技能匹配率）；
    4. 统一破折号与引号，并把连续 3 个以上空行压成 2 个。

    此外会顺手做一次 Unicode 规范化（NFKC 的弱化版：只处理组合字符），
    避免出现"看起来一样但比较不相等"的怪字符。
    """
    if not text:
        return ""

    # 组合字符归一（不改变视觉，只改变编码）
    text = unicodedata.normalize("NFC", text)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _INVISIBLE_RE.sub("", text)
    text = _to_halfwidth(text)
    text = text.translate(_DASH_MAP)
    text = text.translate(_QUOTE_MAP)

    # 行尾空白清掉（但保留行首缩进，因为缩进对判断层级有用）
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    # 压缩过多空行
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip("\n")


# ---------------------------------------------------------------------------
# 行与列表项
# ---------------------------------------------------------------------------

#: 列表项前缀：- * • · ▪ ◦ ● ○ ◆ ◇ ■ □ → ➢ ➤ ‣  以及
#: "1." "1、" "（1）" "(1)" "①" "（一）" 等编号形式
_BULLET_RE = re.compile(
    r"^\s*(?:"
    r"[-*•·▪◦●○◆◇■□→➢➤‣]"
    r"|\(?\d{1,2}\)?[.、)）]"          # 1.  2、  (3)  （4）
    r"|[（(]\d{1,2}[)）]"              # （1）  (2)
    r"|[①-⑳]"
    r"|[（(][一二三四五六七八九十]{1,3}[)）]"
    r")\s*"
)


def is_bullet(line: str) -> bool:
    """判断一行是不是列表项。"""
    return bool(_BULLET_RE.match(line))


def strip_bullet(line: str) -> str:
    """去掉列表项前缀，返回正文。"""
    return _BULLET_RE.sub("", line, count=1).strip()


def indentation(line: str) -> int:
    """计算行首缩进宽度（Tab 按 4 空格算）。"""
    width = 0
    for ch in line:
        if ch == " ":
            width += 1
        elif ch == "\t":
            width += 4
        else:
            break
    return width


def clean_heading(line: str) -> str:
    """清理标题行的装饰，返回纯净标题文字。

    处理 Markdown 井号、中文序号（``一、`` / ``（一）``）、
    方括号与星号包裹（``【教育背景】`` / ``**教育背景**``）等形式。
    """
    text = line.strip()
    text = re.sub(r"^#{1,6}\s*", "", text)                 # Markdown 标题
    text = re.sub(r"^\*\*(.+?)\*\*$", r"\1", text)          # **加粗**
    text = re.sub(r"^__(.+?)__$", r"\1", text)
    text = re.sub(r"^[【\[（(](.+?)[】\]）)]$", r"\1", text)  # 【】 [] （）
    text = re.sub(r"^[一二三四五六七八九十]{1,3}\s*[、.．]\s*", "", text)  # 一、
    text = re.sub(r"^第[一二三四五六七八九十]{1,3}部分\s*[：:]?\s*", "", text)
    text = re.sub(r"[：:]\s*$", "", text)                    # 结尾冒号
    return text.strip()


# ---------------------------------------------------------------------------
# 句子切分
# ---------------------------------------------------------------------------

#: 中文句末标点（全角已被规范化成半角，这里两种都保留以防万一）
_SENT_END = "。！？；!?;"

#: 英文句点保护：点后面**不跟**空白或行尾时，说明它是名字的一部分
#: （``Node.js`` / ``v3.5`` / ``example.com``），不能作为句子边界。
#:
#: 这里踩过一次逻辑写反的坑：最初写的是 ``\.(?=\s|$)``（点后跟空白才保护），
#: 结果恰好把真正的句末点保护了起来、让名字内部的点去当分隔符 ——
#: 行为完全颠倒，而且因为大多数中文文本用「。」断句，很难被发现，
#: 只有在纯英文段落上才会暴露。
_EN_PERIOD_RE = re.compile(r"\.(?!\s|$)")


def split_sentences(text: str) -> list[str]:
    """中文友好的句子切分。

    规则：
    - 中文/通用句末标点（``。！？；!?;``）与英文句点直接切；
    - 英文句点若**紧贴字母数字**（``Node.js`` / ``v3.5`` / 网址）则不切，
      做法是先把这些点替换成占位符，切分完再还原；
    - 换行也作为切分点（简历/JD 大量依赖换行表达层级）。

    Returns:
        去掉空白项与首尾空格的句子列表。
    """
    if not text:
        return []

    placeholder = "\x00"
    guarded = _EN_PERIOD_RE.sub(placeholder, text)

    parts = re.split(rf"[{re.escape(_SENT_END)}.]|\n+", guarded)

    out: list[str] = []
    for part in parts:
        cleaned = part.replace(placeholder, ".").strip()
        cleaned = cleaned.strip(" \t·•-")
        if cleaned:
            out.append(cleaned)
    return out


def split_blocks(text: str, *, max_len: int = 220) -> list[str]:
    """把文本切成"信息块"：优先按行，太长的行再按句切。

    用于抽取要求条目 —— JD 里的一条要求可能是一整行，
    也可能是一行里塞了两句话，需要拆开分别判断需求强度。
    """
    blocks: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        body = strip_bullet(line)
        if not body:
            continue
        if len(body) <= max_len:
            blocks.append(body)
            continue
        # 太长就按句子拆，再把过短的相邻句合并回来
        buf = ""
        for sent in split_sentences(body):
            if buf and len(buf) + len(sent) > max_len:
                blocks.append(buf)
                buf = sent
            else:
                buf = f"{buf} {sent}" if buf else sent
        if buf:
            blocks.append(buf)
    return blocks


# ---------------------------------------------------------------------------
# 证据提取
# ---------------------------------------------------------------------------


def evidence_around(text: str, start: int, end: int, *, span: int = 40) -> str:
    """取命中位置前后的原文片段，作为"我们凭什么这么判断"的证据。

    会尽量把片段对齐到句子边界 —— 截断在句子中间的证据读起来很别扭。
    """
    if not text:
        return ""
    lo = max(0, start - span)
    hi = min(len(text), end + span)

    # 向左扩到句末标点或换行之后
    left = max(text.rfind(ch, lo, start) for ch in _SENT_END + "\n")
    if left >= lo:
        lo = left + 1
    # 向右扩到句末标点或换行
    right_positions = [text.find(ch, end, hi) for ch in _SENT_END + "\n"]
    rights = [p for p in right_positions if p >= 0]
    if rights:
        hi = min(rights) + 1

    snippet = text[lo:hi].strip()
    snippet = re.sub(r"\s+", " ", snippet)
    if len(snippet) > 160:
        snippet = snippet[:157] + "..."
    return snippet


# ---------------------------------------------------------------------------
# 日期与量化
# ---------------------------------------------------------------------------

_MONTH = r"(?P<m>0?[1-9]|1[0-2])(?!\d)"
_YEAR = r"(?P<y>(?:19|20)\d{2})"

#: 「2024.03 - 2024.09」「2023年9月至2024年6月」「2022/07-至今」等
_RANGE_RE = re.compile(
    rf"(?P<sy>(?:19|20)\d{{2}})\s*(?:[.\-/年]\s*(?P<sm>0?[1-9]|1[0-2])(?!\d))?\s*(?:月)?"
    rf"\s*(?:[-~至到]|—|–)+\s*"
    rf"(?:(?P<ey>(?:19|20)\d{{2}})\s*(?:[.\-/年]\s*(?P<em>0?[1-9]|1[0-2])(?!\d))?|(?P<now>至今|现在|今|now|present))",
    re.IGNORECASE,
)

#: 单点日期：「2025年6月毕业」「2023.09 入学」
_SINGLE_RE = re.compile(
    rf"(?P<y>(?:19|20)\d{{2}})\s*(?:[.\-/年]\s*(?P<m>0?[1-9]|1[0-2])(?!\d))?",
)

#: 仅年份范围（「2023-2027」），没有月份
_YEAR_SPAN_RE = re.compile(r"(?P<sy>(?:19|20)\d{2})\s*(?:[-~至到—–])\s*(?P<ey>(?:19|20)\d{2})")

_NOW_YEAR = datetime.now().year
_NOW_MONTH = datetime.now().month


@dataclass
class DateRange:
    """一个时间区间。"""

    start_year: int
    start_month: int | None
    end_year: int
    end_month: int | None
    months: int
    ongoing: bool = False
    raw: str = ""

    @property
    def label(self) -> str:
        """人类可读的区间描述。"""

        def fmt(year: int, month: int | None) -> str:
            return f"{year}.{month:02d}" if month else str(year)

        tail = "至今" if self.ongoing else fmt(self.end_year, self.end_month)
        return f"{fmt(self.start_year, self.start_month)}-{tail}"


def _months_between(y1: int, m1: int, y2: int, m2: int) -> int:
    """两个年月之间相差的月数（含头含尾，最小 1）。"""
    value = (y2 - y1) * 12 + (m2 - m1) + 1
    return max(1, value)


def find_date_range(text: str) -> DateRange | None:
    """在文本里找第一个时间区间。

    按「完整区间 → 年份区间 → 单点日期」的顺序尝试，先匹配到更精确的。
    找不到返回 None。
    """
    if not text:
        return None

    m = _RANGE_RE.search(text)
    if m:
        sy = int(m.group("sy"))
        sm = int(m.group("sm")) if m.group("sm") else None
        ongoing = bool(m.group("now"))
        if ongoing:
            ey, em = _NOW_YEAR, _NOW_MONTH
        else:
            ey = int(m.group("ey"))
            em = int(m.group("em")) if m.group("em") else None

        # 缺月份时按年初/年末折算，宁可保守（少算而不是多算）
        sm_eff = sm or 1
        em_eff = em or 12
        months = _months_between(sy, sm_eff, ey, em_eff)
        if months > 0 and ey >= sy:
            return DateRange(sy, sm, ey, em, months, ongoing=ongoing, raw=m.group(0).strip())

    m2 = _YEAR_SPAN_RE.search(text)
    if m2:
        sy, ey = int(m2.group("sy")), int(m2.group("ey"))
        if ey >= sy:
            return DateRange(sy, None, ey, None, _months_between(sy, 1, ey, 12), raw=m2.group(0).strip())

    return None


def find_years(text: str) -> list[int]:
    """找出文本里所有 4 位年份（去重，保持出现顺序）。"""
    out: list[int] = []
    for m in re.finditer(r"(?<!\d)((?:19|20)\d{2})(?!\d)", text or ""):
        year = int(m.group(1))
        if 1990 <= year <= _NOW_YEAR + 8 and year not in out:
            out.append(year)
    return out


def find_single_date(text: str) -> tuple[int, int | None] | None:
    """找第一个「年[月]」形式的时间点，如「2025年6月毕业」→ (2025, 6)。"""
    m = _SINGLE_RE.search(text or "")
    if not m:
        return None
    month = int(m.group("m")) if m.group("m") else None
    return int(m.group("y")), month


#: 量词 / 单位表。放在一起是为了让"数字 + 单位"这个模式可读、可维护。
#:
#: 注意**刻意收录了单字母以外的单位**：``G`` / ``M`` / ``K`` 这类单字母
#: 容易被 "5G 网络""3D 建模" 误伤，所以只保留 ``GB`` / ``MB`` 这种两字母形式。
_UNITS = (
    "人|用户|客户|次|条|个|台|套|种|篇|份|轮|页|行|所|家"
    "|万次|万条|万行"
    "|qps|tps|uv|pv|dau|mau"
    "|ms|毫秒|秒|分钟|小时|天|周|月|年|倍|分|元|美元"
    "|GB|MB|TB|KB"
)

#: 量化成果识别：「提升 30%」「降低至 200ms」「服务 10 万用户」「覆盖 6 种格式」
_PERCENT_RE = re.compile(r"\d+(?:\.\d+)?\s*%|百分之\s*[一二三四五六七八九十百零\d]+")
_QUANTIFIER_RE = re.compile(rf"\d+(?:\.\d+)?\s*(?:万|亿|千|百|k|K|w|W)?\s*(?:{_UNITS})", re.IGNORECASE)

#: 指标名在前、数字在后：「QPS 达到 5000」「日活 10 万」
_METRIC_FIRST_RE = re.compile(
    r"(?:qps|tps|uv|pv|dau|mau|日活|月活|并发|吞吐|响应时间|延迟|精度|召回率|准确率)\D{0,6}\d+",
    re.IGNORECASE,
)

_NUMBER_RE = re.compile(r"(?<![\d.])\d+(?:\.\d+)?(?![\d.])")


def has_quantified(text: str) -> bool:
    """判断一句话里有没有量化数据（数字 + 单位/百分比/指标名）。

    为什么关心这个？因为筛选简历时，"优化了系统性能" 和
    "把接口 P99 从 800ms 优化到 120ms" 是两个完全不同的分量，
    后者才是真正的亮点。这个信号会参与「项目相关度」维度的打分。
    """
    if not text:
        return False
    if _PERCENT_RE.search(text):
        return True
    if _QUANTIFIER_RE.search(text):
        return True
    return bool(_METRIC_FIRST_RE.search(text))


def count_numbers(text: str) -> int:
    """统计文本里的独立数字个数（用于粗略判断"数据密度"）。"""
    return len(_NUMBER_RE.findall(text or ""))


def shorten(text: str, limit: int = 60, *, suffix: str = "...") -> str:
    """截断到指定长度，超出部分用省略号替代。"""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(suffix))] + suffix


#: 短语边界字符：截上下文短语时遇到这些就停
_PHRASE_BOUNDARY = "。！？；，、,;!?：:\n（）()【】[]|｜·"


def context_phrase(text: str, index: int, length: int, *, left: int = 6, right: int = 5) -> str:
    """取某个词在文中的**短语级上下文**。

    用途：像学历这种字段，直接把整行截出来会得到
    「1. 本科及以上学历，计算机、人工智能相关专业，2027 届毕业生；」，
    又长又杂。这里只向左右各扩几个字符，并且碰到标点就停，
    得到的是「本科及以上学历」这样的干净短语。

    Args:
        text: 全文。
        index: 目标词的起始位置。
        length: 目标词的长度。
        left: 向左最多扩几个字符。
        right: 向右最多扩几个字符。
    """
    if not text:
        return ""

    lo = index
    while lo > 0 and index - lo < left and text[lo - 1] not in _PHRASE_BOUNDARY:
        lo -= 1

    end = index + length
    hi = end
    while hi < len(text) and hi - end < right and text[hi] not in _PHRASE_BOUNDARY:
        hi += 1

    return text[lo:hi].strip()


__all__ = [
    "normalize_text",
    "is_bullet",
    "strip_bullet",
    "indentation",
    "clean_heading",
    "split_sentences",
    "split_blocks",
    "evidence_around",
    "DateRange",
    "find_date_range",
    "find_years",
    "find_single_date",
    "has_quantified",
    "count_numbers",
    "shorten",
    "context_phrase",
]
