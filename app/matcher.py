"""打分引擎：把「已解析的 JD」与「已解析的简历」变成一份可解释的匹配报告。

这是整个系统的核心。设计上有一条贯穿始终的原则：

    拒绝黑盒分数。

具体体现为三点：

1. **六维拆解**。不给一个笼统的"匹配度 73 分"，而是拆成硬性技能、加分技能、
   经验年限、学历、行业领域、项目相关度六个维度，每一维单独算分、单独解释。
   用户能一眼看出"是技能不够还是经验不够"。

2. **每个分数都能反查证据**。技能匹配维度会带出简历原文片段，
   让用户看到"我们是因为你写了这句才给你算上的"。

3. **权重可调且实时重算**。权重不写死在算法里 —— 一个偏重工程能力的岗位和
   一个偏重研究背景的岗位，各维度权重本该不同。用户在界面上拖动滑条，
   系统只重算加权和，不重新解析、不重新调用大模型，毫秒级返回。

关于"权重重分配"
----------------
某些维度在特定 JD 下是**不适用**的：JD 没写年限要求、没写学历要求、
没提行业领域。这时如果把它们按 0 分计入，会让"一份完全不提学历的 JD"
天然低分——这显然不对。

所以这里做了权重再归一化：不适用维度的权重，按比例**分给**其余适用维度。
这个细节是区分"能用"和"好用"的地方。
"""

from __future__ import annotations

import logging
import re
from typing import Iterable, Optional

from .schemas import (
    DEFAULT_WEIGHTS,
    DIMENSION_LABELS,
    DIMENSION_ORDER,
    DimensionScore,
    InterviewQuestion,
    MatchReport,
    ParsedJD,
    ParsedResume,
)
from .skills import DEFAULT_NORMALIZER, SkillNormalizer

logger = logging.getLogger(__name__)


# ===========================================================================
# 一、技能可信度
# ===========================================================================

#: 技能出现在简历哪个位置 -> 可信度系数。
#:
#: 为什么需要这个？因为「熟悉 Python」这句话：
#:   - 写在「专业技能」清单里，是作者主动、结构化地声明 => 可信
#:   - 写在「项目经历」里，是实际用过的痕迹        => 可信
#:   - 写在「自我评价」里，是顺口一提的软性描述    => 打对折
#: 如果一视同仁，那"在自我评价里堆满技术名词"的简历会得到虚高的分数。
ORIGIN_WEIGHT: dict[str, float] = {
    "skill_section": 1.0,
    "experience": 1.0,
    "education": 0.9,
    "personal": 0.8,
    "other": 0.7,
    "summary": 0.55,
}

#: 达到这个可信度才算"实打实的命中"
STRONG_THRESHOLD = 0.8


def skill_strength_map(
    resume: ParsedResume,
    normalizer: SkillNormalizer = DEFAULT_NORMALIZER,
) -> dict[str, float]:
    """构建「技能名 -> 可信度」映射。

    同一技能在简历里可能出现多次（技能清单里写一次、项目里又提一次），
    取**最高的那个可信度**——只要有一处是实打实写的，就当它是真的。

    最后一步做**上位技能继承**：会 FAISS 就等于会用向量数据库。
    不做这一步，JD 里「了解向量数据库（FAISS / Milvus）」这种写法
    会让"向量数据库"永远判成缺失，而"FAISS"又判成命中，
    结果报告自相矛盾。
    """
    strength: dict[str, float] = {}
    for item in resume.skills:
        value = ORIGIN_WEIGHT.get(item.origin, 0.7)
        # 「精通」比「了解」更硬：按程度词再微调一次
        if item.level_score is not None:
            if item.level_score >= 4:
                value = min(1.0, value + 0.05)
            elif item.level_score <= 2:
                value *= 0.85
        if value > strength.get(item.canonical, 0.0):
            strength[item.canonical] = round(value, 3)

    # 上位技能继承（只继承一级）
    for name, value in list(strength.items()):
        for parent in normalizer.parents_of(name):
            if value > strength.get(parent, 0.0):
                strength[parent] = value

    return strength


def _split_by_strength(
    required: Iterable[str],
    strength: dict[str, float],
) -> tuple[list[str], list[str], list[str]]:
    """把要求项分成「命中 / 弱命中 / 缺失」三组。

    弱命中的含义是：这个技能在简历里出现了，但只出现在自我评价之类
    可信度低的位置。它不该被判成"没有"，但也算不上扎实掌握，
    所以单独列出来提醒用户"把它挪到技能清单里会更有说服力"。
    """
    matched: list[str] = []
    weak: list[str] = []
    missing: list[str] = []

    for name in required:
        value = strength.get(name, 0.0)
        if value >= STRONG_THRESHOLD:
            matched.append(name)
        elif value > 0.0:
            weak.append(name)
        else:
            missing.append(name)

    return matched, weak, missing


def _evidence_for(names: Iterable[str], resume: ParsedResume, *, limit: int = 4) -> list[str]:
    """为一批技能挑出简历里的原始证据片段。"""
    wanted = set(names)
    evidence: list[str] = []
    seen: set[str] = set()

    # 优先展示技能清单/经历里的证据（可信度高），自我评价里的排后面
    ordered = sorted(
        resume.skills,
        key=lambda item: -ORIGIN_WEIGHT.get(item.origin, 0.7),
    )
    for item in ordered:
        if item.canonical not in wanted or not item.evidence:
            continue
        if item.evidence in seen:
            continue
        seen.add(item.evidence)
        evidence.append(f"[{item.canonical}] {item.evidence}")
        if len(evidence) >= limit:
            break
    return evidence


# ===========================================================================
# 二、六个维度的打分
# ===========================================================================


def score_must_have(jd: ParsedJD, resume: ParsedResume, strength: dict[str, float]) -> DimensionScore:
    """维度一：硬性技能覆盖度。权重最高的维度。

    得分 = ∑(每个硬性技能的可信度) / 硬性技能总数 × 100。

    注意这里用的是**可信度求和**而不是**命中个数**：
    8 个硬性要求里命中 6 个，但如果其中一个"只是在自我评价里提过",
    得分会略低于 75 而不是直接算 75。
    """
    required = list(jd.must_have)

    if not required:
        return DimensionScore(
            key="must_have",
            label=DIMENSION_LABELS["must_have"],
            score=0.0,
            weight=0.0,
            weighted=0.0,
            applicable=False,
            detail="这份 JD 里没有识别出明确的硬性技能要求，该维度不参与打分。",
        )

    matched, weak, missing = _split_by_strength(required, strength)
    total = sum(strength.get(name, 0.0) for name in required)
    score = round(total / len(required) * 100, 1)

    detail = f"JD 提出 {len(required)} 项硬性技能，简历扎实覆盖 {len(matched)} 项"
    if weak:
        detail += f"，另有 {len(weak)} 项仅在低可信位置出现"
    detail += f"（加权覆盖 {score}%）。"

    gaps = [f"缺失硬性技能：{name}" for name in missing]
    gaps += [f"{name}：仅在自我评价等位置提到，建议补充到技能清单或项目经历里" for name in weak]

    return DimensionScore(
        key="must_have",
        label=DIMENSION_LABELS["must_have"],
        score=score,
        weight=0.0,  # 由 normalize_weights 统一填充
        weighted=0.0,
        applicable=True,
        detail=detail,
        evidence=_evidence_for(matched + weak, resume),
        gaps=gaps,
        matched=matched + weak,
        missing=missing,
    )


def score_nice_to_have(jd: ParsedJD, resume: ParsedResume, strength: dict[str, float]) -> DimensionScore:
    """维度二：加分技能覆盖度。

    加分项通常写在"者优先/加分/了解"下面，属于锦上添花。
    这一维的分数**不应该太高**——如果一个人把加分项全占了，
    说明他可能过度准备了这个岗位，但这不是坏事，所以也不封顶。
    """
    required = list(jd.nice_to_have)

    if not required:
        return DimensionScore(
            key="nice_to_have",
            label=DIMENSION_LABELS["nice_to_have"],
            score=0.0,
            weight=0.0,
            weighted=0.0,
            applicable=False,
            detail="这份 JD 里没有识别出加分项，该维度不参与打分。",
        )

    matched, weak, missing = _split_by_strength(required, strength)
    total = sum(strength.get(name, 0.0) for name in required)
    score = round(total / len(required) * 100, 1)

    detail = f"JD 列出 {len(required)} 项加分技能，简历覆盖 {len(matched)} 项"

    if score >= 60:
        detail += f"（加权覆盖 {score}%）—— 加分项覆盖得不错，是个亮点。"
    elif score >= 30:
        detail += f"（加权覆盖 {score}%）—— 覆盖了一部分，属于正常水平。"
    else:
        detail += f"（加权覆盖 {score}%）—— 覆盖面偏低，但这不构成致命短板。"

    return DimensionScore(
        key="nice_to_have",
        label=DIMENSION_LABELS["nice_to_have"],
        score=score,
        weight=0.0,
        weighted=0.0,
        applicable=True,
        detail=detail,
        evidence=_evidence_for(matched, resume),
        gaps=[f"未覆盖加分项：{name}" for name in missing],
        matched=matched + weak,
        missing=missing,
    )


#: 校招口径下把实习也算进去的判定依据
_ENTRY_LEVEL_SENIORITY = ("实习", "校招", "初级")


def effective_years(jd: ParsedJD, resume: ParsedResume) -> tuple[float, str]:
    """计算用于比对的"有效经验年限"，并返回口径说明。

    口径分两种（这是显式规则，不是藏在代码里的魔法系数）：

    - **校招口径**：JD 写的是实习/校招，或简历显示是在校生/应届生时，
      直接用 ``total_years``（包含实习、项目、科研经历）。
      理由：校招本来就不指望你有正式工作经验，实习和项目才是主要证据。
    - **社招口径**：只算正式工作年限，实习经历按 **50%** 折算
      （``work_years + internship_months / 12 * 0.5``）。
      理由：社招看重的是全职工作产出，实习的等价性明显更低。
    """
    entry_level = (jd.seniority in _ENTRY_LEVEL_SENIORITY) or resume.degree_expected

    if entry_level:
        return resume.total_years, "校招口径（含实习、项目与科研经历）"

    converted = resume.work_years + resume.internship_months / 12 * 0.5
    return round(converted, 1), "社招口径（实习按 50% 折算）"


def score_experience(jd: ParsedJD, resume: ParsedResume) -> DimensionScore:
    """维度三：经验年限匹配度。

    分段线性打分（做一个"达标即满分、差一半得一半分"的折线）::

        分数
        100 |            _______________
            |           /
         50 |          /
            |         /
          0 |________/
            0      0.5×N        N       年限

    JD 没写年限要求时该维度不适用。
    """
    need = jd.min_years

    if need is None:
        return DimensionScore(
            key="experience",
            label=DIMENSION_LABELS["experience"],
            score=0.0,
            weight=0.0,
            weighted=0.0,
            applicable=False,
            detail="这份 JD 没有写明经验年限要求，该维度不参与打分。",
        )

    have, caliber = effective_years(jd, resume)

    if need <= 0:
        score = 100.0
        detail = f"JD 接受应届/无经验，简历折算 {have} 年，{caliber}。"
    elif have >= need:
        score = 100.0
        detail = f"要求 {need:g} 年，简历折算 {have} 年，已达标。{caliber}。"
    elif have >= need * 0.5:
        ratio = (have - need * 0.5) / (need * 0.5)
        score = round(50 + ratio * 50, 1)
        detail = (
            f"要求 {need:g} 年，简历折算 {have} 年，差 {round(need - have, 1)} 年。"
            f"属于「接近但未达标」，差距在半个要求之内，可以尝试投递。{caliber}。"
        )
    else:
        score = round(have / (need * 0.5) * 50, 1)
        detail = f"要求 {need:g} 年，简历折算 {have} 年，差距较大（{round(need - have, 1)} 年）。{caliber}。"

    # 证据：把带时间的经历列出来
    evidence: list[str] = []
    for exp in resume.experiences:
        if exp.period:
            kind_label = {
                "work": "工作",
                "internship": "实习",
                "project": "项目",
                "campus": "校园",
            }.get(exp.kind, "经历")
            org = exp.org or exp.title or "未命名经历"
            evidence.append(f"[{kind_label}] {org} {exp.period}（{exp.duration_months or '?'} 个月）")
    if resume.graduation_year:
        evidence.append(f"预计 {resume.graduation_year} 年毕业")

    gaps: list[str] = []
    if score < 100 and need > 0:
        gaps.append(f"经验年限差 {round(max(0.0, need - have), 1)} 年")

    return DimensionScore(
        key="experience",
        label=DIMENSION_LABELS["experience"],
        score=score,
        weight=0.0,
        weighted=0.0,
        applicable=True,
        detail=detail,
        evidence=evidence[:6],
        gaps=gaps,
        matched=[f"折算 {have} 年"] if have > 0 else [],
        missing=[f"要求 {need:g} 年"] if have < need else [],
    )


#: 学历等级 -> 中文名
_LEVEL_NAMES = {0: "不限", 1: "大专", 2: "本科", 3: "硕士", 4: "博士"}


def score_education(jd: ParsedJD, resume: ParsedResume) -> DimensionScore:
    """维度四：学历匹配度。

    规则刻意做得**宽容**：学历是硬门槛里最不该"一票否决"的一项，
    而且现实中"要求本科"收到硕士简历是加分而非减分。
    所以差一级给 55 分（而不是 0），差两级才给 20 分。
    """
    need = jd.education_level

    if need <= 0:
        return DimensionScore(
            key="education",
            label=DIMENSION_LABELS["education"],
            score=0.0,
            weight=0.0,
            weighted=0.0,
            applicable=False,
            detail="这份 JD 没有明确的学历门槛，该维度不参与打分。",
        )

    have = resume.education_level
    need_name = _LEVEL_NAMES.get(need, f"L{need}")
    have_name = _LEVEL_NAMES.get(have, "未识别") if have else "未识别"

    if have >= need:
        score = 100.0
        detail = f"要求{need_name}，简历为{have_name}，达标。"
        gaps: list[str] = []
    elif have == need - 1:
        score = 55.0
        detail = f"要求{need_name}，简历为{have_name}，差一级。若有过硬的项目与技能表现，仍有机会。"
        gaps = [f"学历差一级：要求{need_name}，简历{have_name}"]
    elif have > 0:
        score = 20.0
        detail = f"要求{need_name}，简历为{have_name}，差距较大，该岗位的学历门槛可能较难通过。"
        gaps = [f"学历不达标：要求{need_name}，简历{have_name}"]
    else:
        score = 0.0
        detail = f"要求{need_name}，但未能从简历里识别出学历信息。"
        gaps = ["简历未写明学历，建议在教育背景里明确标注"]

    evidence: list[str] = []
    if resume.school:
        evidence.append(f"学校：{resume.school}")
    if resume.education:
        evidence.append(f"学历：{resume.education}")
    if resume.major:
        evidence.append(f"专业：{resume.major}")
    if resume.graduation_year:
        evidence.append(f"毕业年份：{resume.graduation_year}")

    # 专业匹配（作为附加证据，不单独占权重）
    if jd.majors:
        overlap = [m for m in jd.majors if resume.major and (m in resume.major or resume.major in m)]
        if overlap:
            evidence.append(f"专业对口：{overlap[0]}")
        elif resume.major:
            evidence.append(f"JD 倾向专业：{'、'.join(jd.majors[:3])}（简历专业为 {resume.major}）")

    return DimensionScore(
        key="education",
        label=DIMENSION_LABELS["education"],
        score=score,
        weight=0.0,
        weighted=0.0,
        applicable=True,
        detail=detail,
        evidence=evidence,
        gaps=gaps,
        matched=[have_name] if have >= need else [],
        missing=[need_name] if have < need else [],
    )


def score_domain(jd: ParsedJD, resume: ParsedResume) -> DimensionScore:
    """维度五：行业/领域匹配度。

    判断依据是 JD 里出现的领域关键词（金融科技、电商、智能客服……）
    在简历里的覆盖情况。

    信息缺失的处理很有意思：如果简历里**完全没有**任何领域线索，
    这里给 55 分的中性分而不是 0 分 —— 因为"没写"和"没有"是两回事，
    很多简历确实不会写行业背景，不该因此重罚。
    """
    required = list(jd.domains)

    if not required:
        return DimensionScore(
            key="domain",
            label=DIMENSION_LABELS["domain"],
            score=0.0,
            weight=0.0,
            weighted=0.0,
            applicable=False,
            detail="这份 JD 没有体现明确的行业方向，该维度不参与打分。",
        )

    resume_pool = set(resume.domains) | set(resume.skill_names)
    matched = [name for name in required if name in resume_pool]
    missing = [name for name in required if name not in resume_pool]

    if matched:
        score = round(len(matched) / len(required) * 100, 1)
        detail = f"JD 涉及 {len(required)} 个领域方向，简历命中 {len(matched)} 个（{score}%）。"
    elif resume.domains:
        # 有行业经验，但方向不同
        score = 45.0
        detail = (
            f"JD 方向为「{'、'.join(required)}」，"
            f"简历体现的行业背景是「{'、'.join(resume.domains)}」，方向不完全一致。"
        )
    else:
        score = 55.0
        detail = (
            f"JD 方向为「{'、'.join(required)}」，但简历里没有体现行业/领域信息，"
            "按中性分处理。建议在项目描述里点明业务场景（如「面向电商的推荐系统」）。"
        )

    evidence = [f"简历领域线索：{'、'.join(resume.domains)}"] if resume.domains else []
    gaps = [f"缺少「{name}」领域的经历体现" for name in missing] if matched else (
        [f"简历未体现「{'、'.join(required)}」行业背景"] if resume.domains else []
    )

    return DimensionScore(
        key="domain",
        label=DIMENSION_LABELS["domain"],
        score=score,
        weight=0.0,
        weighted=0.0,
        applicable=True,
        detail=detail,
        evidence=evidence,
        gaps=gaps,
        matched=matched,
        missing=missing,
    )


def score_project(
    jd: ParsedJD,
    resume: ParsedResume,
    normalizer: SkillNormalizer = DEFAULT_NORMALIZER,
) -> DimensionScore:
    """维度六：项目/经历相关度。

    这一维考察的是"纸上技能"与"项目里真用过"的距离，由两部分合成：

    - **技术重合度（80%）**：JD 提到的技术点里，有多少出现在简历的经历描述中。
      注意数据源是「经历里的技能」而不是「技能清单里的技能」——
      技能清单谁都会写，经历里的才是用过的。
    - **成果量化度（20%）**：带数字的成果条目占比。
      "优化了性能" 和 "把 P99 从 800ms 降到 120ms" 是两个量级。
    """
    exp_skills: set[str] = set()
    for exp in resume.experiences:
        # 展开上位技能：项目里写了 FAISS，就该算作覆盖了"向量数据库"
        exp_skills.update(normalizer.expand_with_parents(exp.skills))

    if not resume.experiences:
        return DimensionScore(
            key="project",
            label=DIMENSION_LABELS["project"],
            score=0.0,
            weight=0.0,
            weighted=0.0,
            applicable=False,
            detail="简历里没有识别到项目或工作经历，该维度不参与打分。",
        )

    jd_skills = set(jd.must_have) | set(jd.nice_to_have)
    overlap = sorted(jd_skills & exp_skills)
    tech_ratio = len(overlap) / len(jd_skills) if jd_skills else 0.0

    total_bullets = sum(len(exp.bullets) for exp in resume.experiences)
    quantified = sum(exp.quantified for exp in resume.experiences)
    quant_ratio = quantified / total_bullets if total_bullets else 0.0

    score = round((tech_ratio * 0.8 + quant_ratio * 0.2) * 100, 1)

    detail = (
        f"JD 涉及的 {len(jd_skills)} 个技术点中，有 {len(overlap)} 个出现在项目/实习经历里"
        f"（{round(tech_ratio * 100)}%）；"
        f"共 {total_bullets} 条经历描述，其中 {quantified} 条带有量化数据（{round(quant_ratio * 100)}%）。"
    )

    gaps: list[str] = []
    if total_bullets and quant_ratio < 0.3:
        gaps.append(
            f"量化成果偏少（{quantified}/{total_bullets} 条）。"
            "建议把「优化了性能」这类描述改成「P99 从 800ms 降到 120ms」这种带数字的形式。"
        )
    if jd_skills and tech_ratio < 0.5:
        missing_tech = sorted(jd_skills - exp_skills)
        gaps.append(
            f"有 {len(missing_tech)} 个 JD 技术点没在任何经历里出现：{'、'.join(missing_tech[:6])}"
        )

    evidence: list[str] = []
    for exp in resume.experiences[:3]:
        if not exp.org and not exp.title:
            continue
        label = exp.org or exp.title
        hit = [s for s in exp.skills if s in jd_skills]
        line = f"[{exp.period or '时间未标注'}] {label}"
        if hit:
            line += f" —— 命中 JD 技术点：{'、'.join(hit[:5])}"
        evidence.append(line)

    return DimensionScore(
        key="project",
        label=DIMENSION_LABELS["project"],
        score=score,
        weight=0.0,
        weighted=0.0,
        applicable=True,
        detail=detail,
        evidence=evidence,
        gaps=gaps,
        matched=overlap[:10],
        missing=sorted(jd_skills - exp_skills)[:10] if jd_skills else [],
    )


# ===========================================================================
# 三、权重归一化
# ===========================================================================


def normalize_weights(
    weights: Optional[dict[str, float]],
    applicable: dict[str, bool],
) -> dict[str, float]:
    """把权重归一化到"仅适用的维度、总和为 1"。

    Args:
        weights: 用户自定义权重；None 表示用默认值。
        applicable: 各维度是否适用。

    Returns:
        归一化后的权重字典。不适用维度的权重为 0.0。

    实现要点（也是这个函数存在的唯一理由）：
    只对**适用**的维度求和再归一化。这样当 JD 不写学历要求时，
    原本分给学历的 0.12 会按比例分给其他五个维度，而不是白白丢掉 ——
    否则"信息更少的 JD"会天然得到更低的分数，这毫无道理。
    """
    raw: dict[str, float] = {}
    for key in DIMENSION_ORDER:
        value = DEFAULT_WEIGHTS.get(key, 0.0)
        if weights and key in weights:
            try:
                value = float(weights[key])
            except (TypeError, ValueError):
                value = DEFAULT_WEIGHTS.get(key, 0.0)
        raw[key] = max(0.0, value)

    active_total = sum(raw[key] for key in DIMENSION_ORDER if applicable.get(key))

    # 退化情况：适用维度的权重之和 <= 0。它有两种成因，处理方式不同。
    if active_total <= 0:
        fallback = sum(DEFAULT_WEIGHTS.get(key, 0.0) for key in DIMENSION_ORDER if applicable.get(key))
        if fallback <= 0:
            # 成因一：**没有任何维度适用**（一份 JD 里什么都没识别出来）。
            # 这时返回全 0。调用方不能把它当成"完全不匹配"，而应看成"无法评估" ——
            # 所以真正的提示语在 `match()` 里，通过 risks 告诉用户"分数不代表真实匹配度"。
            #
            # 这里曾经有一段"让 must_have 兜底为 1.0"的分支，但它**永远走不到**：
            # 上面的 fallback <= 0 已经先一步 return 了。留着会让人误以为
            # 总分不会是 0，属于会误导后来者的死代码，直接删掉。
            return {key: 0.0 for key in DIMENSION_ORDER}
        # 成因二：用户把所有**适用**维度都拖到了 0。这时按默认权重的比例重新分配，
        # 而不是返回全 0 —— 否则界面上拖一下滑条就能把总分变成 0，体验很糟。
        return {
            key: (DEFAULT_WEIGHTS.get(key, 0.0) / fallback if applicable.get(key) else 0.0)
            for key in DIMENSION_ORDER
        }

    return {
        key: (raw[key] / active_total if applicable.get(key) else 0.0)
        for key in DIMENSION_ORDER
    }


# ===========================================================================
# 四、面试题库
# ===========================================================================

#: 技能 -> 深挖问题。覆盖 AI 应用开发岗最常被问的技术点。
#:
#: 这些问题不是"通用面试题大全"，而是**针对本岗位技能栈**挑选的，
#: 每一条都尽量做到"能从简历的一句话追问出真实水平"。
SKILL_QUESTIONS: dict[str, tuple[str, ...]] = {
    "Python": (
        "Python 的 GIL 对多线程的影响是什么？你的项目里是怎么绕开它的？",
        "说说你处理过的最大规模的数据处理任务，用了什么手段控制内存？",
    ),
    "RAG": (
        "你的 RAG 链路里，检索环节是怎么做的？为什么选这种检索方式？",
        "如果检索到的内容互相矛盾，你的系统会怎么处理？",
        "你怎么评测 RAG 的效果？召回率和答案质量分别怎么度量？",
    ),
    "LLM": (
        "你调用大模型时怎么控制成本和延迟？有没有做过缓存或者降级？",
        "模型输出的稳定性你是怎么保证的？温度、重试、校验分别做了什么？",
    ),
    "Agent": (
        "你的 Agent 是怎么决定调用哪个工具的？工具选错时怎么让它自己纠正？",
        "Agent 出现死循环或者无限调用工具时，你怎么截断？",
    ),
    "Function Calling": (
        "Function Calling 和 ReAct 文本协议各自的优劣是什么？你怎么选？",
        "模型编造了不存在的参数，你的系统怎么防住？",
    ),
    "Prompt Engineering": (
        "举一个你把提示词从效果差改到效果好的具体案例，改了什么？",
        "你怎么管理提示词的版本？改了之后怎么确认没让老场景变差？",
    ),
    "微调": (
        "你为什么选择微调而不是提示词工程？微调解决了什么提示词解决不了的问题？",
        "微调的数据集是怎么构造的？多少条？怎么保证质量？",
    ),
    "LoRA": (
        "LoRA 的原理是什么？rank 怎么选？调大调小分别有什么影响？",
        "QLoRA 相比 LoRA 省了什么？代价是什么？",
    ),
    "Embedding": (
        "你怎么评估一个 embedding 模型好不好？在你的业务上怎么验证？",
        "文本向量化时，长文本超过模型上限你怎么办？",
    ),
    "向量数据库": (
        "你用的向量索引是什么类型？为什么选它？",
        "向量检索的召回率怎么提升？你试过哪些手段？",
    ),
    "FAISS": (
        "FAISS 的 IndexFlatL2 和 IVF 系列有什么区别？你的场景用哪个？",
        "数据量到千万级之后，FAISS 的检索性能怎么保障？",
    ),
    "Milvus": (
        "Milvus 的 collection 和 partition 你是怎么设计的？",
        "Milvus 的检索参数 nprobe 调过吗？调它影响什么？",
    ),
    "BM25": (
        "BM25 和向量检索各自的优势场景是什么？融合的时候你怎么定权重？",
        "RRF 融合为什么比直接加权分数更稳？",
    ),
    "FastAPI": (
        "FastAPI 的依赖注入你是怎么用的？有没有用它做过鉴权？",
        "接口的响应时间是怎么优化的？有没有做过压测？",
    ),
    "SSE": (
        "SSE 和 WebSocket 你为什么会选 SSE？断线重连怎么处理？",
        "流式输出时如果中途出错，前端怎么感知？",
    ),
    "Docker": (
        "你的镜像有多少层？做过什么优化把镜像变小、构建变快？",
        "容器里的时区、编码、文件权限问题你踩过哪些坑？",
    ),
    "Kubernetes": (
        "你的服务是怎么做滚动更新的？怎么保证更新期间不丢请求？",
        "Pod 频繁重启你会从哪些地方排查？",
    ),
    "CI/CD": (
        "你的 CI 流程包含哪些环节？哪些是必过的门禁？",
        "怎么保证部署可以回滚？",
    ),
    "PyTorch": (
        "你处理过显存不足的问题吗？用了什么手段？",
        "训练和推理的代码你会怎么组织？有没有做过算子级别的优化？",
    ),
    "Hugging Face": (
        "你用 transformers 加载大模型时，device_map 和 dtype 是怎么设的？",
        "怎么做流式生成？generate 的哪些参数你会调？",
    ),
    "深度学习": (
        "你的模型出现过过拟合吗？怎么发现的、怎么解决的？",
        "训练过程中 loss 不下降，你会按什么顺序排查？",
    ),
    "自然语言处理": (
        "中文分词你会用哪种方案？为什么不用 jieba？",
        "文本分类任务上你试过哪些模型？效果差距有多大？",
    ),
    "计算机视觉": (
        "目标检测你用过哪个模型？mAP 是怎么提升的？",
        "数据增强你用过哪些？哪几种对小样本最有效？",
    ),
    "Git": (
        "团队协作时你怎么处理分支？rebase 和 merge 你倾向哪个？",
        "误提交了敏感信息，你会怎么处理？",
    ),
    "Linux": (
        "线上服务 CPU 飙高，你会用哪些命令定位？",
        "怎么看一个进程的内存到底被谁占了？",
    ),
    "MySQL": (
        "你优化过慢查询吗？怎么定位的、优化后提升了多少？",
        "索引失效的常见原因你知道哪些？",
    ),
    "Redis": (
        "Redis 的缓存穿透、击穿、雪崩你分别怎么防？",
        "你用的数据结构是什么？为什么不用另一种？",
    ),
    "并发编程": (
        "asyncio 里如果有一个阻塞调用会怎么样？你怎么发现和解决？",
        "多进程和多线程你的选择依据是什么？",
    ),
    "性能优化": (
        "你做过的最有价值的一次性能优化是什么？前后数据是多少？",
        "性能优化时你怎么确定瓶颈在哪？",
    ),
    "单元测试": (
        "你的测试覆盖率有多少？哪些部分你选择不测？",
        "怎么测试一个依赖外部 API 的函数？",
    ),
    "架构设计": (
        "你设计过的最复杂的系统是什么？画一下它的结构。",
        "如果流量涨 10 倍，你的系统哪里会先撑不住？",
    ),
    "HarmonyOS": (
        "ArkTS 的严格模式和 TypeScript 有什么区别？你踩过哪些坑？",
        "鸿蒙的分布式能力你在项目里用到了吗？怎么用的？",
    ),
    "JavaScript": (
        "闭包和 this 的指向你能举个例子说明吗？",
        "事件循环里微任务和宏任务的执行顺序是什么？",
    ),
    "C++": (
        "智能指针你用哪几种？循环引用怎么解决？",
        "vector 扩容的代价是什么？你怎么避免？",
    ),
    "开源项目": (
        "你这个开源项目有人提过 issue 或者 PR 吗？你是怎么处理的？",
        "如果有 100 个人用你的项目，你觉得第一个撑不住的地方是哪里？",
    ),
}

#: 通用兜底模板（题库里没有的技能）
_VERIFY_FALLBACK = (
    "你在项目里具体是怎么使用「{skill}」的？遇到过什么问题，最后怎么解决的？",
    "如果让你给一个不熟悉「{skill}」的同学讲清楚它，你会从哪三点讲起？",
)

_GAP_TEMPLATE = (
    "这个岗位要求「{skill}」，你目前的了解程度如何？如果两周内要上手，你打算怎么学？",
    "「{skill}」是你简历里没体现但岗位明确要求的能力。有没有相关的、可迁移的经验？",
)

_PROJECT_TEMPLATE = (
    "在「{org}」这段经历里，你遇到的最有挑战的技术问题是什么？怎么解决的？",
    "「{org}」这个项目如果重做一遍，你会在哪个环节做出不同的选择？",
)


def _pick(seq: tuple[str, ...], index: int) -> str:
    """按索引取模板，越界则回绕。用于让不同岗位拿到不同的问法。"""
    return seq[index % len(seq)]


def build_interview_questions(
    jd: ParsedJD,
    resume: ParsedResume,
    *,
    limit: int = 6,
    normalizer: SkillNormalizer = DEFAULT_NORMALIZER,
) -> list[InterviewQuestion]:
    """按匹配结果生成针对性面试问题。

    三类问题各有分工：

    - ``verify``：简历声称掌握、JD 也要求 —— 面试官会用来验证深度；
    - ``project``：针对具体项目经历 —— 用来挖真实贡献；
    - ``gap``：JD 要求但简历缺失 —— 用来考察学习能力与迁移能力。

    这三类问题对**求职者本人**同样有价值：它实际上是一份
    "你要被问到什么、你还缺什么"的清单。
    """
    if limit <= 0:
        return []

    questions: list[InterviewQuestion] = []
    strength = skill_strength_map(resume, normalizer)
    resume_skill_set = set(strength)

    # ---- 1. verify：JD 要求 ∩ 简历掌握 ----
    verify_targets = [name for name in jd.must_have if name in resume_skill_set]
    # 优先问那些简历里"写得最多"的技能（出现次数多 = 项目里用得深）
    appearance: dict[str, int] = {}
    for item in resume.skills:
        appearance[item.canonical] = appearance.get(item.canonical, 0) + 1
    verify_targets.sort(key=lambda name: -appearance.get(name, 0))

    for idx, skill in enumerate(verify_targets[: max(2, limit // 2)]):
        templates = SKILL_QUESTIONS.get(skill, _VERIFY_FALLBACK)
        questions.append(
            InterviewQuestion(
                kind="verify",
                question=_pick(templates, idx),
                skill=skill,
                rationale=f"该岗位的硬性要求，且你的简历里有「{skill}」的实际使用痕迹，面试官大概率会顺着这里深挖。",
                anchor=next(
                    (item.evidence for item in resume.skills if item.canonical == skill and item.evidence),
                    "",
                ),
            )
        )

    # ---- 2. project：针对最相关的一段经历 ----
    jd_skills = set(jd.must_have) | set(jd.nice_to_have)
    ranked_exp = sorted(
        resume.experiences,
        key=lambda exp: (-len(set(exp.skills) & jd_skills), -exp.quantified),
    )
    for idx, exp in enumerate([e for e in ranked_exp if e.org or e.title][:2]):
        label = exp.org or exp.title
        questions.append(
            InterviewQuestion(
                kind="project",
                question=_pick(_PROJECT_TEMPLATE, idx).format(org=label),
                skill="",
                rationale=(
                    f"这段经历与岗位技术要求重合度最高（命中 "
                    f"{len(set(exp.skills) & jd_skills)} 个技术点），面试官会重点追问细节。"
                ),
                anchor=f"{label}（{exp.period or '时间未标注'}）",
            )
        )

    # ---- 3. gap：JD 要求 ∩ 简历缺失 ----
    for skill in jd.must_have:
        if skill in resume_skill_set or strength.get(skill, 0.0) > 0:
            continue
        questions.append(
            InterviewQuestion(
                kind="gap",
                question=_pick(_GAP_TEMPLATE, len(questions)).format(skill=skill),
                skill=skill,
                rationale=f"「{skill}」是该岗位的硬性要求，但简历里完全没有体现，是最可能被质疑的点。",
                anchor=next((req.text for req in jd.requirements if skill in req.skills), ""),
            )
        )

    # 按 类型优先级 + 原始顺序 排序，然后截断
    kind_order = {"verify": 0, "project": 1, "gap": 2}
    questions.sort(key=lambda q: kind_order.get(q.kind, 9))
    return questions[:limit]


# ===========================================================================
# 五、结论、优势、风险与建议
# ===========================================================================

_VERDICTS: tuple[tuple[float, str, str], ...] = (
    (85.0, "强烈推荐", "A"),
    (70.0, "推荐", "B"),
    (55.0, "可考虑", "C"),
    (0.0, "匹配度偏低", "D"),
)


def judge_verdict(overall: float) -> tuple[str, str]:
    """按总分给出结论文字与等级。

    阈值是显式的、可调的常量，而不是藏在某个 magic number 里。
    """
    for threshold, label, level in _VERDICTS:
        if overall >= threshold:
            return label, level
    return "匹配度偏低", "D"


def build_strengths(dimensions: list[DimensionScore]) -> list[str]:
    """从维度分数里提炼优势陈述。"""
    out: list[str] = []
    for dim in dimensions:
        if not dim.applicable or dim.score < 80:
            continue
        if dim.key == "must_have":
            out.append(f"硬性技能覆盖度高（{dim.score} 分）：{_join(dim.matched, 6)}")
        elif dim.key == "nice_to_have":
            out.append(f"加分项覆盖不错（{dim.score} 分）：{_join(dim.matched, 6)}")
        elif dim.key == "experience":
            out.append("经验年限已满足岗位要求")
        elif dim.key == "education":
            out.append("学历达标")
        elif dim.key == "domain":
            out.append(f"行业方向对口：{_join(dim.matched, 4)}")
        elif dim.key == "project":
            out.append("项目经历与岗位技术栈重合度高，且成果有数据支撑")
    return out


def build_risks(dimensions: list[DimensionScore]) -> list[str]:
    """从维度分数与缺口里提炼风险陈述。"""
    out: list[str] = []
    for dim in dimensions:
        if not dim.applicable:
            continue
        if dim.key == "must_have":
            missing = [g for g in dim.gaps if g.startswith("缺失硬性技能")]
            if missing:
                out.append(f"存在 {len(missing)} 项硬性技能缺口，可能是简历筛选阶段的直接筛除项")
        elif dim.key == "experience" and dim.score < 70:
            out.append("经验年限低于岗位要求，需要在简历里更突出可迁移的实战经历")
        elif dim.key == "education" and dim.score < 70:
            out.append("学历未完全达到门槛，建议用项目与成果补强")
        elif dim.key == "domain" and dim.score < 60:
            out.append("行业背景与岗位方向不一致，面试时需准备好转岗动机的说明")
        elif dim.key == "project" and dim.score < 60:
            out.append("项目经历与岗位技术栈重合度不足，建议补充一个贴岗的项目")
    return out


def build_advice(
    jd: ParsedJD,
    resume: ParsedResume,
    dimensions: list[DimensionScore],
) -> list[str]:
    """生成可执行的改进建议。

    这里刻意不做"通用求职建议"（"建议多刷题""建议提升沟通能力"之类），
    每一条都必须**指向本次匹配检测出的具体问题**。
    """
    advice: list[str] = []

    # 1. 缺失的硬性技能
    must_dim = next((d for d in dimensions if d.key == "must_have"), None)
    if must_dim and must_dim.missing:
        top = must_dim.missing[:4]
        advice.append(
            f"补充这 {len(must_dim.missing)} 项硬性技能的凭证：{'、'.join(top)}"
            + ("等。" if len(must_dim.missing) > 4 else "。")
            + "哪怕只是跟着官方文档写一个小 demo 放进 GitHub，也比简历上留白要好。"
        )

    # 2. 弱可信技能（只在自我评价里出现过）
    weak_hints = [gap for gap in (must_dim.gaps if must_dim else []) if "低可信" in gap or "自我评价" in gap]
    if weak_hints:
        advice.append(
            "有技能只出现在「自我评价」这类低可信位置，建议把它们挪进「专业技能」清单"
            "或直接写进项目经历里——前者是声明，后者是证据，分量完全不同。"
        )

    # 3. 量化成果
    project_dim = next((d for d in dimensions if d.key == "project"), None)
    if project_dim and any("量化成果偏少" in gap for gap in project_dim.gaps):
        advice.append(
            "把项目描述里的定性表述改成定量表述。"
            "「优化了系统性能」→「接口 P99 从 800ms 降到 120ms」；"
            "「提升了检索效果」→「召回率从 62% 提升到 89%」。"
        )

    # 4. 行业背景缺失
    domain_dim = next((d for d in dimensions if d.key == "domain"), None)
    if domain_dim and domain_dim.applicable and domain_dim.score < 60:
        advice.append(
            f"在项目描述里点明业务场景。该岗位面向「{'、'.join(jd.domains)}」，"
            "哪怕你的项目是个人练习，也可以写成「面向 XX 场景的 XX 系统」。"
        )

    # 5. 岗位名称/关键词对齐
    if jd.title and jd.must_have:
        advice.append(
            f"把简历里的技能表述向 JD 用词对齐。这份 JD 用的词是"
            f"「{'、'.join(jd.must_have[:5])}」，如果你写的是同义词（如 k8s / Kubernetes），"
            "很多筛选系统只认其中一种写法——两种都写上最保险。"
        )

    # 6. 明显的强项，建议保留
    if project_dim and project_dim.score >= 75:
        advice.append(
            "项目经历与岗位技术栈重合度高，这是你的核心优势。"
            "建议把这部分经历放在简历最靠前的位置，并准备好在面试里展开讲。"
        )

    return advice[:6]


def _join(items: Iterable[str], limit: int = 5) -> str:
    """把人名列表拼成「A、B、C 等 N 项」。"""
    values = list(items)
    if not values:
        return "无"
    head = "、".join(values[:limit])
    return f"{head} 等 {len(values)} 项" if len(values) > limit else head


# ===========================================================================
# 六、主入口
# ===========================================================================


def match(
    jd: ParsedJD,
    resume: ParsedResume,
    *,
    weights: Optional[dict[str, float]] = None,
    want_questions: bool = True,
    question_count: int = 6,
    engine: str = "rule",
    normalizer: SkillNormalizer = DEFAULT_NORMALIZER,
) -> MatchReport:
    """执行完整的匹配打分，产出报告。

    Args:
        jd: 已解析的 JD。
        resume: 已解析的简历。
        weights: 自定义维度权重；None 用默认值。
        want_questions: 是否生成面试问题。
        question_count: 面试问题数量上限。
        engine: 解析引擎标识（记录在报告里，便于排查"为什么两次结果不同"）。
        normalizer: 技能归一化器（默认用全局词典）。

    Returns:
        MatchReport。
    """
    strength = skill_strength_map(resume, normalizer)

    dimensions = [
        score_must_have(jd, resume, strength),
        score_nice_to_have(jd, resume, strength),
        score_experience(jd, resume),
        score_education(jd, resume),
        score_domain(jd, resume),
        score_project(jd, resume, normalizer),
    ]

    applicable = {dim.key: dim.applicable for dim in dimensions}
    final_weights = normalize_weights(weights, applicable)

    # 把归一化后的权重回填，并算出加权贡献
    for dim in dimensions:
        dim.weight = round(final_weights.get(dim.key, 0.0), 4)
        dim.weighted = round(dim.score * dim.weight, 2)

    overall = round(sum(dim.weighted for dim in dimensions), 1)
    verdict, level = judge_verdict(overall)

    # ---- 技能集合运算 ----
    # 用 strength 的键集而不是 resume.skill_names：前者已包含上位技能
    resume_skill_set = set(strength)
    jd_skill_set = set(jd.must_have) | set(jd.nice_to_have)

    matched_skills = sorted(jd_skill_set & resume_skill_set, key=lambda n: _skill_rank(jd, n))
    missing_must = list(must_dim_missing(dimensions))
    missing_nice = [n for n in jd.nice_to_have if n not in resume_skill_set]
    extra_skills = sorted(
        resume_skill_set - jd_skill_set,
        key=lambda n: (-strength.get(n, 0.0), n),
    )

    questions = (
        build_interview_questions(jd, resume, limit=question_count, normalizer=normalizer)
        if want_questions
        else []
    )

    strengths = build_strengths(dimensions)
    risks = build_risks(dimensions)

    # 退化情况：所有维度都不适用（JD 里技能、年限、学历、领域一个都没识别出来，
    # 或者简历里一条经历都没有）。这时总分必然是 0，但"0 分"会被读成
    # "你完全不匹配"—— 那是**误导**，真实含义是"无法评估"。
    # 所以显式写进风险清单，让用户知道问题出在输入而不是自己。
    if not any(applicable.values()):
        risks.insert(
            0,
            "这份 JD 与简历里没有识别出任何可评估的维度（技能 / 年限 / 学历 / 行业 / 经历都没有），"
            "当前分数不代表真实匹配度，请检查文本是否完整、格式是否规整。",
        )

    return MatchReport(
        overall=overall,
        verdict=verdict,
        verdict_level=level,  # type: ignore[arg-type]
        dimensions=dimensions,
        weights={key: round(value, 4) for key, value in final_weights.items()},
        matched_skills=matched_skills,
        missing_must=missing_must,
        missing_nice=missing_nice,
        extra_skills=extra_skills,
        must_ratio=f"{len(jd.must_have) - len(missing_must)}/{len(jd.must_have)}",
        strengths=strengths,
        risks=risks,
        interview_questions=questions,
        advice=build_advice(jd, resume, dimensions),
        engine=engine,
    )


def must_dim_missing(dimensions: list[DimensionScore]) -> list[str]:
    """从维度结果里取出"完全缺失的硬性技能"。"""
    dim = next((d for d in dimensions if d.key == "must_have"), None)
    if not dim:
        return []
    return list(dim.missing)


def _skill_rank(jd: ParsedJD, name: str) -> int:
    """技能展示排序：硬性技能靠前，其次按 JD 中出现顺序。"""
    if name in jd.must_have:
        return jd.must_have.index(name)
    if name in jd.nice_to_have:
        return 1000 + jd.nice_to_have.index(name)
    return 9999


__all__ = [
    "match",
    "normalize_weights",
    "judge_verdict",
    "skill_strength_map",
    "effective_years",
    "build_interview_questions",
    "build_advice",
    "build_strengths",
    "build_risks",
    "score_must_have",
    "score_nice_to_have",
    "score_experience",
    "score_education",
    "score_domain",
    "score_project",
    "ORIGIN_WEIGHT",
    "SKILL_QUESTIONS",
    "STRONG_THRESHOLD",
]
