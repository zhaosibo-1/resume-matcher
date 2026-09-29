"""前后端唯一的数据契约。

本项目所有跨层传递的数据结构都定义在这里，包含三个方向：

1. **解析产物**：``ParsedJD`` / ``ParsedResume`` —— 非结构化文本的结构化结果；
2. **打分产物**：``MatchReport`` / ``DimensionScore`` —— 可解释的匹配报告；
3. **接口出入参**：``MatchRequest`` / ``ConfigResponse`` 等。

为什么要把它们放在一个文件里？
因为"前端读的字段"和"后端返回的字段"最容易漂移 —— 后端改个名字、
前端照旧读，界面就会静默显示空白（这个坑在项目 1 里踩过一次）。
集中定义之后，写一个契约测试扫一遍前端源码里访问到的字段名，
就能在 CI 阶段把这类问题拦住。
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 通用
# ---------------------------------------------------------------------------

#: 打分的六个维度 key 与中文标签。
#: 顺序即前端雷达图的绘制顺序，改动时要同步 web/index.html 里的 DIM_ORDER。
DIMENSION_ORDER: tuple[str, ...] = (
    "must_have",
    "nice_to_have",
    "experience",
    "education",
    "domain",
    "project",
)

DIMENSION_LABELS: dict[str, str] = {
    "must_have": "硬性技能",
    "nice_to_have": "加分技能",
    "experience": "经验年限",
    "education": "学历要求",
    "domain": "行业领域",
    "project": "项目相关度",
}

DIMENSION_DESCRIPTIONS: dict[str, str] = {
    "must_have": "JD 里以『必须/要求/精通』等措辞提出的技能，简历覆盖了多少。权重最高。",
    "nice_to_have": "JD 里以『加分/优先/了解』提出的技能，属于锦上添花。",
    "experience": "简历累计工作/实习年限与 JD 要求年限的差距。JD 未写年限时该维度不参与打分。",
    "education": "学历等级是否达标（大专 < 本科 < 硕士 < 博士）。JD 不限时该维度不参与打分。",
    "domain": "JD 所属行业领域关键词在简历里的覆盖情况，判断『做过同类业务』。",
    "project": "项目/实习经历与 JD 技能的重合度，并考察成果是否有量化数据支撑。",
}

#: 默认权重（总和必须为 1.0，代码里会做归一化）。
DEFAULT_WEIGHTS: dict[str, float] = {
    "must_have": 0.35,
    "nice_to_have": 0.12,
    "experience": 0.18,
    "education": 0.12,
    "domain": 0.13,
    "project": 0.10,
}


class DimensionDef(BaseModel):
    """维度定义，供前端渲染权重滑条与说明气泡。"""

    key: str
    label: str
    description: str
    default_weight: float


# ---------------------------------------------------------------------------
# 技能
# ---------------------------------------------------------------------------


class SkillItem(BaseModel):
    """一次技能命中及其上下文信息。"""

    canonical: str = Field(description="归一化后的标准名，如 JavaScript")
    raw: str = Field(default="", description="原文里的实际写法，如 js")
    category: str = Field(default="其它", description="技能大类")
    level: Optional[str] = Field(default=None, description="掌握程度词，如 精通 / 熟练 / 了解")
    level_score: Optional[int] = Field(default=None, ge=1, le=5, description="掌握程度分值 1~5")
    demand: Optional[str] = Field(default=None, description="JD 侧需求强度：must / nice")
    origin: str = Field(default="other", description="出现位置：skill_section / experience / summary / other")
    evidence: str = Field(default="", description="命中所处的句子片段，用于向用户展示依据")
    start: Optional[int] = Field(default=None, description="在原文中的起始 offset")
    end: Optional[int] = Field(default=None, description="在原文中的结束 offset")
    source: str = Field(default="dict", description="来源：dict（词典）/ llm（模型抽取）")


# ---------------------------------------------------------------------------
# JD 解析结果
# ---------------------------------------------------------------------------


class Requirement(BaseModel):
    """从 JD 里抽出来的一条要求（通常对应一个条目或一句话）。"""

    text: str
    kind: Literal["must", "nice"] = "must"
    category: str = Field(default="技能", description="技能 / 经验 / 学历 / 项目 / 软技能 / 职责")
    skills: list[str] = Field(default_factory=list, description="该条要求涉及的归一化技能")


class ParsedJD(BaseModel):
    """结构化后的职位描述。"""

    title: str = Field(default="", description="岗位名称")
    company: str = Field(default="", description="公司名（能识别就填）")
    seniority: str = Field(default="", description="招聘层级：实习 / 校招 / 初级 / 中级 / 高级")
    min_years: Optional[float] = Field(default=None, description="要求的最低经验年限")
    max_years: Optional[float] = Field(default=None, description="要求的最高经验年限")
    education: str = Field(default="", description="学历要求原文，如 本科及以上")
    education_level: int = Field(default=0, description="学历等级 0~4，0 表示不限")
    majors: list[str] = Field(default_factory=list, description="专业要求关键词")
    must_have: list[str] = Field(default_factory=list, description="硬性技能（归一化名）")
    nice_to_have: list[str] = Field(default_factory=list, description="加分技能（归一化名）")
    skills: list[SkillItem] = Field(default_factory=list, description="全部技能命中明细")
    domains: list[str] = Field(default_factory=list, description="行业/领域关键词")
    requirements: list[Requirement] = Field(default_factory=list, description="拆解后的要求条目")
    responsibilities: list[str] = Field(default_factory=list, description="岗位职责条目")
    raw_text: str = Field(default="", description="原始文本")
    char_count: int = Field(default=0)


# ---------------------------------------------------------------------------
# 简历解析结果
# ---------------------------------------------------------------------------


class ExperienceItem(BaseModel):
    """一段工作 / 实习 / 项目 / 校园经历。"""

    kind: Literal["work", "internship", "project", "campus"] = "project"
    org: str = Field(default="", description="公司 / 项目名")
    title: str = Field(default="", description="职位 / 角色")
    period: str = Field(default="", description="时间区间原文，如 2024.03-2024.09")
    start_year: Optional[int] = None
    start_month: Optional[int] = None
    end_year: Optional[int] = None
    end_month: Optional[int] = None
    ongoing: bool = Field(default=False, description="是否『至今』")
    duration_months: Optional[int] = Field(default=None, description="估算时长（月）")
    bullets: list[str] = Field(default_factory=list, description="该经历下的条目（已去掉符号）")
    skills: list[str] = Field(default_factory=list, description="该经历涉及的归一化技能")
    quantified: int = Field(default=0, description="含量化数据（数字/百分比）的条目数")
    index: int = Field(default=0, description="在简历中的出现顺序，供前端定位高亮")


class ParsedResume(BaseModel):
    """结构化后的简历。"""

    name: str = Field(default="", description="姓名（识别不到则为空）")
    education: str = Field(default="", description="最高学历原文")
    education_level: int = Field(default=0, description="学历等级 0~4")
    school: str = Field(default="", description="学校")
    major: str = Field(default="", description="专业")
    graduation_year: Optional[int] = Field(default=None, description="毕业年份")
    degree_expected: bool = Field(default=False, description="是否应届/在校生（决定经验年限的判定方式）")
    total_years: float = Field(default=0.0, description="累计相关经验年限（含实习折算）")
    work_years: float = Field(default=0.0, description="纯工作年限（不含实习）")
    internship_months: float = Field(default=0.0, description="实习累计月数")
    skills: list[SkillItem] = Field(default_factory=list, description="技能命中明细")
    skill_names: list[str] = Field(default_factory=list, description="归一化后的技能名列表")
    skill_groups: dict[str, list[str]] = Field(default_factory=dict, description="按大类分组的技能")
    experiences: list[ExperienceItem] = Field(default_factory=list)
    highlights: list[str] = Field(default_factory=list, description="识别出的亮点句（含量化成果等）")
    domains: list[str] = Field(default_factory=list, description="行业/领域关键词")
    raw_text: str = Field(default="", description="原始文本")
    char_count: int = Field(default=0)


# ---------------------------------------------------------------------------
# 打分结果
# ---------------------------------------------------------------------------


class DimensionScore(BaseModel):
    """单个维度的得分与依据。

    这个结构是整个项目的核心卖点：**拒绝黑盒分数**。
    每个维度都要能回答三个问题：
      - 得了多少分（score）
      - 为什么是这个分（detail + evidence）
      - 差在哪里（gaps）
    """

    key: str
    label: str
    score: float = Field(ge=0.0, le=100.0, description="该维度原始分 0~100")
    weight: float = Field(ge=0.0, description="参与加权时的实际权重（已归一化）")
    weighted: float = Field(ge=0.0, description="score * weight，即对总分的贡献")
    applicable: bool = Field(default=True, description="False 表示该维度不适用（如 JD 未写年限），不参与加权")
    detail: str = Field(default="", description="一句话解释这个分数怎么来的")
    evidence: list[str] = Field(default_factory=list, description="支撑得分的原文片段")
    gaps: list[str] = Field(default_factory=list, description="该维度下的缺口清单")
    matched: list[str] = Field(default_factory=list, description="命中的条目")
    missing: list[str] = Field(default_factory=list, description="缺失的条目")


class InterviewQuestion(BaseModel):
    """一道面试题及其生成理由。"""

    kind: Literal["verify", "gap", "project", "basics"] = "verify"
    question: str
    skill: str = Field(default="", description="关联技能，无关联则为空")
    rationale: str = Field(default="", description="为什么问这道题")
    anchor: str = Field(default="", description="锚点：简历原文片段或 JD 要求原文")


class MatchReport(BaseModel):
    """完整匹配报告。"""

    overall: float = Field(ge=0.0, le=100.0, description="加权总分 0~100")
    verdict: str = Field(description="结论文字，如 强烈推荐 / 推荐 / 可考虑 / 不建议")
    verdict_level: Literal["A", "B", "C", "D"] = "C"
    dimensions: list[DimensionScore] = Field(default_factory=list)
    weights: dict[str, float] = Field(default_factory=dict, description="本次实际生效的权重")
    matched_skills: list[str] = Field(default_factory=list, description="双方都提到的技能")
    missing_must: list[str] = Field(default_factory=list, description="JD 必须但简历没有的技能")
    missing_nice: list[str] = Field(default_factory=list, description="JD 加分但简历没有的技能")
    extra_skills: list[str] = Field(default_factory=list, description="简历有但 JD 没提的技能（可能超出预期）")
    must_ratio: str = Field(default="0/0", description="硬性技能命中比，如 6/8")
    strengths: list[str] = Field(default_factory=list, description="优势陈述")
    risks: list[str] = Field(default_factory=list, description="风险/短板陈述")
    interview_questions: list[InterviewQuestion] = Field(default_factory=list)
    advice: list[str] = Field(default_factory=list, description="可执行的改进建议")
    engine: str = Field(default="rule", description="打分引擎：rule（规则）/ llm（模型参与抽取）")


# ---------------------------------------------------------------------------
# 接口出入参
# ---------------------------------------------------------------------------


class DimensionWeight(BaseModel):
    """单个维度的权重项，用于回显生效权重。"""

    key: str
    label: str
    weight: float
    applicable: bool = True


class MatchRequest(BaseModel):
    """单份简历 × 单个 JD 的匹配请求。"""

    jd_text: str = Field(min_length=1, description="职位描述原文")
    resume_text: str = Field(default="", description="简历原文（粘贴文本时使用）")
    resume_id: str = Field(default="", description="已上传简历的 id；与 resume_text 二选一")
    weights: Optional[dict[str, float]] = Field(default=None, description="自定义权重，缺省用默认权重")
    want_questions: bool = Field(default=True, description="是否生成面试问题")
    question_count: int = Field(default=6, ge=0, le=20, description="面试问题数量上限")


class MatchResponse(BaseModel):
    """匹配结果：把解析产物与打分报告一起返回，前端一次拿全。"""

    jd: ParsedJD
    resume: ParsedResume
    report: MatchReport
    elapsed_ms: int = 0


class BatchResumeInput(BaseModel):
    """批量匹配中的单份简历。"""

    resume_id: str = ""
    name: str = ""
    resume_text: str = Field(min_length=1)


class BatchMatchRequest(BaseModel):
    """一个 JD × 多份简历的批量匹配（HR 视角的排序场景）。"""

    jd_text: str = Field(min_length=1)
    resumes: list[BatchResumeInput] = Field(min_length=1, max_length=50)
    weights: Optional[dict[str, float]] = None


class BatchMatchItem(BaseModel):
    resume_id: str
    name: str
    overall: float
    verdict: str
    verdict_level: str
    must_ratio: str
    missing_must: list[str] = Field(default_factory=list)
    top_strength: str = ""


class BatchMatchResponse(BaseModel):
    jd_title: str
    total: int
    items: list[BatchMatchItem] = Field(default_factory=list)
    elapsed_ms: int = 0


class RecomputeRequest(BaseModel):
    """只重算权重，不重新解析。

    这是很有用的一个接口：用户拖动权重滑条时，解析结果没有变，
    没必要再跑一遍解析（更没必要再调一次大模型）。前端把上一次的
    打分输入缓存起来，只重算加权和即可 —— 响应从秒级降到毫秒级。
    """

    jd: ParsedJD
    resume: ParsedResume
    weights: Optional[dict[str, float]] = None
    want_questions: bool = False
    question_count: int = 6


class ResumeUploadResponse(BaseModel):
    resume_id: str
    name: str
    char_count: int
    preview: str = Field(default="", description="前 200 字的预览")


class ResumeListItem(BaseModel):
    resume_id: str
    name: str
    char_count: int
    uploaded_at: str = ""


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = ""
    llm_enabled: bool = False
    skill_count: int = 0


class ConfigResponse(BaseModel):
    """前端启动时拉取的配置（Key 一律脱敏）。"""

    llm_enabled: bool
    provider: str
    model: str
    base_url: str
    engine: str = Field(description="当前生效的解析引擎：rule / llm")
    dimensions: list[DimensionDef] = Field(default_factory=list)
    default_weights: dict[str, float] = Field(default_factory=dict)
    max_upload_mb: int = 5
    skill_count: int = 0
    categories: list[str] = Field(default_factory=list)
    seniority_options: list[str] = Field(default_factory=list)
    version: str = ""


class StatsResponse(BaseModel):
    """服务累计统计，用于前端展示"已分析 N 份简历"。"""

    resumes_stored: int = 0
    matches_run: int = 0
    skills_in_dict: int = 0
    llm_calls: int = 0
    llm_failures: int = 0
    uptime_seconds: int = 0


class ExampleItem(BaseModel):
    key: str
    label: str
    description: str
    kind: Literal["jd", "resume"]


class ExampleDetail(BaseModel):
    key: str
    label: str
    kind: str
    text: str


__all__ = [
    "DIMENSION_ORDER",
    "DIMENSION_LABELS",
    "DIMENSION_DESCRIPTIONS",
    "DEFAULT_WEIGHTS",
    "DimensionDef",
    "SkillItem",
    "Requirement",
    "ParsedJD",
    "ExperienceItem",
    "ParsedResume",
    "DimensionScore",
    "InterviewQuestion",
    "MatchReport",
    "DimensionWeight",
    "MatchRequest",
    "MatchResponse",
    "BatchResumeInput",
    "BatchMatchRequest",
    "BatchMatchItem",
    "BatchMatchResponse",
    "RecomputeRequest",
    "ResumeUploadResponse",
    "ResumeListItem",
    "HealthResponse",
    "ConfigResponse",
    "StatsResponse",
    "ExampleItem",
    "ExampleDetail",
]
