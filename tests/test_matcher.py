"""打分引擎测试。

这是整个项目的核心，测试也最厚。除了正常的分数断言，重点覆盖三类
"容易悄悄算错"的地方：

1. **权重再归一化**：不适用维度的权重必须按比例分给其余维度，
   否则"信息更少的 JD"会天然低分；
2. **可信度加权**：技能写在自我评价里和写在技能清单里，得分不该一样；
3. **可复现性**：同一份输入跑两次必须得到同一个结果 —— 这是招聘场景的硬要求。
"""

from __future__ import annotations

import pytest

from app.matcher import (
    ORIGIN_WEIGHT,
    STRONG_THRESHOLD,
    build_advice,
    build_interview_questions,
    build_risks,
    build_strengths,
    effective_years,
    judge_verdict,
    match,
    normalize_weights,
    score_domain,
    score_education,
    score_experience,
    score_must_have,
    score_nice_to_have,
    score_project,
    skill_strength_map,
)
from app.parser import parse_jd, parse_resume
from app.schemas import (
    DEFAULT_WEIGHTS,
    DIMENSION_ORDER,
    DimensionScore,
    ParsedJD,
    ParsedResume,
    SkillItem,
)

from .helpers import SIMPLE_JD, SIMPLE_RESUME


@pytest.fixture()
def jd() -> ParsedJD:
    return parse_jd(SIMPLE_JD)


@pytest.fixture()
def resume() -> ParsedResume:
    return parse_resume(SIMPLE_RESUME)


def make_resume(
    *skills: tuple[str, str, int | None],
    **overrides,
) -> ParsedResume:
    """快捷构造简历。

    Args:
        *skills: ``(技能名, 出现位置, 掌握程度分)`` 三元组。
        **overrides: 其它字段的直接覆盖。
    """
    items = [
        SkillItem(
            canonical=name,
            raw=name,
            origin=origin,
            level_score=level,
            evidence=f"用过 {name}",
        )
        for name, origin, level in skills
    ]
    data = {
        "skill_names": [item.canonical for item in items],
        "skills": items,
    }
    data.update(overrides)
    return ParsedResume(**data)


# ===========================================================================
# 技能可信度
# ===========================================================================


class TestSkillStrengthMap:
    def test_origin_weights_applied(self) -> None:
        resume = make_resume(
            ("Python", "skill_section", None),
            ("Redis", "summary", None),
        )
        strength = skill_strength_map(resume)
        assert strength["Python"] == ORIGIN_WEIGHT["skill_section"]
        assert strength["Redis"] == pytest.approx(ORIGIN_WEIGHT["summary"], abs=0.001)
        # 技能清单里的声明必须比自我评价里的顺口一提更可信
        assert strength["Python"] > strength["Redis"]

    def test_max_wins_when_skill_appears_twice(self) -> None:
        """同一技能出现多次取最高可信度 —— 有一处实打实就够了。"""
        resume = make_resume(
            ("Python", "summary", None),
            ("Python", "skill_section", None),
        )
        assert skill_strength_map(resume)["Python"] == ORIGIN_WEIGHT["skill_section"]

    def test_high_level_score_boosts(self) -> None:
        """程度词加成要能生效，但不能突破 1.0 的上限。"""
        # skill_section 本身已经是 1.0，加成只会被上限截住
        top = skill_strength_map(make_resume(("Python", "skill_section", 5)))["Python"]
        assert top == 1.0

        # 低可信位置上写「精通」，可信度应当被抬上来
        boosted = skill_strength_map(make_resume(("Python", "summary", 5)))["Python"]
        baseline = skill_strength_map(make_resume(("Python", "summary", None)))["Python"]
        assert boosted > baseline

    def test_low_level_score_discounts(self) -> None:
        """写了「了解」的技能不该和「精通」拿一样的分。"""
        strong = skill_strength_map(make_resume(("Docker", "skill_section", 5)))["Docker"]
        weak = skill_strength_map(make_resume(("Docker", "skill_section", 2)))["Docker"]
        assert weak < strong

    def test_strength_never_exceeds_one(self) -> None:
        resume = make_resume(("Python", "skill_section", 5))
        assert skill_strength_map(resume)["Python"] <= 1.0

    def test_parent_skill_inherited(self) -> None:
        """会 FAISS 就等于会用向量数据库。

        不做这步继承，JD 写「了解向量数据库（FAISS / Milvus）」时会自相矛盾：
        「FAISS」算命中，而它的上位词「向量数据库」算缺失。
        """
        strength = skill_strength_map(make_resume(("FAISS", "skill_section", None)))
        assert strength.get("向量数据库", 0.0) > 0

    def test_unknown_origin_uses_default(self) -> None:
        resume = make_resume(("Python", "某个没见过的位置", None))
        assert skill_strength_map(resume)["Python"] == 0.7

    def test_empty_resume(self) -> None:
        assert skill_strength_map(ParsedResume()) == {}


# ===========================================================================
# 权重归一化
# ===========================================================================


class TestNormalizeWeights:
    def test_all_applicable_matches_defaults(self) -> None:
        applicable = {key: True for key in DIMENSION_ORDER}
        result = normalize_weights(None, applicable)
        for key in DIMENSION_ORDER:
            assert result[key] == pytest.approx(DEFAULT_WEIGHTS[key], abs=0.001)
        assert sum(result.values()) == pytest.approx(1.0, abs=0.001)

    def test_inapplicable_dimension_gets_zero(self) -> None:
        applicable = {key: True for key in DIMENSION_ORDER}
        applicable["education"] = False
        result = normalize_weights(None, applicable)
        assert result["education"] == 0.0

    def test_weights_are_redistributed_to_applicable_dimensions(self) -> None:
        """**核心行为**：不适用的权重按比例分给其余维度，而不是白白丢掉。

        否则"没写学历要求的 JD"会因为 education 恒为 0 而天然低分。
        """
        applicable = {key: True for key in DIMENSION_ORDER}
        applicable["education"] = False
        result = normalize_weights(None, applicable)

        # 适用维度仍然归一化到 1
        assert sum(result.values()) == pytest.approx(1.0, abs=0.001)
        # must_have 相对占比上升了
        assert result["must_have"] > DEFAULT_WEIGHTS["must_have"]
        # 比例关系保持不变
        assert result["must_have"] / result["project"] == pytest.approx(
            DEFAULT_WEIGHTS["must_have"] / DEFAULT_WEIGHTS["project"], abs=0.01
        )

    def test_user_weights_override_defaults(self) -> None:
        """只传部分维度时，其余维度沿用默认权重（前端会传全量，脚本调用可以只传关心的）。"""
        applicable = {key: True for key in DIMENSION_ORDER}
        result = normalize_weights({"must_have": 3.0, "project": 1.0}, applicable)
        assert result["must_have"] / result["project"] == pytest.approx(3.0, abs=0.01)
        assert result["must_have"] > DEFAULT_WEIGHTS["must_have"]
        assert sum(result.values()) == pytest.approx(1.0, abs=0.001)

    def test_negative_and_invalid_weights_are_sanitized(self) -> None:
        applicable = {key: True for key in DIMENSION_ORDER}
        result = normalize_weights({"must_have": -5, "project": "不是数字"}, applicable)
        # 负数被夹到 0，坏值退回默认，最终依旧是一份合法权重
        assert sum(result.values()) == pytest.approx(1.0, abs=0.001)
        assert all(value >= 0 for value in result.values())

    def test_all_active_weights_zero_falls_back_to_defaults(self) -> None:
        """用户把所有适用维度都拖到 0 时，不能返回全 0（那会让总分为 0）。"""
        applicable = {key: True for key in DIMENSION_ORDER}
        result = normalize_weights({key: 0.0 for key in DIMENSION_ORDER}, applicable)
        assert sum(result.values()) == pytest.approx(1.0, abs=0.001)
        assert result["must_have"] > 0

    def test_no_dimension_applicable_returns_all_zero(self) -> None:
        """一个维度都不适用时返回全 0。

        这里曾经有一段"让 must_have 兜底为 1.0"的分支，但它被前面的
        ``fallback <= 0`` 先一步 return 掉了，永远走不到 —— 是死代码。
        删除之后的真实契约就是：全 0，由调用方（``match()``）负责把它
        解释成"无法评估"而不是"不匹配"。
        """
        applicable = {key: False for key in DIMENSION_ORDER}
        result = normalize_weights(None, applicable)
        assert result == {key: 0.0 for key in DIMENSION_ORDER}


# ===========================================================================
# 结论判定
# ===========================================================================


class TestJudgeVerdict:
    @pytest.mark.parametrize(
        ("score", "label", "level"),
        [
            (100.0, "强烈推荐", "A"),
            (85.0, "强烈推荐", "A"),
            (84.9, "推荐", "B"),
            (70.0, "推荐", "B"),
            (69.9, "可考虑", "C"),
            (55.0, "可考虑", "C"),
            (54.9, "匹配度偏低", "D"),
            (0.0, "匹配度偏低", "D"),
        ],
    )
    def test_thresholds(self, score: float, label: str, level: str) -> None:
        assert judge_verdict(score) == (label, level)


# ===========================================================================
# 维度一/二：技能覆盖
# ===========================================================================


class TestSkillDimensions:
    def test_full_coverage_is_100(self, jd: ParsedJD) -> None:
        resume = make_resume(*[(name, "skill_section", None) for name in jd.must_have])
        dim = score_must_have(jd, resume, skill_strength_map(resume))
        assert dim.score == 100.0
        assert dim.missing == []

    def test_no_coverage_is_zero(self, jd: ParsedJD) -> None:
        resume = ParsedResume()
        dim = score_must_have(jd, resume, {})
        assert dim.score == 0.0
        assert len(dim.missing) == len(jd.must_have)
        assert jd.must_have[0] in dim.missing[0]

    def test_weak_coverage_scores_lower_than_strong(self, jd: ParsedJD) -> None:
        """同样的命中项，写在自我评价里要比写在技能清单里得分低。"""
        names = jd.must_have
        strong = make_resume(*[(name, "skill_section", None) for name in names])
        weak = make_resume(*[(name, "summary", None) for name in names])

        strong_score = score_must_have(jd, strong, skill_strength_map(strong)).score
        weak_score = score_must_have(jd, weak, skill_strength_map(weak)).score
        assert weak_score < strong_score

    def test_weak_hits_are_listed_separately_from_missing(self, jd: ParsedJD) -> None:
        """低可信命中既不算"缺"，也不算"扎实" —— 必须单独提示用户挪个位置。"""
        resume = make_resume((jd.must_have[0], "summary", None))
        strength = skill_strength_map(resume)
        assert 0 < strength[jd.must_have[0]] < STRONG_THRESHOLD

        dim = score_must_have(jd, resume, strength)
        assert jd.must_have[0] in dim.matched
        assert jd.must_have[0] not in dim.missing
        assert any("低可信" in gap or "自我评价" in gap for gap in dim.gaps)

    def test_inapplicable_when_no_must_skills(self) -> None:
        jd = ParsedJD(title="某岗位")
        dim = score_must_have(jd, ParsedResume(), {})
        assert dim.applicable is False
        assert dim.score == 0.0
        assert "不参与打分" in dim.detail

    def test_nice_dimension_inapplicable_when_empty(self) -> None:
        dim = score_nice_to_have(ParsedJD(), ParsedResume(), {})
        assert dim.applicable is False

    def test_nice_dimension_scores_coverage(self, jd: ParsedJD) -> None:
        resume = make_resume((jd.nice_to_have[0], "skill_section", None))
        dim = score_nice_to_have(jd, resume, skill_strength_map(resume))
        expected = round(100 / len(jd.nice_to_have), 1)
        assert dim.score == pytest.approx(expected, abs=0.05)

    def test_evidence_is_attached(self, jd: ParsedJD, resume: ParsedResume) -> None:
        """分数必须能反查原文依据，这是"拒绝黑盒分数"的底线。"""
        strength = skill_strength_map(resume)
        dim = score_must_have(jd, resume, strength)
        if dim.matched:
            assert dim.evidence


# ===========================================================================
# 维度三：经验年限
# ===========================================================================


class TestEffectiveYears:
    def test_entry_level_uses_total_years(self) -> None:
        jd = ParsedJD(seniority="校招")
        resume = ParsedResume(total_years=1.5, work_years=0.0, internship_months=6, degree_expected=True)
        years, caliber = effective_years(jd, resume)
        assert years == 1.5
        assert "校招" in caliber

    def test_degree_expected_alone_triggers_entry_caliber(self) -> None:
        resume = ParsedResume(total_years=2.0, degree_expected=True)
        assert effective_years(ParsedJD(), resume)[0] == 2.0

    def test_social_caliber_discounts_internships_by_half(self) -> None:
        """社招看重全职产出，实习只按 50% 折算。"""
        resume = ParsedResume(total_years=9.0, work_years=4.0, internship_months=12, degree_expected=False)
        years, caliber = effective_years(ParsedJD(), resume)
        assert years == pytest.approx(4.0 + 12 / 12 * 0.5, abs=0.05)
        assert "50%" in caliber


class TestScoreExperience:
    def test_inapplicable_without_requirement(self) -> None:
        dim = score_experience(ParsedJD(), ParsedResume(total_years=5))
        assert dim.applicable is False

    def test_accepting_fresh_grads_is_full_score(self) -> None:
        jd = ParsedJD(min_years=0.0, seniority="校招")
        dim = score_experience(jd, ParsedResume(total_years=0.2, degree_expected=True))
        assert dim.score == 100.0

    def test_meeting_requirement_is_full_score(self) -> None:
        jd = ParsedJD(min_years=3.0)
        resume = ParsedResume(work_years=3.0, total_years=3.0)
        assert score_experience(jd, resume).score == 100.0

    def test_exceeding_requirement_caps_at_100(self) -> None:
        jd = ParsedJD(min_years=3.0)
        resume = ParsedResume(work_years=10.0, total_years=10.0)
        assert score_experience(jd, resume).score == 100.0

    def test_halfway_is_50(self) -> None:
        """折线设计：差一半时正好 50 分。"""
        jd = ParsedJD(min_years=4.0)
        resume = ParsedResume(work_years=2.0, total_years=2.0)
        assert score_experience(jd, resume).score == 50.0

    def test_below_half_is_proportional(self) -> None:
        jd = ParsedJD(min_years=4.0)
        resume = ParsedResume(work_years=1.0, total_years=1.0)
        assert score_experience(jd, resume).score == 25.0

    def test_no_experience_is_zero_and_gap_reported(self) -> None:
        jd = ParsedJD(min_years=3.0)
        dim = score_experience(jd, ParsedResume())
        assert dim.score == 0.0
        assert dim.gaps

    def test_evidence_lists_experiences(self) -> None:
        jd = ParsedJD(min_years=3.0)
        resume = parse_resume(SIMPLE_RESUME)
        dim = score_experience(jd, resume)
        assert dim.evidence

    def test_monotonic_in_experience(self) -> None:
        """年限越多分数不能越低 —— 分段折线最容易在拐点写反。"""
        jd = ParsedJD(min_years=4.0)
        scores = [
            score_experience(jd, ParsedResume(work_years=y, total_years=y)).score
            for y in (0.0, 1.0, 2.0, 3.0, 4.0, 6.0)
        ]
        assert scores == sorted(scores)
        assert scores[0] == 0.0 and scores[-1] == 100.0


# ===========================================================================
# 维度四：学历
# ===========================================================================


class TestScoreEducation:
    def test_inapplicable_without_requirement(self) -> None:
        assert score_education(ParsedJD(), ParsedResume(education_level=2)).applicable is False

    def test_meets_requirement(self) -> None:
        jd = ParsedJD(education_level=2, education="本科及以上")
        resume = ParsedResume(education_level=2, education="本科")
        dim = score_education(jd, resume)
        assert dim.score == 100.0
        assert dim.gaps == []

    def test_higher_degree_still_full_score(self) -> None:
        """要求本科、简历是硕士 —— 现实中这是加分，不能倒扣。"""
        jd = ParsedJD(education_level=2)
        assert score_education(jd, ParsedResume(education_level=3)).score == 100.0

    def test_one_level_short_is_lenient(self) -> None:
        jd = ParsedJD(education_level=3, education="硕士")
        dim = score_education(jd, ParsedResume(education_level=2, education="本科"))
        assert dim.score == 55.0
        assert dim.gaps

    def test_two_levels_short(self) -> None:
        jd = ParsedJD(education_level=3)
        assert score_education(jd, ParsedResume(education_level=1)).score == 20.0

    def test_no_education_in_resume(self) -> None:
        jd = ParsedJD(education_level=2)
        dim = score_education(jd, ParsedResume(education_level=0))
        assert dim.score == 0.0
        assert any("未写明学历" in gap for gap in dim.gaps)

    def test_evidence_includes_school_and_major(self) -> None:
        jd = ParsedJD(education_level=2, majors=["计算机"])
        resume = ParsedResume(
            education_level=2, education="本科", school="沈阳大学", major="人工智能", graduation_year=2027
        )
        dim = score_education(jd, resume)
        joined = " ".join(dim.evidence)
        assert "沈阳大学" in joined
        assert "人工智能" in joined
        assert "2027" in joined

    def test_major_overlap_is_reported(self) -> None:
        jd = ParsedJD(education_level=2, majors=["人工智能"])
        resume = ParsedResume(education_level=2, major="人工智能")
        assert any("专业对口" in item for item in score_education(jd, resume).evidence)


# ===========================================================================
# 维度五：行业领域
# ===========================================================================


class TestScoreDomain:
    def test_inapplicable_without_domain(self) -> None:
        assert score_domain(ParsedJD(), ParsedResume()).applicable is False

    def test_matched_domains_score(self) -> None:
        jd = ParsedJD(domains=["智能客服", "金融科技"])
        resume = ParsedResume(domains=["智能客服"], skill_names=["智能客服"])
        dim = score_domain(jd, resume)
        assert dim.score == 50.0
        assert dim.matched == ["智能客服"]

    def test_full_match(self) -> None:
        jd = ParsedJD(domains=["智能客服"])
        resume = ParsedResume(domains=["智能客服"], skill_names=["智能客服"])
        assert score_domain(jd, resume).score == 100.0

    def test_different_direction_is_neutral_low(self) -> None:
        jd = ParsedJD(domains=["金融科技"])
        resume = ParsedResume(domains=["电商"], skill_names=["电商"])
        assert score_domain(jd, resume).score == 45.0

    def test_no_domain_info_is_neutral_not_zero(self) -> None:
        """「没写」和「没有」是两回事 —— 很多简历确实不写行业背景，不该重罚。"""
        jd = ParsedJD(domains=["智能客服"])
        dim = score_domain(jd, ParsedResume())
        assert dim.score == 55.0
        assert "中性分" in dim.detail

    def test_skill_names_count_as_domain_evidence(self) -> None:
        """领域词常出现在技能列表里（如「智能客服」），也算线索。"""
        jd = ParsedJD(domains=["智能客服"])
        resume = ParsedResume(skill_names=["智能客服"])
        assert score_domain(jd, resume).score == 100.0


# ===========================================================================
# 维度六：项目相关度
# ===========================================================================


class TestScoreProject:
    def test_inapplicable_without_experiences(self) -> None:
        jd = ParsedJD(must_have=["Python"])
        assert score_project(jd, ParsedResume()).applicable is False

    def test_tech_and_quantified_both_count(self) -> None:
        """分数 = 技术重合度 × 80% + 成果量化度 × 20%。"""
        jd = parse_jd("任职要求：\n熟悉 Python、Docker。\n")
        resume = parse_resume(
            "工作经历\n某某公司 | 后端 | 2023.01-2024.01\n"
            "- 用 Python 写服务\n"
            "- 用 Docker 部署\n"
        )
        dim = score_project(jd, resume)
        assert dim.score > 0
        assert "技术点" in dim.detail
        assert "量化" in dim.detail

    def test_quantified_ratio_raises_score(self) -> None:
        """同样的技术重合度下，带数字的成果描述应当得分更高。"""
        jd = parse_jd("任职要求：\n熟悉 Python。\n")
        plain = parse_resume("工作经历\n某公司 | 开发 | 2023.01-2024.01\n- 用 Python 写服务\n- 用 Python 优化\n")
        quantified = parse_resume(
            "工作经历\n某公司 | 开发 | 2023.01-2024.01\n- 用 Python 写服务，QPS 提升 3 倍\n- 用 Python 优化，耗时下降 60%\n"
        )
        assert score_project(jd, quantified).score > score_project(jd, plain).score

    def test_low_quantification_is_flagged(self) -> None:
        jd = parse_jd("任职要求：\n熟悉 Python。\n")
        resume = parse_resume("工作经历\n某公司 | 开发 | 2023.01-2024.01\n- 用 Python 写服务\n- 用 Go 写服务\n")
        dim = score_project(jd, resume)
        assert any("量化成果偏少" in gap for gap in dim.gaps)

    def test_missing_tech_is_listed(self) -> None:
        jd = parse_jd("任职要求：\n熟悉 Python、Kubernetes、Milvus。\n")
        resume = parse_resume("工作经历\n某公司 | 开发 | 2023.01-2024.01\n- 用 Python 写服务\n")
        dim = score_project(jd, resume)
        assert any("没在任何经历里出现" in gap for gap in dim.gaps)

    def test_evidence_points_at_experiences(self) -> None:
        jd = parse_jd("任职要求：\n熟悉 Python。\n")
        resume = parse_resume("工作经历\n某某公司 | 开发 | 2023.01-2024.01\n- 用 Python 写服务\n")
        dim = score_project(jd, resume)
        assert any("某某公司" in item for item in dim.evidence)


# ===========================================================================
# 结论、优势、风险、建议
# ===========================================================================


class TestNarrative:
    def test_strengths_only_from_high_scores(self) -> None:
        dims = [
            DimensionScore(key="must_have", label="硬性技能", score=90, weight=0.5, weighted=45, matched=["Python"]),
            DimensionScore(key="project", label="项目相关度", score=40, weight=0.5, weighted=20),
        ]
        strengths = build_strengths(dims)
        assert len(strengths) == 1
        assert "硬性技能" in strengths[0]

    def test_strengths_ignore_inapplicable(self) -> None:
        dims = [
            DimensionScore(
                key="experience", label="经验年限", score=100, weight=0.0, weighted=0.0, applicable=False
            )
        ]
        assert build_strengths(dims) == []

    def test_risks_report_must_gaps(self) -> None:
        dims = [
            DimensionScore(
                key="must_have",
                label="硬性技能",
                score=40,
                weight=0.5,
                weighted=20,
                gaps=["缺失硬性技能：Kubernetes", "缺失硬性技能：Milvus"],
            )
        ]
        risks = build_risks(dims)
        assert risks and "2 项硬性技能缺口" in risks[0]

    def test_risks_for_low_education(self) -> None:
        dims = [
            DimensionScore(
                key="education", label="学历要求", score=20, weight=0.3, weighted=6,
                gaps=["学历不达标：要求硕士，简历本科"],
            )
        ]
        assert any("学历" in risk for risk in build_risks(dims))

    def test_advice_is_tied_to_concrete_gaps(self) -> None:
        """建议必须指向本次匹配检测出的具体问题，不能是通用求职话术。"""
        jd = parse_jd("岗位：AI 工程师\n任职要求：\n熟悉 Kubernetes、Milvus。\n")
        resume = parse_resume(SIMPLE_RESUME)
        report = match(jd, resume)
        assert report.advice
        assert any("Kubernetes" in item or "Milvus" in item for item in report.advice)

    def test_advice_capped(self, jd: ParsedJD, resume: ParsedResume) -> None:
        report = match(jd, resume)
        assert len(report.advice) <= 6


# ===========================================================================
# 面试问题
# ===========================================================================


class TestInterviewQuestions:
    def test_three_kinds_generated(self) -> None:
        jd = parse_jd(SIMPLE_JD)
        resume = parse_resume(SIMPLE_RESUME)
        questions = build_interview_questions(jd, resume, limit=12)
        kinds = {q.kind for q in questions}
        assert "verify" in kinds   # 简历有、JD 也要
        assert "project" in kinds  # 针对具体经历

    def test_gap_questions_for_missing_must_skills(self) -> None:
        """简历里完全没提的硬性技能，才生成「能力缺口」型问题。

        注意这里用的是 Milvus 而不是 Kubernetes ——
        SIMPLE_RESUME 的自我评价里恰好写了「熟悉 Kubernetes」，
        按 (kind, skill) 选中它并不会产生 gap 问题（那是 verify 的职责）。
        """
        jd = parse_jd("岗位：AI 工程师\n任职要求：\n精通 Milvus。\n")
        resume = parse_resume(SIMPLE_RESUME)
        questions = build_interview_questions(jd, resume, limit=12)
        gap = [q for q in questions if q.kind == "gap"]
        assert gap
        assert gap[0].skill == "Milvus"
        assert "Milvus" in gap[0].question

    def test_weakly_mentioned_skill_is_not_a_gap(self) -> None:
        """只在自我评价里提过的技能不该被当成"完全没有"。"""
        jd = parse_jd("岗位：AI 工程师\n任职要求：\n精通 Kubernetes。\n")
        resume = parse_resume(SIMPLE_RESUME)
        questions = build_interview_questions(jd, resume, limit=12)
        assert not [q for q in questions if q.kind == "gap" and q.skill == "Kubernetes"]

    def test_limit_respected(self) -> None:
        jd = parse_jd(SIMPLE_JD)
        resume = parse_resume(SIMPLE_RESUME)
        assert len(build_interview_questions(jd, resume, limit=3)) == 3

    def test_zero_limit(self) -> None:
        assert build_interview_questions(parse_jd(SIMPLE_JD), parse_resume(SIMPLE_RESUME), limit=0) == []

    def test_every_question_has_rationale(self) -> None:
        jd = parse_jd(SIMPLE_JD)
        resume = parse_resume(SIMPLE_RESUME)
        for q in build_interview_questions(jd, resume, limit=12):
            assert q.rationale, f"问题缺少理由：{q.question}"

    def test_kind_ordered_and_deterministic(self) -> None:
        jd = parse_jd(SIMPLE_JD)
        resume = parse_resume(SIMPLE_RESUME)
        first = build_interview_questions(jd, resume, limit=10)
        second = build_interview_questions(jd, resume, limit=10)
        assert [q.question for q in first] == [q.question for q in second]
        order = {"verify": 0, "project": 1, "gap": 2}
        keys = [order[q.kind] for q in first]
        assert keys == sorted(keys)


# ===========================================================================
# 端到端
# ===========================================================================


class TestMatchEndToEnd:
    def test_report_shape(self, jd: ParsedJD, resume: ParsedResume) -> None:
        report = match(jd, resume)
        assert 0 <= report.overall <= 100
        assert report.verdict
        assert report.verdict_level in {"A", "B", "C", "D"}
        assert len(report.dimensions) == len(DIMENSION_ORDER)
        assert [d.key for d in report.dimensions] == list(DIMENSION_ORDER)
        assert report.engine == "rule"

    def test_weights_sum_to_one(self, jd: ParsedJD, resume: ParsedResume) -> None:
        report = match(jd, resume)
        assert sum(report.weights.values()) == pytest.approx(1.0, abs=0.01)

    def test_weighted_contributions_sum_to_overall(self, jd: ParsedJD, resume: ParsedResume) -> None:
        """总分必须严格等于各维度贡献之和 —— 否则分数无法解释。"""
        report = match(jd, resume)
        assert report.overall == pytest.approx(sum(d.weighted for d in report.dimensions), abs=0.15)

    def test_must_and_nice_do_not_overlap(self, jd: ParsedJD, resume: ParsedResume) -> None:
        report = match(jd, resume)
        assert not (set(report.missing_must) & set(report.missing_nice))
        assert set(report.matched_skills) <= set(jd.must_have) | set(jd.nice_to_have)
        assert set(report.matched_skills) & set(report.missing_must) == set()

    def test_must_ratio_format(self, jd: ParsedJD) -> None:
        report = match(jd, parse_resume(SIMPLE_RESUME))
        hit, total = report.must_ratio.split("/")
        assert int(total) == len(jd.must_have)
        assert 0 <= int(hit) <= int(total)

    def test_extra_skills_sorted_by_strength(self) -> None:
        jd = parse_jd("任职要求：\n熟悉 Python。\n")
        resume = parse_resume(
            "专业技能\n- Python（精通）、Rust（了解）、COBOL\n"
        )
        report = match(jd, resume)
        assert "Rust" in report.extra_skills or "COBOL" in report.extra_skills
        # Python 是 JD 要求，不该出现在"简历额外技能"里
        assert "Python" not in report.extra_skills

    def test_questions_can_be_skipped(self, jd: ParsedJD, resume: ParsedResume) -> None:
        assert match(jd, resume, want_questions=False).interview_questions == []

    def test_question_count_respected(self, jd: ParsedJD, resume: ParsedResume) -> None:
        report = match(jd, resume, question_count=4)
        assert len(report.interview_questions) <= 4

    def test_weight_override_changes_overall(self, jd: ParsedJD, resume: ParsedResume) -> None:
        """只把 must_have 的权重拉到 1、其余归零时，总分应当收敛到该维度得分。"""
        only_must = {key: 0.0 for key in DIMENSION_ORDER}
        only_must["must_have"] = 1.0

        report = match(jd, resume, weights=only_must)
        must_dim = next(d for d in report.dimensions if d.key == "must_have")
        assert must_dim.weight == 1.0
        assert report.overall == pytest.approx(must_dim.score, abs=0.05)

    def test_shifting_weight_narrows_to_the_target_dimension(self) -> None:
        """把 90% 权重压到项目相关度上，总分应当被项目得分主导。"""
        shifted = {key: 0.0 for key in DIMENSION_ORDER}
        shifted["project"] = 0.9
        shifted["must_have"] = 0.1

        jd = parse_jd("任职要求：\n熟悉 Python、Milvus。\n")
        resume = parse_resume(SIMPLE_RESUME)
        report = match(jd, resume, weights=shifted)
        project_dim = next(d for d in report.dimensions if d.key == "project")
        assert report.overall == pytest.approx(project_dim.score * 0.9 + report.dimensions[0].score * 0.1, abs=0.1)

    def test_results_are_deterministic(self, jd: ParsedJD, resume: ParsedResume) -> None:
        """**招聘场景的硬要求**：同一份输入跑两次必须完全一致。

        分数只要有一点点随机性，候选人申诉时就没法解释。
        """
        first = match(jd, resume)
        second = match(jd, resume)
        assert first.model_dump() == second.model_dump()

    def test_better_resume_scores_higher(self) -> None:
        """匹配分必须真的能区分简历质量，否则这个系统毫无意义。"""
        jd = parse_jd(
            "岗位：AI 应用开发工程师\n"
            "任职要求：\n精通 Python、FastAPI；\n熟悉 RAG、Docker。\n"
        )
        strong = parse_resume(
            "李强\n专业技能\n- Python（精通）、FastAPI、RAG、Docker\n"
            "工作经历\n某公司 | 后端 | 2022.01-2025.01\n"
            "- 用 Python + FastAPI 搭 RAG 服务，QPS 提升 3 倍\n"
        )
        weak = parse_resume("王弱\n专业技能\n- Excel\n")
        assert match(jd, strong).overall > match(jd, weak).overall

    def test_matching_job_family_beats_mismatched_one(self) -> None:
        """同一份简历，投对口岗位的分数必须高于投不对口的岗位。"""
        resume = parse_resume(
            "张三\n专业技能\n- 编程语言：Python（精通）、Go（了解）\n"
            "- 框架：FastAPI、Django\n"
            "- 数据库：MySQL、Redis\n"
        )
        ai_jd = parse_jd("岗位：AI 应用开发工程师\n任职要求：\n精通 Python、RAG、向量数据库。\n")
        backend_jd = parse_jd("岗位：Python 后端开发工程师\n任职要求：\n精通 Python、FastAPI、MySQL、Redis。\n")
        assert match(backend_jd, resume).overall > match(ai_jd, resume).overall

    def test_empty_jd_does_not_crash_and_explains_itself(self) -> None:
        """极端输入不能抛异常，而且必须说明"0 分"的真实含义。

        总分 0 会被读成"你完全不匹配"。当所有维度都不适用时，
        真实含义是"无法评估"—— 这个区别必须写进报告，否则就是误导用户。
        """
        report = match(ParsedJD(), ParsedResume())
        assert report.overall == 0.0
        assert all(not dim.applicable for dim in report.dimensions)
        assert any("不代表真实匹配度" in risk for risk in report.risks)

    def test_no_experience_resume_explains_itself(self) -> None:
        """只有简历空白、JD 正常时，不该出现"无法评估"的误导性提示。"""
        jd = parse_jd(SIMPLE_JD)
        report = match(jd, ParsedResume())
        assert any(dim.applicable for dim in report.dimensions)
        assert not any("不代表真实匹配度" in risk for risk in report.risks)

    def test_missing_must_matches_dimension_missing(self, jd: ParsedJD, resume: ParsedResume) -> None:
        report = match(jd, resume)
        must_dim = next(d for d in report.dimensions if d.key == "must_have")
        assert report.missing_must == must_dim.missing

    def test_example_fixtures_produce_sane_report(self) -> None:
        """用仓库里的真实示例跑一遍，防止打分曲线被改坏到离谱。"""
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        jd = parse_jd((root / "examples" / "jd_ai_app_engineer.txt").read_text(encoding="utf-8"))
        good = parse_resume((root / "examples" / "resume_zhangsan.txt").read_text(encoding="utf-8"))
        bad = parse_resume((root / "examples" / "resume_lisi.txt").read_text(encoding="utf-8"))

        good_report = match(jd, good)
        bad_report = match(jd, bad)

        assert 0 <= bad_report.overall < good_report.overall <= 100
        assert good_report.verdict_level in {"A", "B", "C", "D"}
        # 硬性技能缺口必须能体现在风险里
        if good_report.missing_must:
            assert any("硬性技能缺口" in risk for risk in good_report.risks)
