"""解析器测试：JD 与简历的结构化抽取。

解析器是"从自然语言到结构化数据"的翻译层，也是错误最容易传导到
下游（打分）的一层。这里的测试既有正向用例（该抽出来的抽出来），
也有大量负向用例（不该抽出来的别乱抽）。
"""

from __future__ import annotations

import pytest

from app.parser import (
    build_experience_items,
    dedupe_skill_names,
    detect_education_max,
    detect_education_min,
    detect_seniority,
    detect_years_requirement,
    find_majors,
    find_major,
    find_name,
    find_school,
    judge_demand,
    locate_skills,
    merge_llm_skills,
    merge_month_ranges,
    parse_entry_header,
    parse_jd,
    parse_resume,
    split_experience_entries,
    _clause_at,
)
from app import sections as sec
from app.sections import split_sections
from app.textutil import normalize_text

from .helpers import MINIMAL_JD, MINIMAL_RESUME, SIMPLE_JD, SIMPLE_RESUME


# ===========================================================================
# 独立检测函数
# ===========================================================================


class TestEducationDetection:
    def test_jd_takes_lowest_level(self) -> None:
        """JD 写「本科及以上」时，要求的是**本科**（硕士当然也满足）。

        这是刻意的：取最高会得到"硕士"这个错误结论，
        导致本科学历的候选人被判成不达标。
        """
        text, level = detect_education_min("本科及以上学历，硕士优先")
        assert level == 2
        assert "本科" in text

    def test_jd_unlimited(self) -> None:
        assert detect_education_min("学历不限，欢迎各背景候选人")[1] == 0

    def test_jd_no_mention(self) -> None:
        assert detect_education_min("熟悉 Python 即可")[1] == 0

    def test_resume_takes_highest_level(self) -> None:
        """简历要取最高学历。"""
        assert detect_education_max("本科 某某大学\n硕士 某某研究院")[1] == 3

    def test_resume_bachelor(self) -> None:
        assert detect_education_max("某某大学 计算机 本科")[1] == 2

    def test_phrase_is_short(self) -> None:
        """学历原文应当是短语，不是整行。"""
        text, _ = detect_education_min("1. 本科及以上学历，计算机相关专业，2027 届毕业生；")
        assert len(text) <= 20
        assert "计算机相关专业" not in text

    def test_list_prefix_stripped(self) -> None:
        text, _ = detect_education_min("1. 本科及以上学历")
        assert not text.startswith("1")


class TestYearsRequirement:
    @pytest.mark.parametrize(
        ("text", "lo", "hi"),
        [
            ("3 年以上后端开发经验", 3.0, None),
            ("要求 5 年及以上工作经验", 5.0, None),
            ("3-5 年开发经验", 3.0, 5.0),
            ("1~3年经验", 1.0, 3.0),
            ("至少有 2 年相关经验", 2.0, None),
        ],
    )
    def test_parsing(self, text: str, lo: float, hi: float | None) -> None:
        assert detect_years_requirement(text) == (lo, hi)

    def test_entry_level(self) -> None:
        assert detect_years_requirement("欢迎应届毕业生投递") == (0.0, None)

    def test_no_requirement(self) -> None:
        assert detect_years_requirement("熟悉 Python 即可") == (None, None)

    def test_reversed_range_normalized(self) -> None:
        lo, hi = detect_years_requirement("5-3 年经验")
        assert lo == 3.0 and hi == 5.0


class TestSeniority:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("招聘算法实习生", "实习"),
            ("2027 届校园招聘", "校招"),
            ("欢迎应届毕业生", "校招"),
            ("资深后端工程师", "高级"),
            ("高级 Python 工程师", "高级"),
            ("初级开发工程师", "初级"),
            ("招聘 Python 工程师", ""),
        ],
    )
    def test_detection(self, text: str, expected: str) -> None:
        assert detect_seniority(text) == expected


class TestSchoolMajorName:
    def test_find_school(self) -> None:
        assert find_school("沈阳大学 | 人工智能 | 本科") == "沈阳大学"
        assert find_school("某某职业技术学院 计算机") == "某某职业技术学院"

    def test_find_school_none(self) -> None:
        assert find_school("没有学校信息") == ""

    def test_find_major_from_label(self) -> None:
        assert find_major("专业：软件工程") == "软件工程"

    def test_find_major_from_keyword(self) -> None:
        assert find_major("本科 人工智能 方向") == "人工智能"

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("张三\n电话 138", "张三"),
            ("姓名：李四", "李四"),
            ("姓名: 王五", "王五"),
            ("张三 | 男 | 1999", "张三"),
            ("John Smith", "John Smith"),
        ],
    )
    def test_find_name(self, text: str, expected: str) -> None:
        index = split_sections(normalize_text(text))
        assert find_name(index.preamble, text) == expected

    def test_find_name_skips_resume_title(self) -> None:
        """第一行是"个人简历"时不能当成姓名。"""
        text = "个人简历\n张三\n电话：138"
        index = split_sections(normalize_text(text))
        assert find_name(index.preamble, text) == "张三"

    def test_find_name_not_fooled_by_school(self) -> None:
        text = "某某大学\n计算机专业"
        index = split_sections(normalize_text(text))
        # "某某大学" 长度超过中文姓名上限（4 字），不该被当成姓名
        assert find_name(index.preamble, text) == ""


class TestMajorKeywords:
    def test_longest_wins(self) -> None:
        """「计算机科学与技术」不该被同时算成「计算机」两项。"""
        majors = find_majors("专业：计算机科学与技术")
        assert majors == ["计算机科学与技术"]

    def test_multiple(self) -> None:
        majors = find_majors("计算机、软件工程、人工智能相关专业")
        assert "软件工程" in majors and "人工智能" in majors

    def test_limit(self) -> None:
        assert len(find_majors("计算机 软件工程 人工智能 数学 统计学 自动化 通信工程", limit=3)) == 3


# ===========================================================================
# must / nice 判定（子句级）
# ===========================================================================


class TestDemandJudgement:
    @pytest.mark.parametrize(
        ("clause", "section", "expected"),
        [
            ("熟悉 Python", sec.KIND_REQUIREMENT, "must"),
            ("精通 Java", sec.KIND_REQUIREMENT, "must"),
            ("掌握 Go", sec.KIND_REQUIREMENT, "must"),
            ("了解 Rust", sec.KIND_REQUIREMENT, "nice"),
            ("熟悉 Kubernetes 者优先", sec.KIND_REQUIREMENT, "nice"),
            ("加分：会用 Docker", sec.KIND_REQUIREMENT, "nice"),
            ("没有任何强度词的技术栈", sec.KIND_REQUIREMENT, "must"),
            ("没有任何强度词的技术栈", sec.KIND_BONUS, "nice"),
        ],
    )
    def test_judge_demand(self, clause: str, section: str, expected: str) -> None:
        assert judge_demand(clause, section) == expected

    def test_must_wins_over_nice_in_same_clause(self) -> None:
        """同一子句里既有"必须"又有"优先"时取 must（宁可高估要求）。"""
        assert judge_demand("必须熟悉 Python，熟练掌握者优先", sec.KIND_REQUIREMENT) == "must"


class TestClauseExtraction:
    def test_splits_on_chinese_comma(self) -> None:
        text = "熟悉 RAG 技术栈，了解向量数据库（FAISS / Milvus）的使用"
        idx = text.index("向量数据库")
        clause = _clause_at(text, idx)
        assert clause.startswith("了解向量数据库")
        assert "RAG" not in clause

    def test_keeps_dunhao_together(self) -> None:
        """顿号表示并列，不该切断 —— 否则会丢掉前面的修饰词「精通」。"""
        text = "精通 Python、PyTorch、TensorFlow"
        idx = text.index("PyTorch")
        clause = _clause_at(text, idx)
        assert "精通" in clause

    def test_respects_line_boundary(self) -> None:
        text = "第一行内容\n熟悉 Python"
        assert _clause_at(text, text.index("Python")) == "熟悉 Python"


class TestSubClauseDemandInPractice:
    """集成测试：验证「熟悉 X，了解 Y」这类句子的正确拆解。"""

    def test_familiar_vs_understand(self) -> None:
        jd = parse_jd("任职要求：\n熟悉 RAG 技术栈，了解向量数据库（FAISS / Milvus）的使用；\n")
        assert "RAG" in jd.must_have
        assert "向量数据库" in jd.nice_to_have
        assert "FAISS" in jd.nice_to_have
        assert "Milvus" in jd.nice_to_have

    def test_must_upgrade_on_repeat(self) -> None:
        """同一技能在"加分项"和"任职要求"里都出现时，取更强的 must。"""
        jd = parse_jd(
            "岗位职责：\n使用 Docker 部署服务。\n\n"
            "任职要求：\n熟悉 Docker。\n\n"
            "加分项：\n了解 Docker 者优先。\n"
        )
        assert "Docker" in jd.must_have
        assert "Docker" not in jd.nice_to_have


# ===========================================================================
# JD 解析
# ===========================================================================


class TestParseJd:
    def test_title_and_company(self) -> None:
        jd = parse_jd(SIMPLE_JD)
        assert "后端开发工程师" in jd.title
        assert jd.company == "某某科技"

    def test_must_and_nice(self) -> None:
        jd = parse_jd(SIMPLE_JD)
        assert "Python" in jd.must_have
        assert "FastAPI" in jd.must_have
        assert "MySQL" in jd.must_have
        assert "Redis" in jd.must_have
        assert "Kubernetes" in jd.nice_to_have
        assert "Rust" in jd.nice_to_have

    def test_no_overlap_between_must_and_nice(self) -> None:
        jd = parse_jd(SIMPLE_JD)
        assert not (set(jd.must_have) & set(jd.nice_to_have))

    def test_years_and_education(self) -> None:
        jd = parse_jd(SIMPLE_JD)
        assert jd.min_years == 3.0
        assert jd.education_level == 2

    def test_majors(self) -> None:
        jd = parse_jd(SIMPLE_JD)
        assert "计算机" in jd.majors

    def test_requirements_classified(self) -> None:
        jd = parse_jd(SIMPLE_JD)
        assert jd.requirements
        categories = {req.category for req in jd.requirements}
        assert "技能" in categories
        assert "学历" in categories
        assert "经验" in categories

    def test_requirements_have_kind(self) -> None:
        jd = parse_jd(SIMPLE_JD)
        kinds = {req.kind for req in jd.requirements}
        assert kinds <= {"must", "nice"}
        assert "must" in kinds and "nice" in kinds

    def test_responsibilities(self) -> None:
        jd = parse_jd(SIMPLE_JD)
        assert jd.responsibilities
        assert any("后端服务" in item for item in jd.responsibilities)

    def test_skills_carry_evidence_and_offset(self) -> None:
        jd = parse_jd(SIMPLE_JD)
        python_hits = [s for s in jd.skills if s.canonical == "Python"]
        assert python_hits
        hit = python_hits[0]
        assert hit.evidence
        assert hit.start is not None and hit.end is not None
        assert jd.raw_text[hit.start : hit.end].lower() == "python"

    def test_raw_text_is_normalized(self) -> None:
        jd = parse_jd("岗位：测试\r\n\r\n任职要求：\r\n熟悉　Python")
        assert "\r" not in jd.raw_text
        assert "\u3000" not in jd.raw_text  # 全角空格已转

    def test_char_count(self) -> None:
        jd = parse_jd(SIMPLE_JD)
        assert jd.char_count == len(jd.raw_text)

    def test_minimal_jd(self) -> None:
        jd = parse_jd(MINIMAL_JD)
        assert "Python" in jd.must_have
        assert "FastAPI" in jd.must_have

    def test_empty_jd_does_not_crash(self) -> None:
        jd = parse_jd("")
        assert jd.must_have == []
        assert jd.char_count == 0

    def test_jd_without_section_headings(self) -> None:
        """没有小标题的 JD 也要能抽出技能（走兜底分支）。"""
        jd = parse_jd("我们在找一位熟悉 Python 和 FastAPI、了解 Docker 的同学加入团队。")
        assert "Python" in jd.must_have
        assert "FastAPI" in jd.must_have

    def test_domains_detected(self) -> None:
        jd = parse_jd("任职要求：\n有金融科技或电商行业经验者优先。")
        assert jd.domains

    def test_seniority_from_title(self) -> None:
        jd = parse_jd("岗位：算法实习生\n任职要求：\n熟悉 Python。")
        assert jd.seniority == "实习"


# ===========================================================================
# 简历解析
# ===========================================================================


class TestParseResume:
    def test_basics(self) -> None:
        resume = parse_resume(SIMPLE_RESUME)
        assert resume.name == "王五"
        assert resume.education_level == 2
        assert resume.school == "某某大学"
        assert resume.major in ("软件工程", "计算机")

    def test_graduation_year(self) -> None:
        resume = parse_resume(SIMPLE_RESUME)
        assert resume.graduation_year == 2023

    def test_not_degree_expected(self) -> None:
        """2023 年毕业、已在工作 → 不是应届。"""
        assert parse_resume(SIMPLE_RESUME).degree_expected is False

    def test_degree_expected_for_student(self) -> None:
        text = "张三\n教育背景\n某某大学 人工智能 本科 2023.09-2027.06\n预计 2027 年毕业"
        assert parse_resume(text).degree_expected is True

    def test_skills_with_level(self) -> None:
        resume = parse_resume(SIMPLE_RESUME)
        python = [s for s in resume.skills if s.canonical == "Python"]
        assert python
        assert any(s.level == "精通" for s in python)

    def test_skill_origin_recorded(self) -> None:
        """技能要记录出现在哪类章节里 —— 这决定了它的可信度。"""
        resume = parse_resume(SIMPLE_RESUME)
        origins = {s.origin for s in resume.skills}
        assert "skill_section" in origins
        assert "experience" in origins

    def test_skill_groups(self) -> None:
        resume = parse_resume(SIMPLE_RESUME)
        assert "编程语言" in resume.skill_groups
        assert "Python" in resume.skill_groups["编程语言"]

    def test_experiences_parsed(self) -> None:
        resume = parse_resume(SIMPLE_RESUME)
        assert len(resume.experiences) >= 1
        work = [e for e in resume.experiences if e.kind == "work"]
        assert work
        first = work[0]
        assert first.org == "某某公司"
        assert "后端开发工程师" in first.title
        assert first.ongoing is True

    def test_bullets_collected(self) -> None:
        resume = parse_resume(SIMPLE_RESUME)
        work = [e for e in resume.experiences if e.kind == "work"][0]
        assert len(work.bullets) >= 2
        assert work.quantified >= 1  # "下降 60%"

    def test_total_years_positive(self) -> None:
        resume = parse_resume(SIMPLE_RESUME)
        assert resume.total_years > 2.0

    def test_highlights_prefer_quantified(self) -> None:
        resume = parse_resume(SIMPLE_RESUME)
        assert resume.highlights
        assert any("60%" in h for h in resume.highlights)

    def test_minimal_resume(self) -> None:
        resume = parse_resume(MINIMAL_RESUME)
        assert resume.name == "赵六"
        assert "Python" in resume.skill_names
        assert "FastAPI" in resume.skill_names

    def test_empty_resume(self) -> None:
        resume = parse_resume("")
        assert resume.skill_names == []
        assert resume.experiences == []
        assert resume.char_count == 0

    def test_offsets_valid(self) -> None:
        resume = parse_resume(SIMPLE_RESUME)
        for skill in resume.skills:
            assert skill.start is not None and skill.end is not None
            assert resume.raw_text[skill.start : skill.end] == skill.raw


# ===========================================================================
# 经历切分
# ===========================================================================


class TestExperienceSplitting:
    def test_two_entries_with_dates(self) -> None:
        lines = [
            "某某科技 | 后端开发 | 2023.07-2024.06",
            "- 负责接口开发",
            "- 优化性能",
            "某某公司 | 实习生 | 2022.09-2023.05",
            "- 编写测试",
        ]
        entries = split_experience_entries(lines)
        assert len(entries) == 2
        assert len(entries[0].bullets) == 2
        assert len(entries[1].bullets) == 1

    def test_entry_without_date(self) -> None:
        lines = ["智能问答系统 后端负责人", "- 完成 RAG 全链路开发", "- 实现流式输出"]
        entries = split_experience_entries(lines)
        assert len(entries) == 1
        assert entries[0].header == "智能问答系统 后端负责人"
        assert len(entries[0].bullets) == 2

    def test_two_line_header_merged(self) -> None:
        """条目头写成两行的情况要合并，而不是被拆成两个条目。"""
        lines = ["某科技公司", "算法实习生 2024.07-2024.10", "- 参与开发"]
        entries = split_experience_entries(lines)
        assert len(entries) == 1
        assert "某科技公司" in entries[0].header
        assert "算法实习生" in entries[0].header

    def test_long_line_becomes_content(self) -> None:
        """标题已经够长时，后面来的行应当归入正文而不是继续拼标题。"""
        long_header = "某" * 70 + " 2024.01-2024.12"
        lines = [long_header, "这是一行明显是正文的内容" * 3]
        entries = split_experience_entries(lines)
        assert len(entries) == 1
        assert entries[0].header == long_header
        assert entries[0].bullets

    def test_empty_lines_ignored(self) -> None:
        entries = split_experience_entries(["", "  ", "\n"])
        assert entries == []

    def test_header_is_date_bearing(self) -> None:
        entries = split_experience_entries(["2023.07-2024.06 某某公司", "- 工作内容"])
        assert len(entries) == 1


class TestEntryHeaderParsing:
    @pytest.mark.parametrize(
        ("header", "org", "title"),
        [
            ("某某科技有限公司 | 后端开发实习生 | 2024.07-2024.10", "某某科技有限公司", "后端开发实习生"),
            ("智能问答系统  后端负责人  2024.03-2024.06", "智能问答系统", "后端负责人"),
            ("某公司  算法工程师", "某公司", "算法工程师"),
        ],
    )
    def test_pipe_and_space_formats(self, header: str, org: str, title: str) -> None:
        parsed = parse_entry_header(header)
        assert parsed["org"] == org
        assert parsed["title"] == title

    def test_date_extracted(self) -> None:
        parsed = parse_entry_header("某公司 | 开发 | 2024.07-2024.10")
        assert parsed["period"] == "2024.07-2024.10"
        assert parsed["date"] is not None

    def test_parenthesis_role(self) -> None:
        parsed = parse_entry_header("智能问答系统（后端开发）")
        assert parsed["org"] == "智能问答系统"
        assert parsed["title"] == "后端开发"

    def test_empty_header(self) -> None:
        parsed = parse_entry_header("")
        assert parsed["org"] == "" and parsed["title"] == ""


# ===========================================================================
# 年限合并
# ===========================================================================


class TestMonthMerging:
    def test_no_overlap(self) -> None:
        assert merge_month_ranges([(0, 5), (10, 15)]) == 12

    def test_overlapping(self) -> None:
        """重叠区间必须合并 —— 否则并行经历会被重复计算，年限虚高。"""
        assert merge_month_ranges([(0, 5), (3, 9)]) == 10

    def test_adjacent_merged(self) -> None:
        assert merge_month_ranges([(0, 5), (6, 10)]) == 11

    def test_empty(self) -> None:
        assert merge_month_ranges([]) == 0

    def test_invalid_range_ignored(self) -> None:
        assert merge_month_ranges([(10, 5)]) == 0

    def test_absurd_range_ignored(self) -> None:
        """超过 30 年的区间视为脏数据（日期写错了）。"""
        assert merge_month_ranges([(0, 12 * 50)]) == 0

    def test_unsorted_input(self) -> None:
        assert merge_month_ranges([(10, 15), (0, 5)]) == 12

    def test_single(self) -> None:
        assert merge_month_ranges([(0, 11)]) == 12


class TestOverlapAffectsResumeYears:
    def test_parallel_internships_not_double_counted(self) -> None:
        """两段并行的实习不该被简单相加。"""
        text = """\
张三
实习经历
A 公司 | 实习生 | 2024.01-2024.06
- 做 A 事
B 公司 | 实习生 | 2024.03-2024.09
- 做 B 事
"""
        resume = parse_resume(text)
        # 两段各 6 个月，重叠 4 个月 → 实际覆盖 9 个月 = 0.75 年
        assert resume.internship_months == 9


# ===========================================================================
# LLM 结果合并
# ===========================================================================


class TestMergeLlmSkills:
    def test_merge_adds_new_skill(self) -> None:
        resume = parse_resume(SIMPLE_RESUME)
        merged = merge_llm_skills(
            resume.skills, ["COBOL"], resume.raw_text, normalizer=__import__("app.skills", fromlist=["x"]).DEFAULT_NORMALIZER
        )
        assert any(item.canonical == "COBOL" for item in merged)

    def test_merge_upgrades_source_on_overlap(self) -> None:
        from app.skills import DEFAULT_NORMALIZER

        resume = parse_resume(SIMPLE_RESUME)
        merged = merge_llm_skills(resume.skills, ["Python"], resume.raw_text, normalizer=DEFAULT_NORMALIZER)
        python_items = [item for item in merged if item.canonical == "Python"]
        assert any(item.source == "dict+llm" for item in python_items)

    def test_merge_normalizes_llm_output(self) -> None:
        """模型给的别名也要过一遍词典归一化。"""
        from app.skills import DEFAULT_NORMALIZER

        resume = parse_resume(MINIMAL_RESUME)
        merged = merge_llm_skills(resume.skills, ["js"], resume.raw_text, normalizer=DEFAULT_NORMALIZER)
        assert any(item.canonical == "JavaScript" for item in merged)

    def test_merge_dedupes(self) -> None:
        from app.skills import DEFAULT_NORMALIZER

        resume = parse_resume(MINIMAL_RESUME)
        merged = merge_llm_skills(resume.skills, ["Python", "python", "PYTHON"], resume.raw_text, normalizer=DEFAULT_NORMALIZER)
        assert len([i for i in merged if i.canonical == "Python"]) == 1

    def test_merge_keeps_order(self) -> None:
        from app.skills import DEFAULT_NORMALIZER

        resume = parse_resume(MINIMAL_RESUME)
        merged = merge_llm_skills(resume.skills, ["Zzz新技能"], resume.raw_text, normalizer=DEFAULT_NORMALIZER)
        assert merged[-1].canonical == "Zzz新技能"

    def test_parse_jd_with_llm_data(self) -> None:
        jd = parse_jd(MINIMAL_JD, llm_data={"title": "AI 工程师", "skills": ["LangChain"]})
        assert jd.title == "AI 工程师"
        assert "LangChain" in jd.must_have

    def test_parse_resume_with_llm_data(self) -> None:
        resume = parse_resume(MINIMAL_RESUME, llm_data={"name": "赵六六", "skills": ["vLLM"]})
        assert resume.name == "赵六六"
        assert "vLLM" in resume.skill_names

    def test_llm_cannot_forge_offsets(self) -> None:
        """模型给的技能如果在原文里找不到，offset 必须留空，而不是编一个。"""
        from app.skills import DEFAULT_NORMALIZER

        resume = parse_resume(MINIMAL_RESUME)
        merged = merge_llm_skills(resume.skills, ["完全不存在的技能"], resume.raw_text, normalizer=DEFAULT_NORMALIZER)
        forged = next(i for i in merged if i.canonical == "完全不存在的技能")
        assert forged.start is None
        assert forged.evidence == ""


# ===========================================================================
# 工具函数
# ===========================================================================


class TestHelpers:
    def test_dedupe_skill_names(self) -> None:
        resume = parse_resume(SIMPLE_RESUME)
        names = dedupe_skill_names(resume.skills)
        assert len(names) == len(set(names))

    def test_locate_skills_origin(self) -> None:
        from app.skills import DEFAULT_NORMALIZER

        text = normalize_text(SIMPLE_RESUME)
        index = split_sections(text)
        items = locate_skills(text, index, DEFAULT_NORMALIZER, with_level=True)
        assert items
        assert all(item.origin for item in items)

    def test_build_experience_items_kinds(self) -> None:
        from app.skills import DEFAULT_NORMALIZER

        text = normalize_text(
            "工作经历\nA 公司 | 开发 | 2023.01-2024.01\n- 做事\n\n"
            "项目经历\n某系统 | 负责人 | 2022.01-2022.12\n- 做事"
        )
        index = split_sections(text)
        items = build_experience_items(index, DEFAULT_NORMALIZER)
        kinds = {item.kind for item in items}
        assert "work" in kinds and "project" in kinds
