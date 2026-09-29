"""章节识别层测试。

这一层决定"这段话属于哪个板块"，而板块决定了技能的权重
（技能清单里的 Python 比自我评价里的 Python 可信得多）。
所以既要测"能识别出来"，也要测"不该识别的不乱识别"。
"""

from __future__ import annotations

import pytest

from app import sections as sec
from app.sections import split_sections, match_heading, strip_sections_for_skills

RESUME = """\
张三
电话：138-0000-0000 | 邮箱：a@b.com

教育背景
某某大学 | 计算机科学与技术 | 本科 | 2023.09-2027.06
- 主修课程：数据结构、操作系统

实习经历
某某科技 | 算法实习生 | 2024.07-2024.10
- 参与智能客服开发
- 使用 FAISS 构建索引

项目经历
知识库问答助手 | 后端负责人 | 2024.03-2024.06
- 完成 RAG 全链路开发

专业技能
- 编程语言：Python、C++
- 框架：FastAPI

荣誉奖项
- 校级一等奖

自我评价
对 AI 应用开发有浓厚兴趣，熟悉 Python 生态。
"""

JD = """\
岗位：AI 应用开发工程师

岗位职责：
1. 负责大模型应用的开发；
2. 参与 RAG 链路搭建。

任职要求：
1. 本科及以上学历；
2. 熟悉 Python 与 FastAPI。

加分项：
1. 熟悉 Kubernetes 者优先。

福利待遇：
- 六险一金、免费三餐。
"""


class TestHeadingMatching:
    @pytest.mark.parametrize(
        ("line", "kind"),
        [
            ("教育背景", sec.KIND_EDUCATION),
            ("教育经历", sec.KIND_EDUCATION),
            ("学历背景", sec.KIND_EDUCATION),
            ("## 教育背景", sec.KIND_EDUCATION),
            ("【教育背景】", sec.KIND_EDUCATION),
            ("一、教育背景", sec.KIND_EDUCATION),
            ("实习经历", sec.KIND_INTERNSHIP),
            ("实习经验", sec.KIND_INTERNSHIP),
            ("工作经历", sec.KIND_WORK),
            ("项目经历", sec.KIND_PROJECT),
            ("项目经验", sec.KIND_PROJECT),
            ("专业技能", sec.KIND_SKILL),
            ("技术栈", sec.KIND_SKILL),
            ("荣誉奖项", sec.KIND_AWARD),
            ("自我评价", sec.KIND_SUMMARY),
            ("个人总结", sec.KIND_SUMMARY),
            ("岗位职责", sec.KIND_DUTY),
            ("任职要求", sec.KIND_REQUIREMENT),
            ("任职要求（急招）", sec.KIND_REQUIREMENT),
            ("加分项", sec.KIND_BONUS),
            ("福利待遇", sec.KIND_BENEFIT),
        ],
    )
    def test_known_headings(self, line: str, kind: str) -> None:
        matched = match_heading(line)
        assert matched is not None, f"{line!r} 应当是标题"
        assert matched[0] == kind

    @pytest.mark.parametrize(
        "line",
        [
            "负责工作流的搭建",                     # 正文里出现"工作"二字
            "参与了项目经历相关的讨论",              # 长句，不是标题
            "教育背景是某某大学计算机专业，成绩优异。",  # 带句号
            "这个项目的技术栈，我们用的是 FastAPI",     # 带逗号
            "123456",
            "",
            "   ",
            "这是一段比较长的正文描述，长度明显超过标题的合理范围，不应该被识别成章节标题。",
        ],
    )
    def test_non_headings(self, line: str) -> None:
        assert match_heading(line) is None, f"{line!r} 不该被当成标题"

    def test_trailing_paren_stripped(self) -> None:
        assert match_heading("专业技能（选填）") == (sec.KIND_SKILL, "专业技能")

    def test_multiple_parens(self) -> None:
        assert match_heading("任职要求（杭州）（急招）") == (sec.KIND_REQUIREMENT, "任职要求")


class TestSplitSections:
    def test_resume_sections(self) -> None:
        index = split_sections(RESUME)
        kinds = index.kinds()
        for expected in (
            sec.KIND_EDUCATION,
            sec.KIND_INTERNSHIP,
            sec.KIND_PROJECT,
            sec.KIND_SKILL,
            sec.KIND_AWARD,
            sec.KIND_SUMMARY,
        ):
            assert expected in kinds, f"缺少章节 {expected}"

    def test_preamble_collects_header(self) -> None:
        """姓名/联系方式常写在第一个标题之前，必须收进 preamble。"""
        index = split_sections(RESUME)
        assert "张三" in index.preamble
        assert "138-0000-0000" in index.preamble

    def test_section_body_excludes_heading(self) -> None:
        index = split_sections(RESUME)
        edu = index.first(sec.KIND_EDUCATION)
        assert edu is not None
        assert "教育背景" not in edu.body
        assert "某某大学" in edu.body

    def test_section_text_includes_heading(self) -> None:
        index = split_sections(RESUME)
        skill_sec = index.first(sec.KIND_SKILL)
        assert skill_sec is not None
        assert skill_sec.text.startswith("专业技能")

    def test_jd_sections(self) -> None:
        index = split_sections(JD)
        assert index.has(sec.KIND_DUTY)
        assert index.has(sec.KIND_REQUIREMENT)
        assert index.has(sec.KIND_BONUS)
        assert index.has(sec.KIND_BENEFIT)

    def test_inline_heading_with_content(self) -> None:
        """「教育背景：某某大学」这种同行形式也要能切开。"""
        index = split_sections("教育背景：某某大学 计算机 本科\n\n专业技能：Python")
        edu = index.first(sec.KIND_EDUCATION)
        assert edu is not None
        assert "某某大学" in edu.body
        assert index.has(sec.KIND_SKILL)

    def test_markdown_resume(self) -> None:
        text = "# 张三\n\n## 教育背景\n- 某某大学\n\n## 专业技能\n- Python"
        index = split_sections(text)
        assert index.has(sec.KIND_EDUCATION)
        assert index.has(sec.KIND_SKILL)
        assert "张三" in index.preamble

    def test_no_sections_at_all(self) -> None:
        """完全无结构的文本：全部归入 preamble，不能崩。"""
        index = split_sections("这是一段没有任何标题的纯文本。")
        assert index.sections == []
        assert "纯文本" in index.preamble

    def test_empty_text(self) -> None:
        index = split_sections("")
        assert index.sections == []
        assert index.preamble == ""

    def test_offsets_are_consistent(self) -> None:
        """章节的 start_offset / end_offset 必须能在原文里定位。"""
        index = split_sections(RESUME)
        for section in index.sections:
            assert 0 <= section.start_offset <= section.end_offset <= len(RESUME)
            assert RESUME[section.start_offset : section.end_offset].strip()
            # 章节的 heading_line 必须包含标题文字（Markdown 井号等装饰已被清理）
            assert section.title in section.heading_line

    def test_line_numbers_ascending(self) -> None:
        index = split_sections(RESUME)
        starts = [s.start_line for s in index.sections]
        assert starts == sorted(starts)
        for section in index.sections:
            assert section.end_line >= section.start_line

    def test_get_and_text_of(self) -> None:
        index = split_sections(RESUME)
        skill_text = index.text_of(sec.KIND_SKILL)
        assert "Python" in skill_text and "FastAPI" in skill_text
        assert "荣誉奖项" not in skill_text

    def test_first_returns_none_when_absent(self) -> None:
        index = split_sections(RESUME)
        assert index.first(sec.KIND_DUTY) is None
        assert index.get(sec.KIND_DUTY) == []

    def test_section_to_dict(self) -> None:
        index = split_sections(RESUME)
        payload = index.to_dict()
        assert payload["count"] == len(index.sections)
        assert isinstance(payload["sections"], list)
        assert "kind" in payload["sections"][0]


class TestSkillSourceText:
    def test_summary_excluded_from_skill_scan(self) -> None:
        """自我评价必须排除在技能扫描之外。

        自我评价里常写"熟悉 Python 生态""有良好的沟通能力"这类软性描述，
        用它做技能扫描会把无关词也算成技能，污染匹配结果。
        注意：这里排除的是**技能扫描的输入**，
        解析器仍会单独扫描自我评价并标记为低可信来源（origin=summary）。
        """
        index = split_sections(RESUME)
        text = strip_sections_for_skills(index)
        assert "自我评价" not in text
        assert "对 AI 应用开发有浓厚兴趣" not in text
        # 但教育背景/实习/项目都要保留
        assert "某某大学" in text
        assert "FAISS" in text

    def test_preamble_kept(self) -> None:
        index = split_sections(RESUME)
        assert "张三" in strip_sections_for_skills(index)
