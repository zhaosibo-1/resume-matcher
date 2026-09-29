"""文本处理工具层测试。

这一层是"脏活收口处"，也是很多隐蔽 bug 的藏身处。测试重点放在
文本规范化（会改变长度，进而影响所有 offset）与日期解析（格式极其多样）。
"""

from __future__ import annotations

import pytest

from app.textutil import (
    clean_heading,
    context_phrase,
    count_numbers,
    evidence_around,
    find_date_range,
    find_single_date,
    find_years,
    has_quantified,
    indentation,
    is_bullet,
    normalize_text,
    shorten,
    split_blocks,
    split_sentences,
    strip_bullet,
)


# ---------------------------------------------------------------------------
# 规范化
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_crlf_unified(self) -> None:
        assert normalize_text("a\r\nb\rc\n") == "a\nb\nc"

    def test_zero_width_removed(self) -> None:
        """PDF 复制出来的简历常夹带零宽字符，会让技能匹配莫名失败。"""
        text = "Py\u200bthon\ufeff 开发\ufeff"
        assert normalize_text(text) == "Python 开发"

    def test_fullwidth_letters_digits_converted(self) -> None:
        """全角字母数字要转半角，否则技能一条都匹配不上。"""
        assert normalize_text("Ｐｙｔｈｏｎ３．１２") == "Python3.12"

    def test_chinese_punctuation_preserved(self) -> None:
        """**中文标点必须原样保留**。

        这是修过的一个真实问题：一开始我把整个 FF01-FF5E 区间都转成了半角，
        结果中文逗号 ``,`` 也被换掉，前端展示的原文变成
        「本科及以上学历,计算机、人工智能相关专业」，用户一眼就能看出
        「这不是我写的原文」。技能匹配是没坏，但"证据溯源"的说服力没了。
        """
        raw = "本科及以上学历，计算机、人工智能相关专业（2027 届）。"
        assert normalize_text(raw) == raw

    def test_dashes_unified(self) -> None:
        text = normalize_text("2024.03—2024.09")
        assert "—" not in text and "2024.03-2024.09" == text

    def test_curly_quotes_unified(self) -> None:
        text = normalize_text("\u201c精通\u201dPython")
        assert text == '"精通"Python'

    def test_fullwidth_space_converted(self) -> None:
        assert normalize_text("Python\u3000Java") == "Python Java"

    def test_blank_lines_compressed(self) -> None:
        assert normalize_text("a\n\n\n\n\nb") == "a\n\nb"

    def test_trailing_spaces_stripped_but_indent_kept(self) -> None:
        result = normalize_text("  缩进保留   \n标题")
        assert result.split("\n")[0] == "  缩进保留"

    def test_length_is_stable_for_offsets(self) -> None:
        """规范化后的文本是 offset 的唯一基准，长度必须可预期。"""
        raw = "ａｂｃ，ｄｅｆ。"
        assert len(normalize_text(raw)) == len(raw)

    def test_empty_input(self) -> None:
        assert normalize_text("") == ""
        assert normalize_text("   \n\n  ") == ""


# ---------------------------------------------------------------------------
# 列表项与标题
# ---------------------------------------------------------------------------


class TestBulletsAndHeadings:
    @pytest.mark.parametrize("line", ["- 工作内容", "* 项目", "• 技能", "1. 参与开发", "2、负责接口", "（1）结果", "① 第一项", "▪ 子项"])
    def test_is_bullet_true(self, line: str) -> None:
        assert is_bullet(line)

    @pytest.mark.parametrize("line", ["工作内容", "2024.03 项目", "Python 开发", "没有符号的一行"])
    def test_is_bullet_false(self, line: str) -> None:
        assert not is_bullet(line)

    def test_strip_bullet(self) -> None:
        assert strip_bullet("- 参与开发") == "参与开发"
        assert strip_bullet("1. 负责接口") == "负责接口"
        assert strip_bullet("① 第一项") == "第一项"
        assert strip_bullet("没有符号") == "没有符号"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("## 教育背景", "教育背景"),
            ("**教育背景**", "教育背景"),
            ("【教育背景】", "教育背景"),
            ("（教育背景）", "教育背景"),
            ("[教育背景]", "教育背景"),
            ("一、教育背景", "教育背景"),
            ("教育背景：", "教育背景"),
            ("第三部分 教育背景", "教育背景"),
            ("  教育背景  ", "教育背景"),
        ],
    )
    def test_clean_heading(self, raw: str, expected: str) -> None:
        assert clean_heading(raw) == expected

    def test_indentation(self) -> None:
        # 用 " " * n 显式构造，避免在源码里数空格数出错
        assert indentation(" " * 4 + "缩进4") == 4
        assert indentation(" " * 2 + "缩进2") == 2
        assert indentation("\t缩进tab") == 4
        assert indentation("\t\t两tab") == 8
        assert indentation("无缩进") == 0


# ---------------------------------------------------------------------------
# 句子与块
# ---------------------------------------------------------------------------


class TestSplitting:
    def test_chinese_sentence_split(self) -> None:
        result = split_sentences("熟悉 Python。了解 Java！会 Go？")
        assert result == ["熟悉 Python", "了解 Java", "会 Go"]

    def test_english_period_needs_space(self) -> None:
        """Node.js / 3.5 / example.com 里的点不该切断句子。"""
        result = split_sentences("使用 Node.js 与 v3.5 版本")
        assert len(result) == 1
        assert "Node.js" in result[0]

    def test_english_sentence_end(self) -> None:
        result = split_sentences("I use Python. I like it.")
        assert len(result) == 2

    def test_newline_is_boundary(self) -> None:
        assert split_sentences("第一行\n第二行") == ["第一行", "第二行"]

    def test_empty_and_blank(self) -> None:
        assert split_sentences("") == []
        assert split_sentences("。。。") == []

    def test_split_blocks_respects_line(self) -> None:
        text = "- 参与开发\n- 负责接口\n\n- 优化性能"
        assert split_blocks(text) == ["参与开发", "负责接口", "优化性能"]

    def test_split_blocks_splits_long_line(self) -> None:
        text = "熟悉 Python 与 FastAPI。" * 12
        blocks = split_blocks(text, max_len=80)
        assert len(blocks) > 1
        assert all(len(b) <= 200 for b in blocks)


# ---------------------------------------------------------------------------
# 证据
# ---------------------------------------------------------------------------


class TestEvidence:
    def test_evidence_aligned_to_sentence(self) -> None:
        text = "第一句无关。使用 FAISS 构建向量索引，召回率提升。第三句无关。"
        start = text.index("FAISS")
        snippet = evidence_around(text, start, start + 5)
        assert "FAISS" in snippet
        assert "第一句" not in snippet

    def test_context_phrase_stops_at_punctuation(self) -> None:
        text = "1. 本科及以上学历，计算机相关专业；"
        idx = text.index("本科")
        assert context_phrase(text, idx, 2) == "1. 本科及以上学历"

    def test_context_phrase_at_boundary(self) -> None:
        assert context_phrase("本科", 0, 2) == "本科"

    def test_shorten(self) -> None:
        assert shorten("短文本", 10) == "短文本"
        assert shorten("这是一个很长的文本需要截断", 6) == "这是一..."
        assert shorten("", 5) == ""


# ---------------------------------------------------------------------------
# 日期
# ---------------------------------------------------------------------------


class TestDateRange:
    @pytest.mark.parametrize(
        ("text", "sy", "sm", "ey", "em"),
        [
            ("2024.03-2024.09", 2024, 3, 2024, 9),
            ("2023年9月至2024年6月", 2023, 9, 2024, 6),
            ("2022/07 - 2024/08", 2022, 7, 2024, 8),
            ("2021.09—2022.06", 2021, 9, 2022, 6),
        ],
    )
    def test_common_formats(self, text: str, sy: int, sm: int, ey: int, em: int) -> None:
        result = find_date_range(text)
        assert result is not None
        assert (result.start_year, result.start_month) == (sy, sm)
        assert (result.end_year, result.end_month) == (ey, em)

    def test_ongoing(self) -> None:
        result = find_date_range("2023.07-至今")
        assert result is not None
        assert result.ongoing is True
        from datetime import datetime

        assert result.end_year == datetime.now().year

    def test_year_only_span(self) -> None:
        """只写年份（2023-2027）也要能解析。"""
        result = find_date_range("2023-2027")
        assert result is not None
        assert result.start_year == 2023 and result.end_year == 2027

    def test_month_label(self) -> None:
        result = find_date_range("2024.03 - 2024.09")
        assert result is not None
        assert result.label == "2024.03-2024.09"

    def test_no_date(self) -> None:
        assert find_date_range("没有时间的文本") is None
        assert find_date_range("") is None

    def test_reversed_range_ignored(self) -> None:
        """结束早于开始的区间是脏数据，应当被拒绝。"""
        result = find_date_range("2024.09-2023.01")
        assert result is None

    def test_months_counting(self) -> None:
        result = find_date_range("2024.01-2024.06")
        assert result is not None
        assert result.months == 6

    def test_find_years(self) -> None:
        assert find_years("2023.09-2027.06，另有 2019 年奖项") == [2023, 2027, 2019]

    def test_find_years_ignores_non_year_numbers(self) -> None:
        assert find_years("电话 13800000000，编号 12345") == []

    def test_find_single_date(self) -> None:
        assert find_single_date("2025年6月毕业") == (2025, 6)
        assert find_single_date("没有日期") is None


# ---------------------------------------------------------------------------
# 量化
# ---------------------------------------------------------------------------


class TestQuantified:
    @pytest.mark.parametrize(
        "text",
        [
            "召回率提升 30%",
            "P99 从 800ms 降到 120ms",
            "服务 10 万用户",
            "QPS 达到 5000",
            "内存降低 70%",
            "覆盖 6 种文档格式",
        ],
    )
    def test_has_quantified_true(self, text: str) -> None:
        assert has_quantified(text)

    @pytest.mark.parametrize(
        "text",
        ["优化了系统性能", "提升了检索效果", "负责后端开发", "参与了项目"],
    )
    def test_has_quantified_false(self, text: str) -> None:
        assert not has_quantified(text)

    def test_count_numbers(self) -> None:
        assert count_numbers("提升了 30% 和 20%") == 2
        assert count_numbers("没有数字") == 0
