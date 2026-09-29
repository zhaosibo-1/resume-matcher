"""技能词典与归一化层测试。

这一层是匹配准确率的地基：归一化错一个别名，下游的分数就会错一片。
所以测试覆盖得比较细，包括几个**踩过的坑**，用测试把它们永久钉住。
"""

from __future__ import annotations

import pytest

from app.skills import (
    DEFAULT_NORMALIZER,
    LEVEL_WORDS,
    SKILL_DB,
    SkillEntry,
    SkillNormalizer,
    group_by_category,
    normalize_skills,
    scan_skills,
)


# ---------------------------------------------------------------------------
# 词典自身的健康度
# ---------------------------------------------------------------------------


class TestDictionaryIntegrity:
    """词典作为数据，本身也需要被校验。"""

    def test_no_duplicate_canonical(self) -> None:
        """canonical 必须唯一 —— 重复定义是维护事故。"""
        names = [entry.canonical for entry in SKILL_DB]
        assert len(names) == len(set(names))

    def test_duplicate_canonical_raises(self) -> None:
        """重复 canonical 应当直接报错，而不是静默覆盖。"""
        bad = (
            SkillEntry("Python", "编程语言", ("python",)),
            SkillEntry("Python", "其它", ("py",)),
        )
        with pytest.raises(ValueError, match="重复的 canonical"):
            SkillNormalizer(bad)

    def test_alias_collision_is_reported(self, caplog: pytest.LogCaptureFixture) -> None:
        """别名跨词条冲突必须告警。

        这条测试守的是一个真实踩过的坑：Git 和 GitHub 的词条一度都收录了
        "github" 这个别名，由于匹配规则是「长别名优先 + 区间占用」，
        先注册的 Git 抢走了 "GitHub" 这段文本，导致简历里写的 GitHub
        被识别成 Git，报告里误报"未覆盖开源经历"。
        这种 bug 不会报错、不会崩，只会在结果里静默错掉。
        """
        conflicted = (
            SkillEntry("Git", "开发工具", ("git", "github")),
            SkillEntry("GitHub", "开发工具", ("github",)),
        )
        with caplog.at_level("WARNING"):
            SkillNormalizer(conflicted)
        assert any("别名冲突" in record.message for record in caplog.records)

    def test_builtin_dictionary_has_no_alias_collision(self, caplog: pytest.LogCaptureFixture) -> None:
        """内置词典必须干净 —— 不允许存在任何别名冲突。"""
        with caplog.at_level("WARNING"):
            SkillNormalizer(SKILL_DB)
        conflicts = [r for r in caplog.records if "别名冲突" in r.message]
        assert not conflicts, f"内置词典存在别名冲突：{[r.message for r in conflicts]}"

    def test_aliases_have_no_whitespace(self) -> None:
        """别名不该有前后空格（有空格会导致永远匹配不到）。"""
        for entry in SKILL_DB:
            for alias in entry.aliases:
                assert alias == alias.strip(), f"{entry.canonical} 的别名 {alias!r} 有不必要的空白"

    def test_dictionary_scale(self) -> None:
        """词典规模的下限保护：少于 100 条说明文件被误删了。"""
        assert len(SKILL_DB) >= 100


# ---------------------------------------------------------------------------
# 归一化
# ---------------------------------------------------------------------------


class TestNormalization:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("js", "JavaScript"),
            ("JavaScript", "JavaScript"),
            ("ES6", "JavaScript"),
            ("ecmascript", "JavaScript"),
            ("k8s", "Kubernetes"),
            ("K8S集群", None),  # 别名是 "k8s集群"，大小写不敏感应命中
            ("大模型", "LLM"),
            ("LLM", "LLM"),
            ("大语言模型", "LLM"),
            ("sklearn", "scikit-learn"),
            ("py3", "Python"),
            ("golang", "Go"),
            ("torch", "PyTorch"),
            ("鸿蒙", "HarmonyOS"),
            ("arkts", "ArkTS"),
            ("向量库", "向量数据库"),
            ("微调", "微调"),
            ("fine-tune", "微调"),
            ("qlora", "LoRA"),
        ],
    )
    def test_alias_mapping(self, raw: str, expected: str) -> None:
        actual = DEFAULT_NORMALIZER.canonical_of(raw)
        if expected is None:
            # "K8S集群" 的别名是小写 "k8s集群"，归一化应该成功指向 Kubernetes
            assert actual == "Kubernetes"
        else:
            assert actual == expected, f"{raw!r} 应归一化为 {expected!r}，实际 {actual!r}"

    def test_unknown_returns_none(self) -> None:
        assert DEFAULT_NORMALIZER.canonical_of("不存在的技术栈XYZ") is None
        assert not DEFAULT_NORMALIZER.is_known("不存在的技术栈XYZ")

    def test_case_insensitive(self) -> None:
        for raw in ("PYTHON", "python", "Python", "pYtHoN"):
            assert DEFAULT_NORMALIZER.canonical_of(raw) == "Python"

    def test_whitespace_tolerated(self) -> None:
        assert DEFAULT_NORMALIZER.canonical_of("  python  ") == "Python"

    def test_normalize_list_keeps_unknown(self) -> None:
        """词典认不出的技能必须原样保留。

        真实 JD 里总有词典覆盖不到的新技术。如果静默丢掉，
        匹配分就会虚高（"要求 10 项我只缺 1 项"变成"要求 9 项我缺 0 项"）。
        """
        result = DEFAULT_NORMALIZER.normalize_list(["js", "某个新框架", "python", "js"])
        assert result == ["JavaScript", "某个新框架", "Python"]

    def test_normalize_list_dedupes(self) -> None:
        assert DEFAULT_NORMALIZER.normalize_list(["python", "Python", "PYTHON"]) == ["Python"]

    def test_normalize_list_skips_blank(self) -> None:
        assert DEFAULT_NORMALIZER.normalize_list(["", "  ", "python"]) == ["Python"]

    def test_module_level_helpers(self) -> None:
        assert normalize_skills(["js"]) == ["JavaScript"]
        assert "Python" in [m.canonical for m in scan_skills("熟悉 Python")]


# ---------------------------------------------------------------------------
# 扫描：边界、优先级、区间占用
# ---------------------------------------------------------------------------


class TestScanning:
    def test_long_alias_wins(self) -> None:
        """JavaScript 不该被拆成 Java。这是长别名优先的直接验证。"""
        hits = DEFAULT_NORMALIZER.scan("熟悉 JavaScript 开发")
        names = [h.canonical for h in hits]
        assert "JavaScript" in names
        assert "Java" not in names

    @pytest.mark.parametrize("text", ["Google 的工程师", "用 Django 开发", "algorithm 算法"])
    def test_go_does_not_match_inside_words(self, text: str) -> None:
        """单字/双字英文技能必须靠词边界拦住，否则 "go" 会在 Google/Django 里乱命中。"""
        assert "Go" not in [h.canonical for h in DEFAULT_NORMALIZER.scan(text)]

    def test_go_matches_standalone(self) -> None:
        assert "Go" in [h.canonical for h in DEFAULT_NORMALIZER.scan("熟悉 Go 语言")]

    def test_python_with_version_suffix(self) -> None:
        """Python3 / Java8 这类带数字的写法应当命中（右侧边界故意放开数字）。"""
        assert "Python" in [h.canonical for h in DEFAULT_NORMALIZER.scan("使用 Python3 开发")]
        assert "Java" in [h.canonical for h in DEFAULT_NORMALIZER.scan("Java8 环境")]

    def test_cpp_and_csharp(self) -> None:
        assert "C++" in [h.canonical for h in DEFAULT_NORMALIZER.scan("熟悉 C++ 与 C#")]
        # C++11 也应当命中 C++
        assert "C++" in [h.canonical for h in DEFAULT_NORMALIZER.scan("C++11 标准")]

    def test_bare_c_is_not_matched(self) -> None:
        """单独的 "C" 不是技能（误报率远高于收益），只有 "C语言" 才算。"""
        assert "C" not in [h.canonical for h in DEFAULT_NORMALIZER.scan("C 选项和 D 选项")]
        assert "C" in [h.canonical for h in DEFAULT_NORMALIZER.scan("熟悉 C语言 编程")]

    def test_chinese_alias(self) -> None:
        hits = DEFAULT_NORMALIZER.scan("熟悉大模型应用开发与检索增强")
        names = [h.canonical for h in hits]
        assert "LLM" in names
        assert "RAG" in names

    def test_interval_occupancy_prevents_overlap(self) -> None:
        """同一段文本不该产出两条重叠命中。"""
        hits = DEFAULT_NORMALIZER.scan("JavaScript")
        spans = [(h.start, h.end) for h in hits]
        assert len(spans) == len(set(spans))
        for i, (s1, e1) in enumerate(spans):
            for s2, e2 in spans[i + 1 :]:
                assert e1 <= s2 or e2 <= s1, "命中区间发生了重叠"

    def test_offsets_point_to_real_text(self) -> None:
        """每个命中的 offset 必须能在原文里取回同样的字符串。"""
        text = "我们用 FastAPI 搭配 Redis 做了缓存，前端是 Vue。"
        for hit in DEFAULT_NORMALIZER.scan(text):
            assert text[hit.start : hit.end] == hit.raw

    def test_hits_are_position_sorted(self) -> None:
        text = "先 Python 后 Docker 再 Kubernetes"
        hits = DEFAULT_NORMALIZER.scan(text)
        assert [h.start for h in hits] == sorted(h.start for h in hits)

    def test_scan_empty_text(self) -> None:
        assert DEFAULT_NORMALIZER.scan("") == []
        assert DEFAULT_NORMALIZER.scan("这段文本里没有任何技术名词") == []

    def test_scan_limit(self) -> None:
        text = "Python Java Go Rust C++ Docker Redis"
        assert len(DEFAULT_NORMALIZER.scan(text, limit=3)) == 3

    def test_scan_canonicals_dedupes(self) -> None:
        text = "Python 和 python 都是 Python"
        assert DEFAULT_NORMALIZER.scan_canonicals(text) == ["Python"]

    def test_html_like_text_not_broken(self) -> None:
        """尖括号/斜杠不该影响匹配。"""
        names = DEFAULT_NORMALIZER.scan_canonicals("<div>FastAPI</div> 与 Flask/Redis")
        assert "FastAPI" in names and "Flask" in names and "Redis" in names


# ---------------------------------------------------------------------------
# 掌握程度分级
# ---------------------------------------------------------------------------


class TestLevelDetection:
    @pytest.mark.parametrize(
        ("text", "skill", "expected_level"),
        [
            ("精通 Python", "Python", "精通"),
            ("Python（熟练）", "Python", "熟练"),
            ("了解 Java，精通 Python", "Python", "精通"),
            ("了解 Java，精通 Python", "Java", "了解"),
            ("熟悉 Kubernetes", "Kubernetes", "熟悉"),
            ("接触过 Rust", "Rust", "接触过"),
        ],
    )
    def test_level_detection(self, text: str, skill: str, expected_level: str) -> None:
        hits = [h for h in DEFAULT_NORMALIZER.scan(text) if h.canonical == skill]
        assert hits, f"{text!r} 里没扫到 {skill}"
        hit = hits[0]
        word, score = DEFAULT_NORMALIZER.detect_level(text, hit.start, hit.end)
        assert word == expected_level, f"期望 {expected_level}，实际 {word}"
        assert score == LEVEL_WORDS[expected_level]

    def test_nearest_level_word_wins(self) -> None:
        """「熟练 Java，精通 Python」里 Python 的程度应由更近的"精通"决定。"""
        text = "熟练 Java，精通 Python"
        hits = {h.canonical: h for h in DEFAULT_NORMALIZER.scan(text)}
        word, _ = DEFAULT_NORMALIZER.detect_level(text, hits["Python"].start, hits["Python"].end)
        assert word == "精通"

    def test_no_level_word(self) -> None:
        text = "使用 Python 开发"
        hit = DEFAULT_NORMALIZER.scan(text)[0]
        word, score = DEFAULT_NORMALIZER.detect_level(text, hit.start, hit.end)
        assert word is None and score is None

    def test_level_is_monotonic(self) -> None:
        """程度词分值应当符合直觉排序。"""
        assert LEVEL_WORDS["精通"] > LEVEL_WORDS["熟练"] > LEVEL_WORDS["了解"] > LEVEL_WORDS["听说过"]


# ---------------------------------------------------------------------------
# 上位技能（parents）
# ---------------------------------------------------------------------------


class TestParents:
    def test_faiss_maps_to_vector_db(self) -> None:
        assert "向量数据库" in DEFAULT_NORMALIZER.parents_of("FAISS")
        assert "向量数据库" in DEFAULT_NORMALIZER.parents_of("Milvus")

    def test_lora_maps_to_finetune(self) -> None:
        assert "微调" in DEFAULT_NORMALIZER.parents_of("LoRA")

    def test_pytorch_maps_to_deep_learning(self) -> None:
        assert "深度学习" in DEFAULT_NORMALIZER.parents_of("PyTorch")

    def test_unknown_has_no_parents(self) -> None:
        assert DEFAULT_NORMALIZER.parents_of("Python") == ()
        assert DEFAULT_NORMALIZER.parents_of("不存在") == ()

    def test_expand_with_parents(self) -> None:
        expanded = DEFAULT_NORMALIZER.expand_with_parents(["FAISS", "Python"])
        assert "FAISS" in expanded and "向量数据库" in expanded and "Python" in expanded

    def test_expand_avoids_infinite_loop_on_cycles(self) -> None:
        """即使词典里出现环，扩展也不能死循环（防御性测试）。

        实现上只扩展一级、不做传递闭包，所以环天然是安全的 ——
        这条测试把这个约束钉住，防止以后有人改成递归实现时踩坑。
        """
        cyclic = (
            SkillEntry("A", "测试", ("a",), parents=("B",)),
            SkillEntry("B", "测试", ("b",), parents=("A",)),
        )
        nz = SkillNormalizer(cyclic)
        result = nz.expand_with_parents(["A"])
        assert set(result) == {"A", "B"}


# ---------------------------------------------------------------------------
# 分组
# ---------------------------------------------------------------------------


class TestGrouping:
    def test_group_by_category(self) -> None:
        grouped = group_by_category(["Python", "PyTorch", "Docker"])
        assert "Python" in grouped["编程语言"]
        assert "PyTorch" in grouped["AI/机器学习"]
        assert "Docker" in grouped["云原生与运维"]

    def test_unknown_category_fallback(self) -> None:
        grouped = group_by_category(["某个未知技能"])
        assert "某个未知技能" in grouped["其它"]

    def test_aliases_of(self) -> None:
        aliases = DEFAULT_NORMALIZER.aliases_of("JavaScript")
        assert "js" in aliases and "javascript" in aliases

    def test_canonical_names_sorted(self) -> None:
        names = DEFAULT_NORMALIZER.canonical_names
        assert names == sorted(names)
        assert "Python" in names
