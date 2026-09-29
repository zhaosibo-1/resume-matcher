"""配置层测试。

配置层的两条硬要求：
  1. **零配置可跑** —— 不填任何环境变量，服务必须能正常启动并完成全流程；
  2. **Key 绝不外泄** —— ``describe()`` 返回的永远只有脱敏值。

这两条都太容易被"顺手改坏"，所以用测试钉住。
"""

from __future__ import annotations

import pytest

from app.config import (
    LLM_PRESETS,
    PARSER_ENGINE_AUTO,
    PARSER_ENGINE_LLM,
    PARSER_ENGINE_RULE,
    Settings,
    load_settings,
    mask_secret,
)


# ===========================================================================
# 密钥脱敏
# ===========================================================================


class TestMaskSecret:
    def test_empty(self) -> None:
        assert mask_secret("") == "(未配置)"

    def test_short_secret_never_leaks_original(self) -> None:
        """短 Key 一律打满星 —— 否则"脱敏"反而暴露了全部内容。"""
        assert mask_secret("sk-1") == "****"
        assert mask_secret("abcdefghij") == "****"

    def test_normal_secret_keeps_head_and_tail(self) -> None:
        masked = mask_secret("sk-1234567890abcdef")
        assert masked.startswith("sk-")
        assert masked.endswith("cdef")
        assert "****" in masked
        # 中间段不能被还原出来
        assert "567890ab" not in masked

    def test_masked_value_never_equals_original(self) -> None:
        secret = "sk-realkey-with-enough-length"
        assert mask_secret(secret) != secret


# ===========================================================================
# 派生属性
# ===========================================================================


class TestDerivedSettings:
    def test_preset_base_url_and_model(self) -> None:
        s = Settings(llm_preset="deepseek", llm_api_key="k")
        assert s.resolved_base_url == LLM_PRESETS["deepseek"]["base_url"]
        assert s.resolved_model == LLM_PRESETS["deepseek"]["model"]

    def test_explicit_config_overrides_preset(self) -> None:
        """显式填了 Base URL / 模型名时，必须压过 preset。"""
        s = Settings(
            llm_preset="deepseek",
            llm_api_key="k",
            llm_base_url="http://localhost:11434/v1",
            llm_model="my-local-model",
        )
        assert s.resolved_base_url == "http://localhost:11434/v1"
        assert s.resolved_model == "my-local-model"

    def test_unknown_preset_falls_back_to_custom(self) -> None:
        s = Settings(llm_preset="not-a-provider", llm_api_key="k")
        assert s.preset == LLM_PRESETS["custom"]
        # custom 的 base_url 是空串，因此即使有 Key 也不算"可用"
        assert s.llm_enabled is False

    @pytest.mark.parametrize(
        ("key", "base_url", "model", "expected"),
        [
            ("", "", "", False),
            ("sk-x", "", "", False),
            ("", "http://x/v1", "m", False),
            ("sk-x", "", "m", False),
            ("sk-x", "http://x/v1", "", False),
            ("sk-x", "http://x/v1", "m", True),
        ],
    )
    def test_llm_enabled_requires_all_three(
        self, key: str, base_url: str, model: str, expected: bool
    ) -> None:
        """Key / Base URL / 模型名三者缺一，都不算"具备调用条件"。"""
        s = Settings(llm_preset="custom", llm_api_key=key, llm_base_url=base_url, llm_model=model)
        assert s.llm_enabled is expected

    def test_active_engine_rule_mode_forces_rule(self) -> None:
        """即使配了 Key，显式指定 rule 也必须走规则引擎（用于对照调试）。"""
        s = Settings(engine_mode=PARSER_ENGINE_RULE, llm_api_key="k", llm_base_url="http://x", llm_model="m")
        assert s.llm_enabled is True
        assert s.active_engine == PARSER_ENGINE_RULE

    def test_active_engine_llm_mode_is_not_silently_downgraded(self) -> None:
        """指定 llm 但没 Key 时，active_engine 仍报 llm —— 由调用方明确报错，
        而不是悄悄降级成 rule（静默降级会让用户以为用的是大模型）。"""
        s = Settings(engine_mode=PARSER_ENGINE_LLM, llm_api_key="")
        assert s.active_engine == PARSER_ENGINE_LLM

    def test_active_engine_auto_follows_key(self) -> None:
        without = Settings(engine_mode=PARSER_ENGINE_AUTO, llm_api_key="")
        with_key = Settings(
            engine_mode=PARSER_ENGINE_AUTO, llm_api_key="k", llm_base_url="http://x", llm_model="m"
        )
        assert without.active_engine == PARSER_ENGINE_RULE
        assert with_key.active_engine == PARSER_ENGINE_LLM

    def test_describe_never_contains_raw_key(self) -> None:
        secret = "sk-this-must-never-leak-0123456789"
        s = Settings(llm_api_key=secret, llm_base_url="http://x", llm_model="m")
        described = s.describe()
        dumped = repr(described)
        assert secret not in dumped
        assert described["api_key_masked"] != secret
        assert described["llm_enabled"] is True

    def test_describe_shows_preset_model_even_without_key(self) -> None:
        """没配 Key 时，模型名仍然要显示出来。

        "未配置" 指的是 **Key 没填**，不是"什么都不知道" ——
        用户需要看到"我连的是哪个模型的地址"，才知道该去哪儿补 Key。
        """
        described = Settings().describe()
        assert described["model"] == LLM_PRESETS["deepseek"]["model"]
        assert described["base_url"] == LLM_PRESETS["deepseek"]["base_url"]
        assert described["api_key_masked"] == "(未配置)"
        assert described["llm_enabled"] is False

    def test_describe_placeholders_when_preset_has_no_defaults(self) -> None:
        """custom 预设没有内置地址，这时才该显示占位符。"""
        described = Settings(llm_preset="custom").describe()
        assert described["model"] == "(未配置)"
        assert described["base_url"] == "(未配置)"
        assert described["api_key_masked"] == "(未配置)"

    def test_resumes_dir_under_data_dir(self) -> None:
        from pathlib import Path

        s = Settings(data_dir=Path("some/dir"))
        assert s.resumes_dir == Path("some/dir") / "resumes"

    def test_ensure_dirs_creates_directory(self, tmp_path) -> None:
        target = tmp_path / "nested" / "data"
        s = Settings(data_dir=target)
        s.ensure_dirs()
        assert s.resumes_dir.is_dir()


# ===========================================================================
# 环境变量解析
# ===========================================================================


class TestLoadSettings:
    def test_zero_config_is_runnable(self, monkeypatch) -> None:
        """**最重要的用例**：清空所有环境变量后，配置依然完整可用。"""
        for name in (
            "LLM_PRESET",
            "LLM_API_KEY",
            "LLM_BASE_URL",
            "LLM_MODEL",
            "PARSER_ENGINE",
            "PORT",
            "MAX_UPLOAD_MB",
            "DATA_DIR",
            "STORE_PERSIST",
        ):
            monkeypatch.delenv(name, raising=False)

        s = load_settings()
        assert s.llm_enabled is False
        assert s.active_engine == PARSER_ENGINE_RULE
        assert s.port == 8000
        assert s.store_persist is True
        assert s.started_at > 0

    def test_invalid_preset_and_engine_fall_back(self, monkeypatch) -> None:
        monkeypatch.setenv("LLM_PRESET", "某不存在的厂商")
        monkeypatch.setenv("PARSER_ENGINE", "某不存在的模式")
        s = load_settings()
        assert s.llm_preset == "custom"
        assert s.engine_mode == PARSER_ENGINE_AUTO

    def test_preset_is_case_insensitive_and_stripped(self, monkeypatch) -> None:
        monkeypatch.setenv("LLM_PRESET", "  DeepSeek  ")
        assert load_settings().llm_preset == "deepseek"

    def test_value_is_stripped(self, monkeypatch) -> None:
        """`.env` 里很容易带行尾空格，必须 strip 掉。"""
        monkeypatch.setenv("LLM_API_KEY", "  sk-abc  ")
        assert load_settings().llm_api_key == "sk-abc"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("1", True),
            ("true", True),
            ("TRUE", True),
            ("yes", True),
            ("on", True),
            ("0", False),
            ("false", False),
            ("随便写的", False),
            ("", True),  # 空串 -> 用默认值（默认 True）
        ],
    )
    def test_bool_parsing(self, monkeypatch, raw: str, expected: bool) -> None:
        monkeypatch.setenv("STORE_PERSIST", raw)
        assert load_settings().store_persist is expected

    def test_int_out_of_range_is_clamped(self, monkeypatch) -> None:
        """配置写错不该让服务起不来，而应该夹取到合法范围。"""
        monkeypatch.setenv("PORT", "99999")
        monkeypatch.setenv("MAX_UPLOAD_MB", "0")
        s = load_settings()
        assert s.port == 65535
        assert s.max_upload_mb == 1

    def test_int_garbage_falls_back_to_default(self, monkeypatch) -> None:
        monkeypatch.setenv("PORT", "不是数字")
        assert load_settings().port == 8000

    def test_float_temperature_clamped(self, monkeypatch) -> None:
        monkeypatch.setenv("LLM_TEMPERATURE", "9.9")
        assert load_settings().llm_temperature == 2.0
        monkeypatch.setenv("LLM_TEMPERATURE", "-1")
        assert load_settings().llm_temperature == 0.0

    def test_cors_origins_list(self, monkeypatch) -> None:
        monkeypatch.setenv("CORS_ORIGINS", "https://a.com, https://b.com ,,")
        assert load_settings().cors_origins == ["https://a.com", "https://b.com"]

    def test_cors_origins_default_wildcard(self, monkeypatch) -> None:
        monkeypatch.delenv("CORS_ORIGINS", raising=False)
        assert load_settings().cors_origins == ["*"]
