"""配置层：环境变量 -> Settings 对象。

设计原则：
- **零配置可跑**。不填任何环境变量，服务也能正常启动并完成全流程匹配，
  只是解析阶段会走规则化引擎（离线引擎）而不是大模型。
  这对 CI 很关键 —— 测试环境不应该依赖任何 secret。
- **Key 只从环境变量读**，且 ``describe()`` 返回的永远是脱敏值，
  避免 Key 通过 ``/api/config`` 之类的接口泄漏到前端。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# 大模型厂商预设（OpenAI 兼容协议）
# ---------------------------------------------------------------------------

#: 常见厂商的默认接入点。用户只需填 Key，Base URL 与模型名可自动带出。
LLM_PRESETS: dict[str, dict[str, str]] = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "label": "DeepSeek",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "label": "阿里通义千问",
    },
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "model": "moonshot-v1-8k",
        "label": "月之暗面 Kimi",
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4-flash",
        "label": "智谱 GLM",
    },
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "label": "硅基流动",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "label": "OpenAI",
    },
    "custom": {
        "base_url": "",
        "model": "",
        "label": "自定义（需同时填 LLM_BASE_URL 与 LLM_MODEL）",
    },
}

#: 解析引擎模式
PARSER_ENGINE_AUTO = "auto"  # 有 Key 用模型，无 Key 自动降级到规则
PARSER_ENGINE_RULE = "rule"  # 强制规则引擎（结果确定，便于对照调试）
PARSER_ENGINE_LLM = "llm"    # 强制模型引擎（无 Key 直接报错，不静默降级）
PARSER_ENGINES = (PARSER_ENGINE_AUTO, PARSER_ENGINE_RULE, PARSER_ENGINE_LLM)

VERSION = "1.0.0"


def _env_str(name: str, default: str = "") -> str:
    """读字符串环境变量（顺带 strip，避免 .env 里的行尾空格惹祸）。"""
    value = os.environ.get(name)
    return default if value is None else value.strip()


def _env_bool(name: str, default: bool) -> bool:
    """读布尔环境变量。接受 1/true/yes/on（大小写不敏感）。"""
    raw = _env_str(name)
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on", "y"}


def _env_int(name: str, default: int, *, minimum: int | None = None, maximum: int | None = None) -> int:
    """读整型环境变量；解析失败或越界时回落到默认值（并夹取到合法范围）。"""
    raw = _env_str(name)
    try:
        value = int(raw) if raw else default
    except ValueError:
        value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _env_float(name: str, default: float, *, minimum: float | None = None, maximum: float | None = None) -> float:
    """读浮点环境变量；解析失败或越界时回落到默认值。"""
    raw = _env_str(name)
    try:
        value = float(raw) if raw else default
    except ValueError:
        value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _env_list(name: str, default: list[str] | None = None) -> list[str]:
    """读逗号分隔的列表环境变量，自动去空白与空项。"""
    raw = _env_str(name)
    if not raw:
        return list(default or [])
    return [part.strip() for part in raw.split(",") if part.strip()]


def mask_secret(secret: str, *, keep_head: int = 3, keep_tail: int = 4) -> str:
    """把密钥脱敏成 ``sk-xxx****abcd`` 形式。

    空白或过短的密钥一律返回 ``"(未配置)"``，避免把短 Key 原样漏出去。
    """
    if not secret:
        return "(未配置)"
    if len(secret) < keep_head + keep_tail + 4:
        return "****"
    return f"{secret[:keep_head]}****{secret[-keep_tail:]}"


@dataclass
class Settings:
    """运行期配置。由 :func:`load_settings` 从环境变量装配。"""

    # ---- 大模型 ----
    llm_preset: str = "deepseek"
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = ""
    llm_temperature: float = 0.0
    llm_max_tokens: int = 4096
    llm_timeout: int = 60

    # ---- 解析引擎 ----
    engine_mode: str = PARSER_ENGINE_AUTO

    # ---- 服务 ----
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    max_upload_mb: int = 5
    data_dir: Path = field(default_factory=lambda: Path("data"))

    # ---- 匹配行为 ----
    #: 硬性技能权重为 0（用户手动拖到 0）时是否仍然照常打分。留成开关便于实验。
    clamp_weights: bool = True

    #: 上传的简历是否落盘。测试环境设为 False，避免在仓库里留下测试残留文件。
    store_persist: bool = True

    # ---- 运行期统计（进程内）----
    started_at: float = 0.0

    # -- 派生属性 ----------------------------------------------------------

    @property
    def preset(self) -> dict[str, str]:
        """当前 preset 的原始配置。"""
        return LLM_PRESETS.get(self.llm_preset, LLM_PRESETS["custom"])

    @property
    def resolved_base_url(self) -> str:
        """实际生效的 Base URL：显式配置优先于 preset。"""
        return self.llm_base_url or self.preset.get("base_url", "")

    @property
    def resolved_model(self) -> str:
        """实际生效的模型名：显式配置优先于 preset。"""
        return self.llm_model or self.preset.get("model", "")

    @property
    def llm_enabled(self) -> bool:
        """是否具备调用大模型的条件（Key + Base URL + 模型名都齐了才算）。"""
        return bool(self.llm_api_key and self.resolved_base_url and self.resolved_model)

    @property
    def active_engine(self) -> str:
        """当前真正生效的解析引擎。

        - ``rule``：强制规则引擎；
        - ``llm``：强制模型引擎（配置不全时会由调用方报错）；
        - ``auto``：有 Key 就上模型，没 Key 就降级到规则 —— 但**明确告知**用户降级了。
        """
        if self.engine_mode == PARSER_ENGINE_RULE:
            return PARSER_ENGINE_RULE
        if self.engine_mode == PARSER_ENGINE_LLM:
            return PARSER_ENGINE_LLM
        return PARSER_ENGINE_LLM if self.llm_enabled else PARSER_ENGINE_RULE

    @property
    def resumes_dir(self) -> Path:
        """上传简历的落盘目录。"""
        return self.data_dir / "resumes"

    # -- 输出 --------------------------------------------------------------

    def describe(self) -> dict[str, object]:
        """导出给 ``/api/config`` 的脱敏配置。

        **绝不能**把 ``llm_api_key`` 原样返回 —— 前端只需要知道
        「有没有配 Key」，不需要知道 Key 是什么。
        """
        return {
            "llm_enabled": self.llm_enabled,
            "provider": self.preset.get("label", self.llm_preset),
            "model": self.resolved_model or "(未配置)",
            "base_url": self.resolved_base_url or "(未配置)",
            "api_key_masked": mask_secret(self.llm_api_key),
            "engine": self.active_engine,
            "engine_mode": self.engine_mode,
            "max_upload_mb": self.max_upload_mb,
        }

    def ensure_dirs(self) -> None:
        """确保运行期需要的目录存在。"""
        self.resumes_dir.mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    """从环境变量装配 Settings。

    所有取值都有兜底，环境变量写错不会让服务起不来 —— 只会回落到默认值。
    这是刻意的：**配置错误不应该表现为启动崩溃**，而应该表现为
    ``/api/config`` 里一眼可见的降级提示。
    """
    import time

    preset = _env_str("LLM_PRESET", "deepseek").lower()
    if preset not in LLM_PRESETS:
        preset = "custom"

    engine = _env_str("PARSER_ENGINE", PARSER_ENGINE_AUTO).lower()
    if engine not in PARSER_ENGINES:
        engine = PARSER_ENGINE_AUTO

    return Settings(
        llm_preset=preset,
        llm_api_key=_env_str("LLM_API_KEY"),
        llm_base_url=_env_str("LLM_BASE_URL"),
        llm_model=_env_str("LLM_MODEL"),
        llm_temperature=_env_float("LLM_TEMPERATURE", 0.0, minimum=0.0, maximum=2.0),
        llm_max_tokens=_env_int("LLM_MAX_TOKENS", 4096, minimum=256, maximum=32768),
        llm_timeout=_env_int("LLM_TIMEOUT", 60, minimum=5, maximum=600),
        engine_mode=engine,
        host=_env_str("HOST", "0.0.0.0"),
        port=_env_int("PORT", 8000, minimum=1, maximum=65535),
        cors_origins=_env_list("CORS_ORIGINS", ["*"]),
        max_upload_mb=_env_int("MAX_UPLOAD_MB", 5, minimum=1, maximum=100),
        data_dir=Path(_env_str("DATA_DIR", "data")),
        clamp_weights=_env_bool("CLAMP_WEIGHTS", True),
        store_persist=_env_bool("STORE_PERSIST", True),
        started_at=time.time(),
    )


__all__ = [
    "LLM_PRESETS",
    "PARSER_ENGINE_AUTO",
    "PARSER_ENGINE_RULE",
    "PARSER_ENGINE_LLM",
    "PARSER_ENGINES",
    "VERSION",
    "Settings",
    "load_settings",
    "mask_secret",
]
