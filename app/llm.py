"""大模型客户端（OpenAI 兼容协议）。

职责边界很窄：**只做一件事 —— 把自然语言变成 JSON**。

本项目不在这里做对话、做流式 token、做多轮 Agent（那些是另外两个项目的活儿）。
这里的目标是：给定一段 JD 或简历文本，让模型吐出我们约定好的结构化 JSON，
然后交给规则层去归一化和打分。

为什么这样切分很有价值？
因为「模型负责理解、规则负责判定」这条分界线，直接决定了系统的性质：
- 模型可以换来换去（甚至不接模型），打分逻辑完全不变；
- 同一份简历的结果**稳定可复现**（温度固定 0，归一化是确定性的）；
- 每一个分数都能反查到原文依据，而不是"模型说它值 80 分"。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from .config import Settings, mask_secret

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 错误类型
# ---------------------------------------------------------------------------

#: 错误分类。前端据此给出不同的提示文案（"检查 Key" vs "稍后重试"）。
ERR_CONFIG = "config"       # 配置不全（没填 Key / Base URL）
ERR_AUTH = "auth"           # Key 无效或无权限
ERR_RATE = "rate_limit"     # 限流 / 余额不足
ERR_NETWORK = "network"     # 连不上 / 超时
ERR_SERVER = "server"       # 服务端 5xx
ERR_BAD_JSON = "bad_json"   # 返回内容不是合法 JSON（模型跑偏了）
ERR_UNSUPPORTED = "unsupported"  # 该模型不支持 json_object 模式
ERR_UNKNOWN = "unknown"


class LLMError(RuntimeError):
    """调用大模型过程中的错误，带可分类的 kind，方便上层决定是否降级。"""

    def __init__(
        self,
        message: str,
        *,
        kind: str = ERR_UNKNOWN,
        status: Optional[int] = None,
        retriable: bool = False,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.status = status
        self.retriable = retriable

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "message": str(self), "status": self.status, "retriable": self.retriable}


@dataclass
class LLMResponse:
    """一次模型调用的结果。"""

    text: str
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str = ""

    @property
    def total_tokens(self) -> int:
        return int(self.usage.get("total_tokens", 0) or 0)


# ---------------------------------------------------------------------------
# JSON 提取
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def _balanced_json_slice(text: str) -> str | None:
    """从一段文本里找出**第一个完整的顶层 JSON 对象**。

    为什么要自己写括号扫描，而不是简单 ``text[text.find('{'):text.rfind('}')+1]``？
    因为后者在模型输出「我先说一句，然后给 JSON，然后又补一句」时会切出一堆
    垃圾（把两段内容连在一起）。括号扫描能精确地在第一个平衡点收尾。

    扫描时会跳过字符串字面量内部的括号与转义字符，避免被
    ``{"note": "这是 } 一个右括号"}`` 这种内容骗到。
    """
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_json(text: str) -> dict[str, Any]:
    """尽最大努力从模型回复里解析出 JSON 对象。

    依次尝试三种策略，覆盖实际会遇到的全部情况：

    1. 整段就是 JSON（最理想）；
    2. 被 ``` 代码块包着（很常见）；
    3. 混在解释性文字里（做了长度限制并提示"直接输出 JSON"之后仍偶有发生）。

    Raises:
        LLMError: 三种策略都失败时抛出 ``kind=bad_json``。
    """
    if not text or not text.strip():
        raise LLMError("模型返回了空内容", kind=ERR_BAD_JSON)

    candidate = text.strip()

    # 策略 1：直接解析
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            # 模型有时会把结果包成数组，取第一个对象
            for item in parsed:
                if isinstance(item, dict):
                    return item
    except json.JSONDecodeError:
        pass

    # 策略 2：剥掉 markdown 代码围栏
    for match in _FENCE_RE.finditer(text):
        inner = match.group(1).strip()
        try:
            parsed = json.loads(inner)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            sliced = _balanced_json_slice(inner)
            if sliced:
                try:
                    parsed = json.loads(sliced)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    continue

    # 策略 3：括号平衡切片
    sliced = _balanced_json_slice(text)
    if sliced:
        try:
            parsed = json.loads(sliced)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    preview = text[:180].replace("\n", " ")
    raise LLMError(f"模型返回的内容不是合法 JSON（开头：{preview}）", kind=ERR_BAD_JSON)


# ---------------------------------------------------------------------------
# 客户端
# ---------------------------------------------------------------------------


def classify_http_error(status: int, body: str) -> LLMError:
    """把 HTTP 状态码翻译成带分类的 LLMError。

    单独抽成函数是为了能脱离网络单测：给定状态码 + 响应体，
    断言它被翻译成哪种 kind。
    """
    snippet = (body or "").strip()[:300]

    if status in (401, 403):
        return LLMError(f"鉴权失败（HTTP {status}）：请检查 LLM_API_KEY。{snippet}", kind=ERR_AUTH, status=status)
    if status == 402:
        return LLMError(f"账户余额不足或未开通（HTTP 402）。{snippet}", kind=ERR_RATE, status=status)
    if status == 404:
        return LLMError(
            f"接口或模型不存在（HTTP 404）：请检查 LLM_BASE_URL 与 LLM_MODEL。{snippet}",
            kind=ERR_CONFIG,
            status=status,
        )
    if status == 429:
        return LLMError(f"请求过于频繁或被限流（HTTP 429）。{snippet}", kind=ERR_RATE, status=status, retriable=True)
    if 400 <= status < 500:
        # 400 里很大一类是「不支持 response_format」，单独识别以便自动重试
        lowered = snippet.lower()
        if "response_format" in lowered or "json_object" in lowered or "json mode" in lowered:
            return LLMError(
                f"该模型不支持 json_object 模式（HTTP {status}）。{snippet}",
                kind=ERR_UNSUPPORTED,
                status=status,
            )
        return LLMError(f"请求被拒绝（HTTP {status}）。{snippet}", kind=ERR_CONFIG, status=status)
    if status >= 500:
        return LLMError(f"服务端错误（HTTP {status}）。{snippet}", kind=ERR_SERVER, status=status, retriable=True)
    return LLMError(f"未知错误（HTTP {status}）。{snippet}", kind=ERR_UNKNOWN, status=status)


class LLMClient:
    """极简 OpenAI 兼容客户端。

    只暴露三个方法：``chat`` / ``chat_json`` / ``ping``。
    刻意不做重试风暴 —— 失败就快速失败并把错误分类交给上层决定降级策略，
    因为本项目的降级路径（规则引擎）本来就是完整可用的。
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.calls = 0
        self.failures = 0
        self._client: httpx.AsyncClient | None = None
        #: 记录服务端是否拒绝过 json_object 模式，避免每次都白试一轮
        self._json_mode_unsupported = False

    # -- 生命周期 ----------------------------------------------------------

    @property
    def client(self) -> httpx.AsyncClient:
        """懒加载 httpx 客户端（复用连接池，省掉每次 TLS 握手）。"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.settings.llm_timeout, connect=10.0),
                # 本项目的请求都是直连厂商 API，环境里若有代理反而会添乱
                trust_env=False,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # -- 底层调用 ----------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """发起一次 chat completion。

        Args:
            messages: OpenAI 格式的消息列表。
            temperature: 覆盖默认温度。抽取任务建议用 0 保证确定性。
            max_tokens: 覆盖默认上限。
            json_mode: 是否请求 ``response_format={"type": "json_object"}``。
                注意这个参数**不是所有厂商都支持** —— 遇到明确不支持时，
                我们会把标志位记下来，后续请求自动改用"提示词约束 + 本地解析"。

        Raises:
            LLMError: 配置不全、网络异常、鉴权失败、返回不合法等。
        """
        if not self.settings.llm_enabled:
            raise LLMError(
                "未配置大模型：请设置 LLM_API_KEY（可选 LLM_PRESET / LLM_BASE_URL / LLM_MODEL）",
                kind=ERR_CONFIG,
            )

        url = self.settings.resolved_base_url.rstrip("/") + "/chat/completions"
        payload: dict[str, Any] = {
            "model": self.settings.resolved_model,
            "messages": messages,
            "temperature": self.settings.llm_temperature if temperature is None else temperature,
            "max_tokens": self.settings.llm_max_tokens if max_tokens is None else max_tokens,
        }
        if json_mode and not self._json_mode_unsupported:
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self.settings.llm_api_key}",
            "Content-Type": "application/json",
        }

        self.calls += 1
        try:
            resp = await self.client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            self.failures += 1
            raise LLMError(f"请求超时（{self.settings.llm_timeout}s）：{exc}", kind=ERR_NETWORK, retriable=True) from exc
        except httpx.HTTPError as exc:
            self.failures += 1
            raise LLMError(f"网络请求失败：{exc}", kind=ERR_NETWORK, retriable=True) from exc

        if resp.status_code >= 400:
            self.failures += 1
            error = classify_http_error(resp.status_code, resp.text)
            if error.kind == ERR_UNSUPPORTED:
                # 记下来，下次不再带 response_format；同时本次用非 json_mode 重试一次
                self._json_mode_unsupported = True
                logger.warning("模型不支持 json_object，改为提示词约束模式重试：%s", error)
                return await self.chat(messages, temperature=temperature, max_tokens=max_tokens, json_mode=False)
            raise error

        try:
            data = resp.json()
        except ValueError as exc:
            self.failures += 1
            raise LLMError("响应不是合法 JSON（可能是网关返回了 HTML 错误页）", kind=ERR_SERVER) from exc

        try:
            choice = data["choices"][0]
            content = choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            self.failures += 1
            raise LLMError(f"响应结构不符合 OpenAI 协议：{str(data)[:200]}", kind=ERR_SERVER) from exc

        usage_raw = data.get("usage") or {}
        usage = {
            "prompt_tokens": int(usage_raw.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage_raw.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage_raw.get("total_tokens", 0) or 0),
        }

        return LLMResponse(
            text=content,
            model=str(data.get("model", self.settings.resolved_model)),
            usage=usage,
            finish_reason=str(choice.get("finish_reason", "")),
        )

    async def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """让模型输出 JSON 并解析成 dict。

        这是本项目最主要的使用方式：给定抽取指令，拿回结构化结果。
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        resp = await self.chat(messages, temperature=temperature, max_tokens=max_tokens, json_mode=True)
        try:
            return extract_json(resp.text)
        except LLMError as exc:
            self.failures += 1
            raise exc

    async def ping(self) -> dict[str, Any]:
        """连通性探测。返回结构统一，不抛异常（前端要展示失败原因）。"""
        if not self.settings.llm_enabled:
            return {
                "ok": False,
                "kind": ERR_CONFIG,
                "message": "未配置 LLM_API_KEY，服务将以规则引擎运行（功能完整，只是抽取用规则实现）",
                "model": "",
                "latency_ms": 0,
            }

        import time

        started = time.perf_counter()
        try:
            resp = await self.chat(
                [{"role": "user", "content": "回复两个字：就绪"}],
                temperature=0.0,
                max_tokens=16,
            )
        except LLMError as exc:
            return {
                "ok": False,
                "kind": exc.kind,
                "message": str(exc),
                "model": self.settings.resolved_model,
                "latency_ms": int((time.perf_counter() - started) * 1000),
            }

        return {
            "ok": True,
            "kind": "ok",
            "message": f"连通正常，返回：{resp.text.strip()[:40]}",
            "model": resp.model or self.settings.resolved_model,
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "usage": resp.usage,
        }

    # -- 可观测性 ----------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """给前端/日志用的脱敏状态。"""
        return {
            "enabled": self.settings.llm_enabled,
            "provider": self.settings.preset.get("label", self.settings.llm_preset),
            "model": self.settings.resolved_model,
            "base_url": self.settings.resolved_base_url,
            "api_key": mask_secret(self.settings.llm_api_key),
            "calls": self.calls,
            "failures": self.failures,
            "json_mode_unsupported": self._json_mode_unsupported,
        }


__all__ = [
    "LLMClient",
    "LLMResponse",
    "LLMError",
    "extract_json",
    "classify_http_error",
    "ERR_CONFIG",
    "ERR_AUTH",
    "ERR_RATE",
    "ERR_NETWORK",
    "ERR_SERVER",
    "ERR_BAD_JSON",
    "ERR_UNSUPPORTED",
    "ERR_UNKNOWN",
]
