"""大模型客户端测试。

这个模块只做一件窄事：**把自然语言变成 JSON**。所以测试也集中在两处：
  1. ``extract_json`` —— 模型输出千奇百怪，解析器必须足够宽容；
  2. ``classify_http_error`` —— 错误必须被正确分类，上层才能决定是否降级
     （把 401 当成网络抖动去重试，只会让用户等更久还看不到原因）。

所有用例都**不联网**。
"""

from __future__ import annotations

import httpx
import pytest

from app.config import Settings
from app.llm import (
    ERR_AUTH,
    ERR_BAD_JSON,
    ERR_CONFIG,
    ERR_NETWORK,
    ERR_RATE,
    ERR_SERVER,
    ERR_UNSUPPORTED,
    LLMClient,
    LLMError,
    LLMResponse,
    classify_http_error,
    extract_json,
)


# ===========================================================================
# JSON 提取
# ===========================================================================


class TestExtractJson:
    def test_plain_json(self) -> None:
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_json_with_surrounding_whitespace(self) -> None:
        assert extract_json('\n\n  {"a": 1}  \n') == {"a": 1}

    def test_fenced_json(self) -> None:
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_fenced_without_language_tag(self) -> None:
        assert extract_json("```\n{\"a\": 1}\n```") == {"a": 1}

    def test_json_embedded_in_prose(self) -> None:
        """模型偶尔会先寒暄一句再给 JSON —— 必须能捞出来。"""
        text = "好的，我分析完了。\n\n{\"title\": \"AI 工程师\", \"skills\": [\"RAG\"]}\n\n希望有帮助！"
        assert extract_json(text) == {"title": "AI 工程师", "skills": ["RAG"]}

    def test_only_first_balanced_object_is_taken(self) -> None:
        """两段 JSON 连在一起时，只取第一个完整的顶层对象。

        简单的 ``text[find('{'):rfind('}')+1]`` 会把两段粘成一个非法字符串，
        这是真实踩过的坑。
        """
        text = '{"a": 1}\n\n顺便再给一个：{"b": 2}'
        assert extract_json(text) == {"a": 1}

    def test_braces_inside_string_do_not_confuse_scanner(self) -> None:
        """字符串字面量里的花括号不能干扰括号计数。"""
        text = '{"note": "这里有一个 } 右括号", "n": 2}'
        assert extract_json(text) == {"note": "这里有一个 } 右括号", "n": 2}

    def test_escaped_quote_inside_string(self) -> None:
        text = '{"note": "他说\\"你好\\"", "n": 3}'
        assert extract_json(text) == {"note": '他说"你好"', "n": 3}

    def test_array_wrapped_object(self) -> None:
        """模型有时把结果包进数组，取第一个对象。"""
        assert extract_json('[{"a": 1}]') == {"a": 1}

    def test_nested_object(self) -> None:
        payload = {"analysis": {"skills": ["Python"], "score": 8}, "ok": True}
        assert extract_json('{"analysis": {"skills": ["Python"], "score": 8}, "ok": true}') == payload

    @pytest.mark.parametrize("text", ["", "   ", "\n\n"])
    def test_empty_raises_bad_json(self, text: str) -> None:
        with pytest.raises(LLMError) as exc:
            extract_json(text)
        assert exc.value.kind == ERR_BAD_JSON
        assert "空内容" in str(exc.value)

    def test_garbage_raises_bad_json_with_preview(self) -> None:
        with pytest.raises(LLMError) as exc:
            extract_json("我无法完成这个任务，因为……")
        assert exc.value.kind == ERR_BAD_JSON
        # 报错信息要带上开头片段，方便排查模型到底吐了什么
        assert "我无法完成" in str(exc.value)

    def test_unbalanced_json_raises(self) -> None:
        with pytest.raises(LLMError) as exc:
            extract_json('{"a": 1')
        assert exc.value.kind == ERR_BAD_JSON

    def test_bare_scalar_is_not_an_object(self) -> None:
        """模型回了个裸字符串 / 数字，对调用方来说等于没有结果。"""
        with pytest.raises(LLMError):
            extract_json('"just a string"')
        with pytest.raises(LLMError):
            extract_json("42")


# ===========================================================================
# HTTP 错误分类
# ===========================================================================


class TestClassifyHttpError:
    @pytest.mark.parametrize("status", [401, 403])
    def test_auth(self, status: int) -> None:
        err = classify_http_error(status, '{"error":"invalid api key"}')
        assert err.kind == ERR_AUTH
        assert err.status == status
        # 鉴权失败重试没有意义，必须标记为不可重试
        assert err.retriable is False

    def test_insufficient_balance(self) -> None:
        assert classify_http_error(402, "Insufficient Balance").kind == ERR_RATE

    def test_not_found_is_config_problem(self) -> None:
        """404 在本项目里几乎总是"Base URL 或模型名写错了"。"""
        err = classify_http_error(404, "model not found")
        assert err.kind == ERR_CONFIG
        assert "LLM_BASE_URL" in str(err)

    def test_rate_limit_is_retriable(self) -> None:
        err = classify_http_error(429, "too many requests")
        assert err.kind == ERR_RATE
        assert err.retriable is True

    def test_unsupported_json_mode_detected_via_body(self) -> None:
        """400 里有一类很常见：模型不支持 response_format。

        识别出来才能自动改用"提示词约束 + 本地解析"，而不是直接失败。
        """
        err = classify_http_error(400, '{"error":{"message":"response_format is not supported"}}')
        assert err.kind == ERR_UNSUPPORTED

    def test_unsupported_json_mode_without_marker_is_config_error(self) -> None:
        err = classify_http_error(400, '{"error":{"message":"invalid parameter: temperature"}}')
        assert err.kind == ERR_CONFIG

    @pytest.mark.parametrize("status", [500, 502, 503])
    def test_server_errors_are_retriable(self, status: int) -> None:
        err = classify_http_error(status, "internal error")
        assert err.kind == ERR_SERVER
        assert err.retriable is True

    def test_body_is_truncated_in_message(self) -> None:
        """错误信息里带响应体方便排查，但不能把几 MB 的 HTML 错误页塞进去。"""
        err = classify_http_error(500, "x" * 5000)
        assert len(str(err)) < 1000

    def test_error_serializes_to_dict(self) -> None:
        payload = classify_http_error(429, "slow down").to_dict()
        assert payload["kind"] == ERR_RATE
        assert payload["status"] == 429
        assert payload["retriable"] is True


# ===========================================================================
# 响应对象
# ===========================================================================


class TestLLMResponse:
    def test_total_tokens(self) -> None:
        assert LLMResponse(text="x", usage={"total_tokens": 42}).total_tokens == 42

    def test_total_tokens_defaults_to_zero(self) -> None:
        assert LLMResponse(text="x").total_tokens == 0


# ===========================================================================
# 客户端（不联网）
# ===========================================================================


class TestLLMClientOffline:
    @pytest.fixture()
    def disabled(self) -> LLMClient:
        return LLMClient(Settings(llm_api_key=""))

    @pytest.fixture()
    def stub(self) -> LLMClient:
        """配了 Key 但**不真的发请求** —— 客户端只在被调用时才建连接池。"""
        return LLMClient(
            Settings(llm_preset="custom", llm_api_key="sk-test", llm_base_url="http://127.0.0.1:1/v1", llm_model="m")
        )

    async def test_chat_without_key_raises_config_error(self, disabled: LLMClient) -> None:
        with pytest.raises(LLMError) as exc:
            await disabled.chat([{"role": "user", "content": "hi"}])
        assert exc.value.kind == ERR_CONFIG

    async def test_chat_json_without_key_raises_config_error(self, disabled: LLMClient) -> None:
        with pytest.raises(LLMError) as exc:
            await disabled.chat_json("sys", "user")
        assert exc.value.kind == ERR_CONFIG
        # 配置错误不该被记成"调用失败"（它根本没发出去）
        assert disabled.calls == 0

    async def test_ping_without_key_returns_structured_result(self, disabled: LLMClient) -> None:
        """ping 不能抛异常 —— 前端要拿 message 直接展示给用户。"""
        result = await disabled.ping()
        assert result["ok"] is False
        assert result["kind"] == ERR_CONFIG
        assert "规则引擎" in result["message"]
        assert result["latency_ms"] == 0

    async def test_network_failure_is_classified(self) -> None:
        """指向一个必然连不上的地址，验证异常被翻译成 ERR_NETWORK 而不是裸抛。"""
        client = LLMClient(
            Settings(
                llm_preset="custom",
                llm_api_key="sk-test",
                llm_base_url="http://127.0.0.1:1/v1",
                llm_model="m",
                llm_timeout=5,
            )
        )
        try:
            with pytest.raises(LLMError) as exc:
                await client.chat([{"role": "user", "content": "hi"}])
            assert exc.value.kind == ERR_NETWORK
            assert exc.value.retriable is True
            assert client.calls == 1
            assert client.failures == 1
        finally:
            await client.aclose()

    async def test_aclose_is_idempotent(self, stub: LLMClient) -> None:
        """关闭两次不能报错（lifespan 与异常路径可能都调到）。"""
        await stub.aclose()
        await stub.aclose()

    def test_describe_masks_key(self, stub: LLMClient) -> None:
        described = stub.describe()
        assert described["enabled"] is True
        assert described["api_key"] != "sk-test"
        assert described["calls"] == 0
        assert described["json_mode_unsupported"] is False

    def test_client_is_lazily_created(self, stub: LLMClient) -> None:
        """连接池必须懒加载 —— 否则每次启动服务都会白白建一次。"""
        assert stub._client is None
        assert isinstance(stub.client, httpx.AsyncClient)
        assert stub._client is not None

    async def test_json_mode_fallback_retries_without_response_format(self) -> None:
        """服务端拒绝 response_format 时，应自动去掉该参数重试一次。

        用 httpx 的 MockTransport 拦截请求，验证第二次请求的 payload 里
        确实没有 response_format，并且标志位被记下来了。
        """
        seen_payloads: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            import json as _json

            body = _json.loads(request.content.decode())
            seen_payloads.append(body)
            if len(seen_payloads) == 1:
                return httpx.Response(400, json={"error": {"message": "response_format is not supported"}})
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}], "model": "m"},
            )

        client = LLMClient(
            Settings(llm_preset="custom", llm_api_key="sk-test", llm_base_url="http://mock/v1", llm_model="m")
        )
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            result = await client.chat_json("sys", "user")
        finally:
            await client.aclose()

        assert result == {"ok": True}
        assert len(seen_payloads) == 2
        assert "response_format" in seen_payloads[0]
        assert "response_format" not in seen_payloads[1]
        assert client._json_mode_unsupported is True

    async def test_response_structure_violation_is_server_error(self) -> None:
        """网关返回了结构不对的 JSON（不是 OpenAI 协议）要报清楚。"""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"unexpected": "shape"})

        client = LLMClient(
            Settings(llm_preset="custom", llm_api_key="sk-test", llm_base_url="http://mock/v1", llm_model="m")
        )
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            with pytest.raises(LLMError) as exc:
                await client.chat([{"role": "user", "content": "hi"}])
            assert exc.value.kind == ERR_SERVER
        finally:
            await client.aclose()

    async def test_html_error_page_is_not_json_but_reported_as_server_error(self) -> None:
        """真实场景：反向代理挂了，返回一坨 HTML。"""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html><body>502 Bad Gateway</body></html>")

        client = LLMClient(
            Settings(llm_preset="custom", llm_api_key="sk-test", llm_base_url="http://mock/v1", llm_model="m")
        )
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            with pytest.raises(LLMError) as exc:
                await client.chat([{"role": "user", "content": "hi"}])
            assert exc.value.kind == ERR_SERVER
        finally:
            await client.aclose()

    async def test_successful_usage_is_parsed(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "model": "real-model",
                    "choices": [{"message": {"content": "就绪"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                },
            )

        client = LLMClient(
            Settings(llm_preset="custom", llm_api_key="sk-test", llm_base_url="http://mock/v1", llm_model="m")
        )
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            resp = await client.chat([{"role": "user", "content": "hi"}])
            assert resp.text == "就绪"
            assert resp.model == "real-model"
            assert resp.total_tokens == 12
            assert resp.finish_reason == "stop"

            ping = await client.ping()
            assert ping["ok"] is True
            assert ping["model"] == "real-model"
        finally:
            await client.aclose()
