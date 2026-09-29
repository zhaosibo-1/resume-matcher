"""测试辅助工具。

包含两块：

1. **文本样本**：短小的、每个测试都自解释的 JD / 简历片段。
   不用内置的长示例，是为了让断言更聚焦 —— 一个测试只验证一个行为。
2. **ScriptedLLM**：脚本化的大模型替身。
   让"需要模型的代码路径"也能被单元测试覆盖，而不必真的联网。
   这个类在测试里被反复使用，是整个测试套件能在 CI 上零依赖运行的关键。
"""

from __future__ import annotations

from typing import Any, Optional

from app.config import Settings
from app.llm import LLMError, LLMResponse

# ---------------------------------------------------------------------------
# 文本样本
# ---------------------------------------------------------------------------

SIMPLE_JD = """\
岗位：Python 后端开发工程师
公司：某某科技

岗位职责：
1. 负责后端服务的开发与维护；
2. 参与系统性能优化。

任职要求：
1. 本科及以上学历，计算机相关专业；
2. 精通 Python，熟悉 FastAPI 框架；
3. 熟悉 MySQL 与 Redis；
4. 有 3 年以上后端开发经验。

加分项：
1. 熟悉 Kubernetes 者优先；
2. 了解 Rust 者加分。
"""

SIMPLE_RESUME = """\
王五
电话：137-0000-0000

教育背景
某某大学 | 软件工程 | 本科 | 2019.09-2023.06

工作经历
某某公司 | 后端开发工程师 | 2023.07-至今
- 负责核心接口开发，使用 FastAPI + MySQL
- 优化慢查询，接口响应时间下降 60%

专业技能
- 编程语言：Python（精通）、Go（了解）
- 框架：FastAPI、Django
- 数据库：MySQL、Redis

自我评价
熟悉 Kubernetes 的基本使用。
"""

MINIMAL_JD = "招聘 Python 开发，要求熟悉 FastAPI。"

MINIMAL_RESUME = """\
赵六
教育背景
某某大学 计算机 本科

专业技能
- Python、FastAPI
"""


# ---------------------------------------------------------------------------
# LLM 替身
# ---------------------------------------------------------------------------


class ScriptedLLM:
    """按脚本返回结果的假 LLM 客户端。

    用法::

        client = ScriptedLLM(Settings(), responses=['{"title": "x"}'])
        data = await client.chat_json("sys", "user")   # -> {"title": "x"}

    如果 ``responses`` 用尽，或者 ``error`` 被设置，则抛出对应的 ``LLMError``。
    这样可以精确地测试"模型返回脏数据 / 调用失败"时的降级行为。
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        responses: Optional[list[str]] = None,
        error: Optional[LLMError] = None,
    ) -> None:
        self.settings = settings or Settings(llm_api_key="sk-test-key", llm_base_url="http://x", llm_model="m")
        self.responses = list(responses or [])
        self.error = error
        self.calls = 0
        self.failures = 0
        self.requests: list[list[dict[str, str]]] = []
        self._json_mode_unsupported = False

    async def chat_json(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> dict[str, Any]:
        self.requests.append(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        )
        resp = await self.chat([{"role": "user", "content": user_prompt}])
        from app.llm import extract_json

        try:
            return extract_json(resp.text)
        except LLMError:
            self.failures += 1
            raise

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        self.calls += 1
        if self.error is not None:
            self.failures += 1
            raise self.error
        if not self.responses:
            self.failures += 1
            raise LLMError("脚本已用完（测试内部错误）", kind="bad_json")
        return LLMResponse(text=self.responses.pop(0), model="scripted-model")

    async def ping(self) -> dict[str, Any]:
        if self.error is not None:
            return {"ok": False, "kind": self.error.kind, "message": str(self.error), "model": "m", "latency_ms": 1}
        return {"ok": True, "kind": "ok", "message": "脚本化就绪", "model": "scripted-model", "latency_ms": 1}

    async def aclose(self) -> None:
        return None


# ---------------------------------------------------------------------------
# 断言辅助
# ---------------------------------------------------------------------------


def dim_by_key(report: Any, key: str) -> Any:
    """从报告里按 key 取维度（取不到则断言失败，让错误信息更直观）。"""
    for dim in report.dimensions:
        if dim.key == key:
            return dim
    raise AssertionError(f"报告里没有维度 {key!r}，实际有：{[d.key for d in report.dimensions]}")
