"""pytest 全局配置与共享 fixture。

关键点：**测试全程不依赖任何 API Key**。
CI 环境里 LLM_API_KEY 明确设为空字符串，服务会自动走规则引擎 ——
这恰好也验证了"零配置可跑"这个设计目标真的成立。
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterator

import pytest

# 必须在导入 app 之前设置环境变量：main.lifespan 会读它们
_TMP_DIR = tempfile.mkdtemp(prefix="resume-matcher-test-")
os.environ["STORE_PERSIST"] = "0"           # 不落盘，避免测试残留
os.environ["DATA_DIR"] = _TMP_DIR
os.environ["LLM_API_KEY"] = ""              # 强制走规则引擎
os.environ["PARSER_ENGINE"] = "auto"
os.environ["CORS_ORIGINS"] = "*"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples"


@pytest.fixture()
def client() -> Iterator["object"]:
    """带完整生命周期的 TestClient（每个测试一份干净状态）。"""
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def sample_jd_text() -> str:
    """内置的 AI 应用岗 JD 示例。"""
    return (EXAMPLES_DIR / "jd_ai_app_engineer.txt").read_text(encoding="utf-8")


@pytest.fixture()
def sample_resume_text() -> str:
    """内置的在读生简历示例。"""
    return (EXAMPLES_DIR / "resume_zhangsan.txt").read_text(encoding="utf-8")


@pytest.fixture()
def other_resume_text() -> str:
    """另一份方向不同的简历（后端），用于对比与批量排序测试。"""
    return (EXAMPLES_DIR / "resume_lisi.txt").read_text(encoding="utf-8")


@pytest.fixture()
def backend_jd_text() -> str:
    """后端岗位 JD，用于验证不同岗位导向下的打分差异。"""
    return (EXAMPLES_DIR / "jd_backend_python.txt").read_text(encoding="utf-8")
