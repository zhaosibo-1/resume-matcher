"""FastAPI 服务层：把解析与打分能力暴露成 HTTP 接口。

接口一览
--------
========================================  ==========================================
GET    /api/health                         健康检查
GET    /api/config                         脱敏配置（含当前生效引擎与维度定义）
GET    /api/stats                          运行统计
GET    /api/skills                         技能词典查询（可搜索/按大类过滤）
GET    /api/examples                       内置示例列表
GET    /api/examples/{key}                 示例内容
POST   /api/resumes                        上传/粘贴简历
GET    /api/resumes                        简历列表
GET    /api/resumes/{id}                   简历详情
DELETE /api/resumes/{id}                   删除简历
POST   /api/parse/jd                       只解析 JD
POST   /api/parse/resume                   只解析简历
POST   /api/match                          单份匹配（核心接口）
POST   /api/match/stream                   SSE 流式匹配（逐步推送解析进度）
POST   /api/match/batch                    一个 JD × 多份简历排序
POST   /api/recompute                      只重算权重（毫秒级，不重新解析）
POST   /api/llm/ping                        测试大模型连通性
========================================  ==========================================

关于 `/api/recompute`
---------------------
这是本服务在工程上值得一提的一个设计。前端拖动权重滑条时，
解析结果并没有变化，变化的只是"怎么看这些结果"。如果每次都重跑
完整流水线（尤其是调用大模型），体验会非常糟糕（秒级等待）。

所以前端在第一次匹配后会缓存住 ``ParsedJD`` 与 ``ParsedResume``，
拖动滑条时把这两个对象原样回传，服务端只重算加权和 —— 响应稳定在毫秒级。
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from . import __version__
from .config import Settings, load_settings
from .extractor import Extractor
from .llm import ERR_CONFIG, LLMClient, LLMError
from .matcher import match as run_match
from .parser import SENIORITY_LEVELS, parse_jd, parse_resume
from .schemas import (
    DEFAULT_WEIGHTS,
    DIMENSION_DESCRIPTIONS,
    DIMENSION_LABELS,
    DIMENSION_ORDER,
    BatchMatchItem,
    BatchMatchRequest,
    BatchMatchResponse,
    ConfigResponse,
    DimensionDef,
    ExampleDetail,
    ExampleItem,
    HealthResponse,
    MatchRequest,
    MatchResponse,
    RecomputeRequest,
    ResumeListItem,
    ResumeUploadResponse,
    StatsResponse,
)
from .skills import DEFAULT_NORMALIZER
from .store import MAX_RESUME_CHARS, ResumeStore

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = PROJECT_ROOT / "web"
EXAMPLES_DIR = PROJECT_ROOT / "examples"

#: 内置示例的文件名 -> (展示名, 描述, 类型)
EXAMPLE_CATALOG: dict[str, tuple[str, str, str]] = {
    "jd_ai_app_engineer": ("AI 应用开发工程师（校招）", "大厂校招 JD，覆盖 RAG / Agent / 微调方向", "jd"),
    "jd_backend_python": ("Python 后端开发工程师", "偏工程的后端岗位，考察框架与运维能力", "jd"),
    "resume_zhangsan": ("简历 · 张三（AI 应用方向）", "在校生简历，RAG 项目 + 大模型实习", "resume"),
    "resume_lisi": ("简历 · 李四（后端方向）", "后端方向简历，与 AI 岗匹配度较低，适合做对比", "resume"),
}

#: 允许上传的纯文本类扩展名
_TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".json", ".csv", ".log"}

#: 需要可选依赖的扩展名 -> (依赖包名, 加载函数名)
_BINARY_SUFFIXES = {".pdf": "pypdf", ".docx": "python-docx"}


# ===========================================================================
# 应用生命周期
# ===========================================================================


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """启动时装配依赖，关闭时释放 HTTP 连接池。

    所有可变状态都挂在 ``app.state`` 上，而不是模块级全局变量 ——
    这样测试里可以同时起多个互不干扰的实例。
    """
    settings = load_settings()
    settings.ensure_dirs()

    client = LLMClient(settings)
    extractor = Extractor(settings, client)
    store = ResumeStore(settings.data_dir, persist=settings.store_persist)

    app.state.settings = settings
    app.state.llm = client
    app.state.extractor = extractor
    app.state.store = store
    app.state.stats = {
        "matches_run": 0,
        "llm_calls": 0,
        "llm_failures": 0,
    }

    logger.info(
        "服务已就绪：引擎=%s，技能词典 %d 条，简历存储 %d 份",
        settings.active_engine,
        len(DEFAULT_NORMALIZER.canonical_names),
        store.count,
    )

    try:
        yield
    finally:
        await client.aclose()


app = FastAPI(
    title="简历 - JD 智能匹配系统",
    description=(
        "把一份 JD 和一份简历变成**可解释**的匹配报告："
        "六维加权打分、逐条证据溯源、缺口清单与针对性面试问题。"
    ),
    version=__version__,
    lifespan=lifespan,
)

# CORS：默认放开是为了让前端能独立部署；生产环境应收窄到具体域名
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================================
# 小工具
# ===========================================================================


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_store(request: Request) -> ResumeStore:
    return request.app.state.store


def get_extractor(request: Request) -> Extractor:
    return request.app.state.extractor


def get_llm(request: Request) -> LLMClient:
    return request.app.state.llm


def bump(request: Request, key: str, delta: int = 1) -> None:
    """累加运行统计。"""
    stats = request.app.state.stats
    stats[key] = stats.get(key, 0) + delta


def dimension_defs() -> list[DimensionDef]:
    """六个维度的定义（前端渲染权重滑条与说明用）。"""
    return [
        DimensionDef(
            key=key,
            label=DIMENSION_LABELS[key],
            description=DIMENSION_DESCRIPTIONS[key],
            default_weight=DEFAULT_WEIGHTS[key],
        )
        for key in DIMENSION_ORDER
    ]


def read_upload(filename: str, data: bytes) -> str:
    """把上传的文件字节解码成文本。

    策略：
    - 纯文本类（.txt/.md/...）：优先 UTF-8（带 BOM 也能正确处理），
      失败再退 GBK —— 国内用户用记事本存的 txt 经常是 GBK 编码。
    - PDF / DOCX：需要可选依赖，缺失时**明确报错并给出安装命令**，
      而不是静默返回空文本让用户以为"我的简历没问题"。

    Raises:
        HTTPException: 格式不支持或依赖缺失。
    """
    if not data:
        raise HTTPException(status_code=400, detail="文件内容为空")

    suffix = Path(filename or "").suffix.lower()

    if suffix in _TEXT_SUFFIXES or suffix == "":
        # utf-8-sig 能同时正确处理"有 BOM"和"没 BOM"两种情况
        for encoding in ("utf-8-sig", "utf-8", "gbk", "big5"):
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise HTTPException(status_code=400, detail="无法识别文件编码，请转存为 UTF-8 后重试")

    if suffix in _BINARY_SUFFIXES:
        package = _BINARY_SUFFIXES[suffix]
        try:
            if suffix == ".pdf":
                import pypdf  # type: ignore

                import io

                reader = pypdf.PdfReader(io.BytesIO(data))
                return "\n".join((page.extract_text() or "") for page in reader.pages)
            else:
                import io

                import docx  # type: ignore

                document = docx.Document(io.BytesIO(data))
                parts = [p.text for p in document.paragraphs]
                for table in document.tables:
                    for row in table.rows:
                        parts.append(" | ".join(cell.text for cell in row.cells))
                return "\n".join(parts)
        except ImportError:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"解析 {suffix} 文件需要可选依赖 {package}。"
                    f"请执行 pip install {package} 后重试，或直接把简历内容粘贴到文本框。"
                ),
            ) from None
        except Exception as exc:  # 解析库对损坏文件会抛各种异常
            raise HTTPException(status_code=400, detail=f"文件解析失败：{exc}") from exc

    raise HTTPException(
        status_code=400,
        detail=f"暂不支持的文件格式 {suffix}。支持：txt / md / json / csv / pdf / docx，或直接粘贴文本。",
    )


# ===========================================================================
# 元信息接口
# ===========================================================================


@app.get("/api/health", response_model=HealthResponse, tags=["元信息"])
def health(request: Request) -> HealthResponse:
    """健康检查。容器与 CI 用它探测服务是否就绪。"""
    settings = get_settings(request)
    return HealthResponse(
        status="ok",
        version=__version__,
        llm_enabled=settings.llm_enabled,
        skill_count=len(DEFAULT_NORMALIZER.canonical_names),
    )


@app.get("/api/config", response_model=ConfigResponse, tags=["元信息"])
def config(request: Request) -> ConfigResponse:
    """返回脱敏配置。

    **绝不返回 API Key 原文**。前端只需要知道"有没有配 Key、用的哪个模型"。
    """
    settings = get_settings(request)
    extractor = get_extractor(request)
    describe = settings.describe()

    categories: list[str] = []
    for name in DEFAULT_NORMALIZER.canonical_names:
        cat = DEFAULT_NORMALIZER.category_of(name)
        if cat not in categories:
            categories.append(cat)

    return ConfigResponse(
        llm_enabled=settings.llm_enabled,
        provider=str(describe["provider"]),
        model=str(describe["model"]),
        base_url=str(describe["base_url"]),
        engine=extractor.describe()["active_engine"],
        dimensions=dimension_defs(),
        default_weights=dict(DEFAULT_WEIGHTS),
        max_upload_mb=settings.max_upload_mb,
        skill_count=len(DEFAULT_NORMALIZER.canonical_names),
        categories=categories,
        # 直接复用解析器的白名单，避免"后端认的层级"和"前端给的选项"两份定义漂移
        seniority_options=list(SENIORITY_LEVELS),
        version=__version__,
    )


@app.get("/api/stats", response_model=StatsResponse, tags=["元信息"])
def stats(request: Request) -> StatsResponse:
    """运行统计，用于前端展示"已分析 N 次"。"""
    settings = get_settings(request)
    client = get_llm(request)
    store = get_store(request)
    counters = request.app.state.stats

    return StatsResponse(
        resumes_stored=store.count,
        matches_run=int(counters.get("matches_run", 0)),
        skills_in_dict=len(DEFAULT_NORMALIZER.canonical_names),
        llm_calls=client.calls,
        llm_failures=client.failures,
        uptime_seconds=int(time.time() - settings.started_at),
    )


@app.get("/api/skills", tags=["元信息"])
def skills(
    request: Request,
    q: str = "",
    category: str = "",
    limit: int = 300,
) -> dict[str, Any]:
    """技能词典查询。

    这个接口存在的意义：让用户能亲眼看到归一化是怎么做的 ——
    输入 ``js`` 能看到它指向 ``JavaScript``，输入 ``k8s`` 能看到 ``Kubernetes``。
    这是"可解释"的一部分，比在文档里写一段说明有效得多。
    """
    keyword = (q or "").strip().lower()
    cat_filter = (category or "").strip()

    items: list[dict[str, Any]] = []
    for name in DEFAULT_NORMALIZER.canonical_names:
        cat = DEFAULT_NORMALIZER.category_of(name)
        if cat_filter and cat != cat_filter:
            continue
        aliases = DEFAULT_NORMALIZER.aliases_of(name)
        parents = list(DEFAULT_NORMALIZER.parents_of(name))
        if keyword and keyword not in name.lower() and not any(keyword in a for a in aliases):
            continue
        items.append(
            {
                "canonical": name,
                "category": cat,
                "aliases": aliases,
                "parents": parents,
                "alias_count": len(aliases),
            }
        )
        if len(items) >= max(1, min(limit, 1000)):
            break

    return {
        "total": len(DEFAULT_NORMALIZER.canonical_names),
        "matched": len(items),
        "query": q,
        "category": category,
        "items": items,
        "categories": sorted({DEFAULT_NORMALIZER.category_of(n) for n in DEFAULT_NORMALIZER.canonical_names}),
    }


@app.get("/api/examples", response_model=list[ExampleItem], tags=["元信息"])
def examples() -> list[ExampleItem]:
    """列出内置示例（前端"一键填充"按钮用）。"""
    out: list[ExampleItem] = []
    for key, (label, description, kind) in EXAMPLE_CATALOG.items():
        path = EXAMPLES_DIR / f"{key}.txt"
        if not path.exists():
            continue
        out.append(ExampleItem(key=key, label=label, description=description, kind=kind))  # type: ignore[arg-type]
    return out


@app.get("/api/examples/{key}", response_model=ExampleDetail, tags=["元信息"])
def example_detail(key: str) -> ExampleDetail:
    """读取某个示例的全文。"""
    if key not in EXAMPLE_CATALOG:
        raise HTTPException(status_code=404, detail=f"示例不存在：{key}")
    path = EXAMPLES_DIR / f"{key}.txt"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"示例文件缺失：{path.name}")
    label, _description, kind = EXAMPLE_CATALOG[key]
    return ExampleDetail(key=key, label=label, kind=kind, text=path.read_text(encoding="utf-8"))


# ===========================================================================
# 简历管理
# ===========================================================================


@app.post("/api/resumes", response_model=ResumeUploadResponse, tags=["简历"])
async def upload_resume(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    text: str = Form(default=""),
    name: str = Form(default=""),
) -> ResumeUploadResponse:
    """上传或粘贴一份简历。

    支持两种方式（二选一）：
    - ``file``：上传文件（txt / md / json / csv / pdf / docx）
    - ``text``：直接粘贴文本

    也支持 ``application/json`` 请求体（``{"text": "...", "name": "..."}``），
    方便脚本调用。
    """
    settings = get_settings(request)
    store = get_store(request)

    body_text = (text or "").strip()
    display_name = (name or "").strip()

    if file is not None:
        data = await file.read()
        limit = settings.max_upload_mb * 1024 * 1024
        if len(data) > limit:
            raise HTTPException(
                status_code=413,
                detail=f"文件过大（{len(data) / 1024 / 1024:.1f} MB，上限 {settings.max_upload_mb} MB）",
            )
        body_text = read_upload(file.filename or "", data)
        display_name = display_name or Path(file.filename or "").stem
    elif not body_text:
        # 兼容 JSON 请求体
        try:
            payload = await request.json()
        except Exception:
            payload = None
        if isinstance(payload, dict):
            body_text = str(payload.get("text") or "").strip()
            display_name = display_name or str(payload.get("name") or "").strip()

    if not body_text:
        raise HTTPException(status_code=400, detail="请提供 file 或 text 参数")

    if len(body_text) > MAX_RESUME_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"简历内容过长（{len(body_text)} 字符，上限 {MAX_RESUME_CHARS}）",
        )

    try:
        record = store.add(body_text, name=display_name, source="upload" if file else "paste")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ResumeUploadResponse(
        resume_id=record.resume_id,
        name=record.name,
        char_count=record.char_count,
        preview=record.text[:200],
    )


@app.get("/api/resumes", response_model=list[ResumeListItem], tags=["简历"])
def list_resumes(request: Request) -> list[ResumeListItem]:
    """列出已上传的简历。"""
    return [
        ResumeListItem(
            resume_id=item.resume_id,
            name=item.name,
            char_count=item.char_count,
            uploaded_at=item.uploaded_at,
        )
        for item in get_store(request).list()
    ]


@app.get("/api/resumes/{resume_id}", tags=["简历"])
def resume_detail(request: Request, resume_id: str) -> dict[str, Any]:
    """读取一份简历的原文。"""
    record = get_store(request).get(resume_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"简历不存在：{resume_id}")
    return {
        "resume_id": record.resume_id,
        "name": record.name,
        "text": record.text,
        "char_count": record.char_count,
        "uploaded_at": record.uploaded_at,
    }


@app.delete("/api/resumes/{resume_id}", tags=["简历"])
def delete_resume(request: Request, resume_id: str) -> dict[str, Any]:
    """删除一份简历。"""
    ok = get_store(request).remove(resume_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"简历不存在：{resume_id}")
    return {"deleted": True, "resume_id": resume_id}


# ===========================================================================
# 解析接口
# ===========================================================================


@app.post("/api/parse/jd", tags=["解析"])
async def parse_jd_endpoint(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
    """只解析 JD，返回结构化结果（用于调试与前端即时预览）。"""
    text = str(payload.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="请提供 text 字段")

    extractor = get_extractor(request)
    settings = get_settings(request)
    llm_data = await extractor.extract_jd(text)
    parsed = parse_jd(text, llm_data=llm_data) if llm_data else parse_jd(text)

    return {
        "jd": parsed.model_dump(),
        "engine": "llm" if llm_data else "rule",
        "llm_error": extractor.last_error if settings.llm_enabled and not llm_data else "",
    }


@app.post("/api/parse/resume", tags=["解析"])
async def parse_resume_endpoint(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
    """只解析简历，返回结构化结果。"""
    text = str(payload.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="请提供 text 字段")

    extractor = get_extractor(request)
    llm_data = await extractor.extract_resume(text)
    parsed = parse_resume(text, llm_data=llm_data) if llm_data else parse_resume(text)

    return {
        "resume": parsed.model_dump(),
        "engine": "llm" if llm_data else "rule",
        "llm_error": extractor.last_error if not llm_data else "",
    }


# ===========================================================================
# 匹配接口
# ===========================================================================


async def resolve_resume_text(request: Request, body: MatchRequest) -> str:
    """从请求里取出简历文本（支持 resume_text 与 resume_id 两种来源）。"""
    text = (body.resume_text or "").strip()
    if text:
        return text

    if body.resume_id:
        record = get_store(request).get(body.resume_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"简历不存在：{body.resume_id}")
        return record.text

    raise HTTPException(status_code=400, detail="请提供 resume_text（粘贴文本）或 resume_id（已上传简历）")


async def run_pipeline(
    request: Request,
    jd_text: str,
    resume_text: str,
    *,
    weights: Optional[dict[str, float]] = None,
    want_questions: bool = True,
    question_count: int = 6,
    emit: Any = None,
) -> MatchResponse:
    """完整流水线：解析 JD → 解析简历 → 六维打分。

    Args:
        emit: 可选的异步回调 ``emit(event_name, payload)``，用于 SSE 推送进度。
    """
    extractor = get_extractor(request)
    settings = get_settings(request)
    started = time.perf_counter()

    async def notify(event: str, payload: dict[str, Any]) -> None:
        if emit is not None:
            await emit(event, payload)

    # ---- 解析 JD ----
    await notify("stage", {"stage": "jd", "label": "解析职位描述", "progress": 15})
    jd_llm = await extractor.extract_jd(jd_text)
    bump(request, "llm_calls", 1 if jd_llm else 0)
    jd = parse_jd(jd_text, llm_data=jd_llm) if jd_llm else parse_jd(jd_text)
    await notify(
        "parsed",
        {
            "target": "jd",
            "title": jd.title,
            "must_have": jd.must_have,
            "nice_to_have": jd.nice_to_have,
            "engine": "llm" if jd_llm else "rule",
        },
    )

    # ---- 解析简历 ----
    await notify("stage", {"stage": "resume", "label": "解析简历", "progress": 45})
    resume_llm = await extractor.extract_resume(resume_text)
    bump(request, "llm_calls", 1 if resume_llm else 0)
    resume = parse_resume(resume_text, llm_data=resume_llm) if resume_llm else parse_resume(resume_text)
    await notify(
        "parsed",
        {
            "target": "resume",
            "name": resume.name,
            "skill_count": len(resume.skill_names),
            "total_years": resume.total_years,
            "engine": "llm" if resume_llm else "rule",
        },
    )

    # ---- 打分 ----
    await notify("stage", {"stage": "score", "label": "六维加权打分", "progress": 75})
    report = run_match(
        jd,
        resume,
        weights=weights,
        want_questions=want_questions,
        question_count=question_count,
        engine=settings.active_engine,
    )
    bump(request, "matches_run")

    elapsed = int((time.perf_counter() - started) * 1000)
    await notify(
        "scored",
        {
            "overall": report.overall,
            "verdict": report.verdict,
            "verdict_level": report.verdict_level,
            "dimensions": [
                {"key": d.key, "label": d.label, "score": d.score, "weight": d.weight}
                for d in report.dimensions
            ],
        },
    )

    return MatchResponse(jd=jd, resume=resume, report=report, elapsed_ms=elapsed)


@app.post("/api/match", response_model=MatchResponse, tags=["匹配"])
async def match_endpoint(request: Request, body: MatchRequest) -> MatchResponse:
    """单份简历 × 单个 JD 的完整匹配。**这是核心接口。**"""
    resume_text = await resolve_resume_text(request, body)
    return await run_pipeline(
        request,
        body.jd_text,
        resume_text,
        weights=body.weights,
        want_questions=body.want_questions,
        question_count=body.question_count,
    )


@app.post("/api/match/stream", tags=["匹配"])
async def match_stream(request: Request, body: MatchRequest) -> StreamingResponse:
    """SSE 流式匹配：把解析进度实时推给前端。

    事件序列::

        event: start      -> 请求开始
        event: stage      -> 当前阶段（jd / resume / score）+ 进度百分比
        event: parsed     -> 某一边解析完成（含关键字段的摘要）
        event: scored     -> 打分完成（含六维分数）
        event: done       -> 完整报告（data 为 MatchResponse 的 JSON）
        event: error      -> 出错
        event: [DONE]     -> 流结束

    为什么用 SSE 而不是 WebSocket？因为这是**单向**的进度推送，
    SSE 基于普通 HTTP，不需要协议升级、不需要心跳、断线由浏览器自动重连，
    在这个场景下比 WebSocket 简单得多也稳得多。
    """
    resume_text = await resolve_resume_text(request, body)

    async def event_stream() -> AsyncIterator[str]:
        import asyncio

        def pack(event: str, data: Any) -> str:
            return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        # 用队列把 run_pipeline 的进度回调转成 SSE 事件流。
        # 之所以不直接在回调里 yield：回调是被 run_pipeline 调用的普通协程，
        # 没法跨函数边界 yield；队列是这里最自然的解耦方式。
        queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()

        async def emit(event: str, payload: dict[str, Any]) -> None:
            await queue.put((event, payload))

        task = asyncio.create_task(
            run_pipeline(
                request,
                body.jd_text,
                resume_text,
                weights=body.weights,
                want_questions=body.want_questions,
                question_count=body.question_count,
                emit=emit,
            )
        )

        try:
            yield pack("start", {"elapsed": 0})

            # 边跑边推：队列有数据就立刻发出，没有就短暂等待后重试。
            # 用 50ms 轮询而不是复杂的事件同步，是因为进度推送对延迟本就不敏感，
            # 简单可靠的实现比精巧的实现更值得。
            while not task.done() or not queue.empty():
                try:
                    event, payload = await asyncio.wait_for(queue.get(), timeout=0.05)
                except asyncio.TimeoutError:
                    continue
                yield pack(event, payload)

            try:
                result: MatchResponse = await task
            except HTTPException as exc:
                yield pack("error", {"kind": "http", "message": exc.detail, "status": exc.status_code})
            except Exception as exc:  # noqa: BLE001 - 兜底：任何异常都不能让流挂断
                logger.exception("流式匹配失败")
                yield pack("error", {"kind": "internal", "message": str(exc)})
            else:
                yield pack("done", json.loads(result.model_dump_json()))
        finally:
            # 客户端提前断开时，取消还在跑的任务，避免后台空转
            if not task.done():
                task.cancel()

        yield "event: [DONE]\ndata: {}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # 关掉 Nginx 等反向代理的缓冲，否则事件会被攒着一起发，失去流式意义
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/match/batch", response_model=BatchMatchResponse, tags=["匹配"])
async def match_batch(request: Request, body: BatchMatchRequest) -> BatchMatchResponse:
    """一个 JD × 多份简历的批量排序（HR 视角）。

    实现上复用了同一套解析与打分逻辑，只是循环处理多份简历。
    JD 只解析一次 —— 这是这个接口最主要的性能优化点。
    """
    started = time.perf_counter()
    extractor = get_extractor(request)
    settings = get_settings(request)

    jd_llm = await extractor.extract_jd(body.jd_text)
    jd = parse_jd(body.jd_text, llm_data=jd_llm) if jd_llm else parse_jd(body.jd_text)

    items: list[BatchMatchItem] = []
    for entry in body.resumes:
        text = (entry.resume_text or "").strip()
        if not text:
            continue
        resume_llm = await extractor.extract_resume(text)
        resume = parse_resume(text, llm_data=resume_llm) if resume_llm else parse_resume(text)
        report = run_match(
            jd,
            resume,
            weights=body.weights,
            want_questions=False,
            engine=settings.active_engine,
        )
        bump(request, "matches_run")

        items.append(
            BatchMatchItem(
                resume_id=entry.resume_id,
                name=entry.name or resume.name or "未命名",
                overall=report.overall,
                verdict=report.verdict,
                verdict_level=report.verdict_level,
                must_ratio=report.must_ratio,
                missing_must=report.missing_must,
                top_strength=report.strengths[0] if report.strengths else "",
            )
        )

    # 按总分从高到低排序 —— 这是 HR 最需要的视图
    items.sort(key=lambda item: -item.overall)

    return BatchMatchResponse(
        jd_title=jd.title,
        total=len(items),
        items=items,
        elapsed_ms=int((time.perf_counter() - started) * 1000),
    )


@app.post("/api/recompute", response_model=MatchResponse, tags=["匹配"])
async def recompute(request: Request, body: RecomputeRequest) -> MatchResponse:
    """只重算权重，不重新解析（毫秒级）。

    前端拖动权重滑条时调用。把上一次的 ``ParsedJD`` / ``ParsedResume``
    原样回传即可 —— 解析结果没有变，变的只是"怎么看它"。
    """
    settings = get_settings(request)
    started = time.perf_counter()

    report = run_match(
        body.jd,
        body.resume,
        weights=body.weights,
        want_questions=body.want_questions,
        question_count=body.question_count,
        engine=settings.active_engine,
    )

    return MatchResponse(
        jd=body.jd,
        resume=body.resume,
        report=report,
        elapsed_ms=int((time.perf_counter() - started) * 1000),
    )


@app.post("/api/llm/ping", tags=["元信息"])
async def llm_ping(request: Request) -> dict[str, Any]:
    """测试大模型连通性。返回结构化结果而不是抛异常，方便前端展示失败原因。"""
    settings = get_settings(request)
    if not settings.llm_enabled:
        return {
            "ok": False,
            "kind": ERR_CONFIG,
            "message": "未配置 LLM_API_KEY。服务当前以**规则引擎**运行，功能完整、结果确定，只是抽取精度略低。",
            "model": "",
            "latency_ms": 0,
        }

    result = await get_llm(request).ping()
    if not result["ok"]:
        bump(request, "llm_failures")
    return result


# ===========================================================================
# 前端页面
# ===========================================================================


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> HTMLResponse:
    """零构建单页前端。"""
    page = WEB_DIR / "index.html"
    if not page.exists():
        return HTMLResponse(
            "<h1>前端页面缺失</h1><p>未找到 web/index.html。"
            "可以直接访问 <a href='/docs'>/docs</a> 使用接口。</p>",
            status_code=404,
        )
    return HTMLResponse(page.read_text(encoding="utf-8"))


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> JSONResponse:
    """避免浏览器请求 favicon 时打出 404 日志噪声。"""
    return JSONResponse({}, status_code=204)


__all__ = ["app"]
