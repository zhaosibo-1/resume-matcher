"""端到端验收脚本：对**真实运行中的服务**做一轮黑盒验证。

和 pytest 的分工
-----------------
pytest 覆盖的是"每个模块内部对不对"；这个脚本覆盖的是
"把服务真的装起来、跑起来，用户能不能用"。两者的失效模式完全不同：
单测全绿但服务起不来的情况在真实项目里并不少见（缺依赖、静态文件路径错、
lifespan 装配顺序错……）。

只依赖标准库（urllib / json），所以任何环境都能跑：

    # 先起服务
    uvicorn app.main:app --host 127.0.0.1 --port 8111
    # 再跑验收
    python scripts/e2e_check.py http://127.0.0.1:8111

退出码：全部通过为 0，有失败为 1（CI 靠它判断成败）。
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8111"

# 绕过系统代理：本地回环不该走代理（开发机上常有全局 http_proxy）
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: object = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  [+] {name}")
    else:
        FAILED.append(name)
        print(f"  [-] {name}   {str(detail)[:300]}")


def request(method: str, path: str, payload: dict | None = None, raw: bytes | None = None,
            content_type: str = "application/json") -> tuple[int, str]:
    if raw is not None:
        data = raw
    elif payload is not None:
        data = json.dumps(payload).encode("utf-8")
    else:
        data = None

    req = urllib.request.Request(BASE + path, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", content_type)
    try:
        with OPENER.open(req, timeout=120) as resp:
            return resp.status, resp.read().decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="ignore")


def get(path: str) -> tuple[int, str]:
    return request("GET", path)


def post(path: str, payload: dict | None = None) -> tuple[int, str]:
    return request("POST", path, payload=payload or {})


def delete(path: str) -> tuple[int, str]:
    return request("DELETE", path)


def jloads(body: str) -> dict:
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {}


def parse_sse(raw: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in raw.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        name, data = "", ""
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data += line[6:]
        if name:
            events.append((name, jloads(data) if data and data != "[DONE]" else {}))
    return events


def section(title: str) -> None:
    print(f"\n[{title}]")


print("=" * 74)
print("简历 · JD 智能匹配系统 —— 端到端验收")
print(f"目标：{BASE}")
print("=" * 74)

# ---------------------------------------------------------------- 1. 基础状态
section("1/9 基础状态")
status, body = get("/api/health")
health = jloads(body)
check("GET /api/health 返回 200", status == 200, body)
check("健康检查报告技能词典规模", health.get("skill_count", 0) > 50, health)
check("未配置 Key 时 llm_enabled=false（零配置可跑）", health.get("llm_enabled") is False)

status, body = get("/api/config")
config = jloads(body)
check("GET /api/config 返回 200", status == 200, body)
check(
    "config 含前端读取的全部字段",
    all(k in config for k in ("engine", "skill_count", "llm_enabled", "model", "provider",
                              "categories", "default_weights")),
    sorted(config),
)
check("未配置 Key 时引擎降级为 rule", config.get("engine") == "rule", config.get("engine"))
check("六个维度定义齐全", len(config.get("dimensions", [])) == 6, len(config.get("dimensions", [])))
check(
    "维度定义带前端需要的 label/description",
    all(d.get("label") and d.get("description") for d in config.get("dimensions", [])),
)
check("默认权重合计为 1", abs(sum(config.get("default_weights", {}).values()) - 1.0) < 0.01)
check("API Key 未出现在响应里", "api_key" not in json.dumps(config, ensure_ascii=False)[:2000], body[:200])

status, body = get("/api/stats")
stats = jloads(body)
check("GET /api/stats 返回 200", status == 200, body)
check("stats 含前端读取的 matches_run", isinstance(stats.get("matches_run"), int), stats)

# ---------------------------------------------------------------- 2. 技能词典
section("2/9 技能词典与归一化")
status, body = get("/api/skills?q=js")
data = jloads(body)
canonicals = {item["canonical"] for item in data.get("items", [])}
check("GET /api/skills 返回 200", status == 200, body[:160])
check("别名 js 能查到 JavaScript", "JavaScript" in canonicals, sorted(canonicals))
check("词典条目带 aliases", all(item.get("aliases") is not None for item in data.get("items", [])))

status, body = get("/api/skills?q=k8s")
check("别名 k8s 能查到 Kubernetes", "Kubernetes" in {i["canonical"] for i in jloads(body)["items"]})

status, body = get("/api/skills?category=%E7%BC%96%E7%A8%8B%E8%AF%AD%E8%A8%80")
check(
    "按大类过滤生效",
    all(i["category"] == "编程语言" for i in jloads(body).get("items", [])),
    body[:160],
)

status, body = get("/api/skills?q=%E7%BB%9D%E5%AF%B9%E4%B8%8D%E5%AD%98%E5%9C%A8")
check("查不到时返回空列表而不是报错", status == 200 and jloads(body)["items"] == [], body[:160])

# ---------------------------------------------------------------- 3. 内置示例
section("3/9 内置示例")
status, body = get("/api/examples")
examples = jloads(body)
check("GET /api/examples 返回 200", status == 200, body[:160])
check("至少 2 份 JD + 2 份简历", len(examples) >= 4, len(examples))

all_readable = True
for item in examples:
    st, bd = get(f"/api/examples/{item['key']}")
    if st != 200 or not jloads(bd).get("text", "").strip():
        all_readable = False
check("每个示例都能真的读到内容（一键填充不会静默失败）", all_readable)

status, _ = get("/api/examples/%E4%B8%8D%E5%AD%98%E5%9C%A8")
check("未知示例返回 404", status == 404, status)

jd_text = jloads(get("/api/examples/jd_ai_app_engineer")[1])["text"]
resume_text = jloads(get("/api/examples/resume_zhangsan")[1])["text"]
other_resume = jloads(get("/api/examples/resume_lisi")[1])["text"]

# ---------------------------------------------------------------- 4. 简历 CRUD
section("4/9 简历管理")
status, body = post("/api/resumes", {"text": resume_text, "name": "验收-张三"})
created = jloads(body)
check("POST /api/resumes 返回 200", status == 200, body[:200])
check("返回了 12 位十六进制 id", len(created.get("resume_id", "")) == 12, created.get("resume_id"))
check("返回了字数与预览", created.get("char_count", 0) > 0 and created.get("preview"))

resume_id = created.get("resume_id", "")
status, body = get("/api/resumes")
check("列表里能查到刚上传的简历", resume_id in {i["resume_id"] for i in jloads(body)}, body[:200])

status, body = get(f"/api/resumes/{resume_id}")
check("详情能取回原文", status == 200 and len(jloads(body).get("text", "")) > 100, status)

status, _ = get("/api/resumes/aaaaaaaaaaaa")
check("取不存在的简历返回 404", status == 404, status)

# 路径穿越：URL 里塞 ../ 不能读到磁盘上的其他文件
status, body = get("/api/resumes/..%2F..%2Fetc%2Fpasswd")
check("路径穿越尝试被拒绝（非 200）", status in (400, 404), status)

status, body = request(
    "POST", "/api/resumes", raw=resume_text.encode("gbk"),
    content_type="text/plain",
)
# 用 multipart 才走文件上传分支；这里退一步直接验证接口不会 500
check("异常请求不会导致 5xx", status < 500, status)

status, body = post("/api/resumes", {})
check("既无 file 也无 text 时返回 400（并给出可操作提示）", status == 400 and "file" in body, body[:200])

status, body = request(
    "POST", "/api/resumes",
    raw=b"",
    content_type="multipart/form-data; boundary=X",
)
check("畸形 multipart 不会导致 5xx", status < 500, status)

status, body = delete(f"/api/resumes/{resume_id}")
check("DELETE 返回 deleted=true", status == 200 and jloads(body).get("deleted") is True, body[:160])
status, _ = delete(f"/api/resumes/{resume_id}")
check("重复删除返回 404", status == 404, status)

# ---------------------------------------------------------------- 5. 解析
section("5/9 解析（JD / 简历）")
status, body = post("/api/parse/jd", {"text": jd_text})
data = jloads(body)
jd = data.get("jd", {})
check("POST /api/parse/jd 返回 200", status == 200, body[:200])
check("识别出岗位名称", bool(jd.get("title")), jd.get("title"))
check("识别出硬性技能", len(jd.get("must_have", [])) >= 5, jd.get("must_have"))
check("识别出加分技能", len(jd.get("nice_to_have", [])) >= 2, jd.get("nice_to_have"))
check("硬性与加分不重叠", not (set(jd.get("must_have", [])) & set(jd.get("nice_to_have", []))))
check("识别出学历要求", jd.get("education_level", 0) >= 2, jd.get("education"))
check("要求条目带了分类", all(r.get("category") for r in jd.get("requirements", [])), jd.get("requirements")[:2])
check("「者优先」没有被判成硬性要求", "Kubernetes" not in jd.get("must_have", []) or
      "Kubernetes" in jd.get("nice_to_have", []), jd.get("must_have"))
check("引擎标记为 rule（测试环境无 Key）", data.get("engine") == "rule", data.get("engine"))

status, body = post("/api/parse/resume", {"text": resume_text})
data = jloads(body)
resume = data.get("resume", {})
check("POST /api/parse/resume 返回 200", status == 200, body[:200])
check("识别出姓名", bool(resume.get("name")), resume.get("name"))
check("识别出学校（且没把学校当成姓名）", "大学" in resume.get("school", ""), resume.get("school"))
check("识别出专业", bool(resume.get("major")), resume.get("major"))
check("识别出技能列表", len(resume.get("skill_names", [])) >= 5, len(resume.get("skill_names", [])))
check("技能能按大类分组", bool(resume.get("skill_groups")), list(resume.get("skill_groups", {}))[:3])
check("识别出带时间的经历", bool(resume.get("experiences")), len(resume.get("experiences", [])))
check("经历带了技能明细", any(e.get("skills") for e in resume.get("experiences", [])))
check("亮点句含量化成果", any(any(ch.isdigit() for ch in h) for h in resume.get("highlights", [])),
      resume.get("highlights")[:3])

status, body = post("/api/parse/jd", {"text": "   "})
check("空文本被拒绝（400）", status == 400, status)

# ---------------------------------------------------------------- 6. 匹配
section("6/9 单份匹配")
status, body = post("/api/match", {"jd_text": jd_text, "resume_text": resume_text})
result = jloads(body)
report = result.get("report", {})
check("POST /api/match 返回 200", status == 200, body[:200])
check("返回了 jd / resume / report / elapsed_ms",
      {"jd", "resume", "report", "elapsed_ms"} <= set(result), sorted(result))
check("总分在 0~100 之间", 0 <= report.get("overall", -1) <= 100, report.get("overall"))
check("给出了结论与等级", bool(report.get("verdict")) and report.get("verdict_level") in "ABCD",
      f"{report.get('verdict')} / {report.get('verdict_level')}")
check("六个维度齐全", len(report.get("dimensions", [])) == 6, len(report.get("dimensions", [])))
check("权重合计为 1", abs(sum(report.get("weights", {}).values()) - 1.0) < 0.02, report.get("weights"))
check("总分 = 各维度加权贡献之和（分数可解释）",
      abs(report.get("overall", 0) - sum(d.get("weighted", 0) for d in report.get("dimensions", []))) < 0.2,
      f"{report.get('overall')} vs {sum(d.get('weighted', 0) for d in report.get('dimensions', []))}")
check("每个维度都有文字说明", all(d.get("detail") for d in report.get("dimensions", [])))
check("维度带 evidence / gaps 字段（前端要渲染）",
      all("evidence" in d and "gaps" in d for d in report.get("dimensions", [])))
check("硬性技能命中比格式为 a/b", "/" in report.get("must_ratio", ""), report.get("must_ratio"))
check("给出了优势/风险/建议",
      isinstance(report.get("strengths"), list) and isinstance(report.get("risks"), list)
      and isinstance(report.get("advice"), list))
check("生成了面试问题", len(report.get("interview_questions", [])) > 0)
check("面试问题都有理由",
      all(q.get("question") and q.get("rationale") for q in report.get("interview_questions", [])))
check("缺失硬性技能与匹配集互斥",
      not (set(report.get("missing_must", [])) & set(report.get("matched_skills", []))))

# 可复现性：招聘场景里同一份简历必须得到同一个分数
status2, body2 = post("/api/match", {"jd_text": jd_text, "resume_text": resume_text})
check("同一输入两次结果完全一致（分数可复现）",
      jloads(body2)["report"] == report, "两次 report 不一致")

# 匹配分必须真的能区分简历质量
status, body = post("/api/match", {"jd_text": jd_text, "resume_text": other_resume})
lisi = jloads(body)["report"]
check("对口简历得分高于不对口简历（分数有区分度）",
      lisi["overall"] < report["overall"], f"李四 {lisi['overall']} vs 张三 {report['overall']}")

status, body = post("/api/match", {"jd_text": jd_text})
check("缺少简历来源返回 400", status == 400 and "resume" in body, body[:200])

status, body = post("/api/match", {"jd_text": "", "resume_text": resume_text})
check("空 JD 被 schema 拦下（422）", status == 422, status)

# ---------------------------------------------------------------- 7. SSE
section("7/9 SSE 流式匹配")
status, body = post("/api/match/stream", {"jd_text": jd_text, "resume_text": resume_text})
events = parse_sse(body)
names = [name for name, _ in events]
check("POST /api/match/stream 返回 200", status == 200, status)
check("事件以 start 开头", names and names[0] == "start", names[:3])
check("事件以 [DONE] 结尾", names and names[-1] == "[DONE]", names[-3:])
check("含 done 事件", "done" in names, names)
stages = [p.get("stage") for n, p in events if n == "stage"]
check("阶段顺序为 jd -> resume -> score", stages == ["jd", "resume", "score"], stages)
progress = [p.get("progress") for n, p in events if n == "stage"]
check("进度百分比单调不减", progress == sorted(progress), progress)
parsed = {p.get("target"): p for n, p in events if n == "parsed"}
check("parsed 事件含 jd / resume 两侧摘要", {"jd", "resume"} <= set(parsed), list(parsed))
check("jd 摘要带硬性技能", bool(parsed.get("jd", {}).get("must_have")))
check("resume 摘要带姓名与技能数",
      bool(parsed.get("resume", {}).get("name")) and parsed.get("resume", {}).get("skill_count", 0) > 0,
      parsed.get("resume"))
scored = next((p for n, p in events if n == "scored"), {})
check("scored 事件带六维分数", len(scored.get("dimensions", [])) == 6, scored)
done = next((p for n, p in events if n == "done"), {})
check("done 事件是完整的 MatchResponse", {"jd", "resume", "report", "elapsed_ms"} <= set(done), sorted(done))
check("流式结果与非流式结果一致",
      abs(done.get("report", {}).get("overall", -1) - report.get("overall", -2)) < 0.001,
      f"{done.get('report', {}).get('overall')} vs {report.get('overall')}")

status, body = post("/api/match/stream", {"jd_text": jd_text, "resume_id": "aaaaaaaaaaaa"})
check("流式接口的参数错误在建流前返回普通 404", status == 404, status)

# ---------------------------------------------------------------- 8. 批量与重算
section("8/9 批量排序与权重重算")
status, body = post("/api/match/batch", {
    "jd_text": jd_text,
    "resumes": [
        {"name": "张三", "resume_text": resume_text},
        {"name": "李四", "resume_text": other_resume},
    ],
})
batch = jloads(body)
check("POST /api/match/batch 返回 200", status == 200, body[:200])
check("批量返回 2 条", batch.get("total") == 2, batch.get("total"))
scores = [i["overall"] for i in batch.get("items", [])]
check("结果按总分降序排列（HR 视角最需要的视图）", scores == sorted(scores, reverse=True), scores)
check("批量条目带 must_ratio 与 verdict",
      all(i.get("must_ratio") and i.get("verdict") for i in batch.get("items", [])))

status, body = post("/api/recompute", {"jd": result["jd"], "resume": result["resume"]})
recomputed = jloads(body)
check("POST /api/recompute 返回 200", status == 200, body[:200])
check("不传权重时重算结果与首次一致",
      abs(recomputed["report"]["overall"] - report["overall"]) < 0.001,
      f"{recomputed['report']['overall']} vs {report['overall']}")
check("重算不改动解析结果", recomputed["jd"] == result["jd"] and recomputed["resume"] == result["resume"])

zero = {key: 0.0 for key in report["weights"]}
zero["must_have"] = 1.0
status, body = post("/api/recompute", {"jd": result["jd"], "resume": result["resume"], "weights": zero})
tuned = jloads(body)["report"]
must_dim = next(d for d in tuned["dimensions"] if d["key"] == "must_have")
check("把权重全压到硬性技能后，总分收敛到该维度得分",
      abs(tuned["overall"] - must_dim["score"]) < 0.05, f"{tuned['overall']} vs {must_dim['score']}")
check("重算是毫秒级的（前端拖滑条的体验基础）",
      jloads(body).get("elapsed_ms", 9999) < 500, jloads(body).get("elapsed_ms"))

status, body = post("/api/recompute", {"jd": {}, "resume": {}})
empty_report = jloads(body).get("report", {})
check("空解析对象也能返回（不 5xx）", status == 200, status)
check("全维度不适用时说明「不代表真实匹配度」",
      any("不代表真实匹配度" in r for r in empty_report.get("risks", [])),
      empty_report.get("risks"))

# ---------------------------------------------------------------- 9. 前端与大模型
section("9/9 前端页面与大模型连通性")
status, body = get("/")
check("GET / 返回 200", status == 200, status)
check("页面含标题", "匹配" in body)
check("页面引入了前端脚本", "<script>" in body)
for element_id in ("jdText", "resumeText", "btnMatch", "resultArea", "dictGrid", "btnPing"):
    check(f"页面含元素 #{element_id}", f'id="{element_id}"' in body)

status, body = post("/api/llm/ping")
ping = jloads(body)
check("POST /api/llm/ping 返回结构化结果（不抛异常）", status == 200 and "ok" in ping, body[:200])
check("未配置 Key 时给出可读的降级说明", ping.get("kind") == "config" and "规则引擎" in ping.get("message", ""),
      ping)

status, body = get("/openapi.json")
check("OpenAPI 文档可访问", status == 200 and "/api/match" in body)

# ---------------------------------------------------------------- 汇总
print("\n" + "=" * 74)
total = len(PASSED) + len(FAILED)
print(f"验收结果：{len(PASSED)}/{total} 项通过")
if FAILED:
    print("\n失败项：")
    for name in FAILED:
        print(f"  [-] {name}")
    print("=" * 74)
    sys.exit(1)

print("全部通过 ✓")
print("=" * 74)
