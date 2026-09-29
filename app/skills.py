"""技能词典与归一化层。

这是本项目最"接地气"的一层：真实 JD 与简历里同一个技能会有五花八门的写法
（`js` / `JavaScript` / `javascript` / `ES6`），如果直接做字符串相等判断，
匹配率会低得离谱。

本模块解决三件事：

1. **别名归一化**：把各种写法收敛到唯一的 canonical 名字。
   例：`js` / `ecmascript` / `es6` → `JavaScript`
2. **带位置扫描**：在文本里找出所有技能出现的位置（含 offset），
   这是"证据可溯源"的基础——前端能据此把命中的原文高亮出来。
3. **分级词表**：把"精通 / 熟练 / 了解"这类程度词映射成分值，
   用于判断简历里某项技能的真实掌握程度。

设计取舍（重要）
----------------
本层**刻意不使用大模型**，理由：
- 归一化是确定性任务，规则做又快又稳，且结果 100% 可解释、可复现；
- 上 LLM 会让「同一份简历跑两次结果不一样」，这在匹配场景里是致命的；
- 大模型的算力应该花在它真正擅长的地方——**从非结构化文本里抽取**。

因此本项目的分工是：
    词典规则  → 归一化、位置定位、分级   （确定性）
    LLM/离线器 → 从自然语言里识别与抽取   （理解性）
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 技能词典
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SkillEntry:
    """一个标准技能及其所有已知写法。

    Attributes:
        canonical: 归一化后的标准名（作为全系统唯一标识）。
        category:  技能大类，用于前端分组展示与"技能栈结构"分析。
        aliases:   别名元组，**必须包含 canonical 本身**（大小写不敏感）。
        parents:   上位技能（"会 FAISS" 隐含 "会用向量数据库"）。
            这一层用于解决 JD 里"概念词 + 具体实现"同时出现的情况：
            JD 写「了解向量数据库（FAISS / Milvus）」，简历只写了 FAISS，
            按字面匹配会判成"向量数据库缺失"，这显然是错的。
        auto_alias: 是否把 canonical 本身自动登记为一个别名。默认 True。
            只有极短且歧义极大的名字才需要关掉，例如单字母的 ``C`` 与 ``R``：
            它们的 canonical 是 "C"、"R"，一旦自动登记成别名，
            正文里任意一个大写的 C/R 都会被当成技能命中。
            这类技能只能通过显式别名（"C语言"、"R语言"）匹配。
    """

    canonical: str
    category: str
    aliases: tuple[str, ...]
    parents: tuple[str, ...] = ()
    auto_alias: bool = True


# 技能大类常量：集中定义，避免各处硬编码字符串导致前后端对不上
CAT_LANGUAGE = "编程语言"
CAT_AI = "AI/机器学习"
CAT_LLM = "大模型应用"
CAT_VECTOR_DB = "向量检索"
CAT_BACKEND = "后端开发"
CAT_FRONTEND = "前端开发"
CAT_DATABASE = "数据库"
CAT_CLOUD = "云原生与运维"
CAT_MOBILE = "移动端"
CAT_DATA = "数据处理"
CAT_TOOL = "开发工具"
CAT_METHOD = "工程方法"
CAT_DOMAIN = "领域方向"


# 说明：中文别名尽量选长度 >= 3 的词，避免「模型」「学习」这类两字词造成大面积误报。
# 英文别名统一小写书写（匹配时用 IGNORECASE，这里的小写只是便于人工维护）。
SKILL_DB: tuple[SkillEntry, ...] = (
    # ---------------- 编程语言 ----------------
    SkillEntry("Python", CAT_LANGUAGE, ("python", "py3", "python3", "python2")),
    SkillEntry("Java", CAT_LANGUAGE, ("java", "java8", "java11", "java17")),
    SkillEntry("C++", CAT_LANGUAGE, ("c++", "cpp", "cplusplus")),
    SkillEntry("C#", CAT_LANGUAGE, ("c#", "csharp", "c sharp")),
    # 注意：单词 "C" 未收录为可自动匹配的别名（auto_alias=False）。
    # 原因是它在英文文本里几乎无处不在（变量名、编号 "C 选项"），
    # 误报率远高于收益。若确需匹配，走 "C语言" 这个显式别名。
    SkillEntry("C", CAT_LANGUAGE, ("c语言", "c 语言"), auto_alias=False),
    SkillEntry("Go", CAT_LANGUAGE, ("golang", "go语言", "go 语言")),
    SkillEntry("Rust", CAT_LANGUAGE, ("rust",)),
    SkillEntry("JavaScript", CAT_LANGUAGE, ("javascript", "js", "es6", "ecmascript", "js/es6")),
    SkillEntry("TypeScript", CAT_LANGUAGE, ("typescript", "ts", "ts类型")),
    SkillEntry("SQL", CAT_LANGUAGE, ("sql", "sql语言")),
    SkillEntry("Shell", CAT_LANGUAGE, ("shell", "bash", "shell脚本", "shell 脚本", "zsh")),
    # "R" 同理：单字母误报严重，只认「R语言」这一明确写法。
    SkillEntry("R", CAT_LANGUAGE, ("r语言", "r 语言"), auto_alias=False),
    SkillEntry("Scala", CAT_LANGUAGE, ("scala",)),
    SkillEntry("Kotlin", CAT_LANGUAGE, ("kotlin",)),
    SkillEntry("Swift", CAT_LANGUAGE, ("swift",)),
    SkillEntry("PHP", CAT_LANGUAGE, ("php",)),
    SkillEntry("Ruby", CAT_LANGUAGE, ("ruby",)),
    SkillEntry("MATLAB", CAT_LANGUAGE, ("matlab",)),
    SkillEntry("Dart", CAT_LANGUAGE, ("dart",)),

    # ---------------- AI / 机器学习 ----------------
    SkillEntry("PyTorch", CAT_AI, ("pytorch", "torch"), parents=("深度学习",)),
    SkillEntry("TensorFlow", CAT_AI, ("tensorflow", "tf2", "tf框架"), parents=("深度学习",)),
    SkillEntry("Keras", CAT_AI, ("keras",), parents=("深度学习",)),
    SkillEntry("JAX", CAT_AI, ("jax",), parents=("深度学习",)),
    SkillEntry("scikit-learn", CAT_AI, ("scikit-learn", "sklearn", "scikit learn")),
    SkillEntry("XGBoost", CAT_AI, ("xgboost", "xgb")),
    SkillEntry("LightGBM", CAT_AI, ("lightgbm", "lgbm")),
    SkillEntry("CatBoost", CAT_AI, ("catboost",)),
    SkillEntry("ONNX", CAT_AI, ("onnx", "onnxruntime")),
    SkillEntry("OpenCV", CAT_AI, ("opencv", "cv2")),
    SkillEntry("深度学习", CAT_AI, ("深度学习", "神经网络", "deep learning", "cnn", "rnn", "lstm", "transformer架构")),
    SkillEntry("机器学习", CAT_AI, ("机器学习", "machine learning", "ml算法", "传统机器学习")),
    SkillEntry("计算机视觉", CAT_AI, ("计算机视觉", "computer vision", "图像识别", "目标检测", "图像分类", "yolo")),
    SkillEntry("自然语言处理", CAT_AI, ("自然语言处理", "nlp", "文本分类", "命名实体识别", "ner", "分词", "语义理解")),
    SkillEntry("语音处理", CAT_AI, ("语音识别", "语音合成", "asr", "tts", "语音信号处理", "声学模型")),
    SkillEntry("推荐系统", CAT_AI, ("推荐系统", "推荐算法", "排序模型", "ctr预估", "协同过滤", "召回策略")),
    SkillEntry("知识图谱", CAT_AI, ("知识图谱", "knowledge graph", "neo4j", "图数据库", "实体链接")),
    SkillEntry("强化学习", CAT_AI, ("强化学习", "reinforcement learning", "rlhf")),
    SkillEntry("多模态", CAT_AI, ("多模态", "multimodal", "图文理解", "视觉语言模型", "vlm")),

    # ---------------- 大模型应用（本项目的重点方向） ----------------
    SkillEntry("LLM", CAT_LLM, ("llm", "大语言模型", "大模型", "large language model", "语言模型")),
    SkillEntry("RAG", CAT_LLM, ("rag", "检索增强", "检索增强生成", "retrieval augmented generation")),
    SkillEntry("Prompt Engineering", CAT_LLM, ("prompt engineering", "提示词工程", "提示工程", "prompt优化", "提示词优化", "prompt 设计")),
    SkillEntry("Function Calling", CAT_LLM, ("function calling", "tool use", "工具调用", "函数调用", "tools调用")),
    SkillEntry("Agent", CAT_LLM, ("agent", "智能体", "ai agent", "multi-agent", "多智能体")),
    SkillEntry("微调", CAT_LLM, ("微调", "fine-tune", "finetune", "fine tuning", "指令微调", "sft")),
    SkillEntry("LoRA", CAT_LLM, ("lora", "qlora", "peft", "低秩适配"), parents=("微调",)),
    SkillEntry("Embedding", CAT_LLM, ("embedding", "向量化", "文本向量", "词向量")),
    SkillEntry("Hugging Face", CAT_LLM, ("hugging face", "huggingface", "transformers库", "transformers", "hf hub", "hf 生态")),
    SkillEntry("vLLM", CAT_LLM, ("vllm", "tgi", "推理加速", "推理部署框架")),
    SkillEntry("Ollama", CAT_LLM, ("ollama", "llama.cpp", "本地推理")),
    SkillEntry("LangChain", CAT_LLM, ("langchain", "langgraph")),
    SkillEntry("LlamaIndex", CAT_LLM, ("llamaindex", "llama index")),
    SkillEntry("Dify", CAT_LLM, ("dify",)),
    SkillEntry("Coze", CAT_LLM, ("coze", "扣子")),
    SkillEntry("n8n", CAT_LLM, ("n8n",)),
    SkillEntry("模型部署", CAT_LLM, ("模型部署", "模型服务化", "模型上线", "推理服务")),
    SkillEntry("AIGC 应用", CAT_LLM, ("aigc", "生成式ai", "生成式人工智能", "文生图", "text to image", "ai 应用", "ai应用")),

    # ---------------- 向量检索 ----------------
    SkillEntry("FAISS", CAT_VECTOR_DB, ("faiss",), parents=("向量数据库",)),
    SkillEntry("Milvus", CAT_VECTOR_DB, ("milvus",), parents=("向量数据库",)),
    SkillEntry("Chroma", CAT_VECTOR_DB, ("chroma", "chromadb"), parents=("向量数据库",)),
    SkillEntry("Qdrant", CAT_VECTOR_DB, ("qdrant",), parents=("向量数据库",)),
    SkillEntry("Weaviate", CAT_VECTOR_DB, ("weaviate",), parents=("向量数据库",)),
    SkillEntry("pgvector", CAT_VECTOR_DB, ("pgvector",), parents=("向量数据库",)),
    SkillEntry("Elasticsearch", CAT_VECTOR_DB, ("elasticsearch", "es集群", "elastic search", "opensearch")),
    SkillEntry("向量数据库", CAT_VECTOR_DB, ("向量数据库", "向量库", "vector database", "向量检索")),
    SkillEntry("BM25", CAT_VECTOR_DB, ("bm25", "全文检索", "倒排索引", "关键词检索")),

    # ---------------- 后端开发 ----------------
    SkillEntry("FastAPI", CAT_BACKEND, ("fastapi",)),
    SkillEntry("Flask", CAT_BACKEND, ("flask",)),
    SkillEntry("Django", CAT_BACKEND, ("django", "drf")),
    SkillEntry("Tornado", CAT_BACKEND, ("tornado",)),
    SkillEntry("Spring Boot", CAT_BACKEND, ("spring boot", "springboot", "spring cloud", "springcloud")),
    SkillEntry("Node.js", CAT_BACKEND, ("node.js", "nodejs", "node")),
    SkillEntry("Express", CAT_BACKEND, ("express", "express.js")),
    SkillEntry("Nest.js", CAT_BACKEND, ("nest.js", "nestjs")),
    SkillEntry("gRPC", CAT_BACKEND, ("grpc", "protobuf", "protocol buffers")),
    SkillEntry("RESTful API", CAT_BACKEND, ("restful", "rest api", "rest接口", "http接口", "接口开发")),
    SkillEntry("GraphQL", CAT_BACKEND, ("graphql",)),
    SkillEntry("WebSocket", CAT_BACKEND, ("websocket", "长连接")),
    SkillEntry("SSE", CAT_BACKEND, ("sse", "server-sent event", "流式输出", "流式返回", "流式响应")),
    SkillEntry("并发编程", CAT_BACKEND, ("并发编程", "多线程", "多进程", "asyncio", "协程", "异步编程")),
    SkillEntry("微服务", CAT_BACKEND, ("微服务", "microservice", "服务拆分", "分布式系统")),
    SkillEntry("消息队列", CAT_BACKEND, ("消息队列", "rabbitmq", "rocketmq", "activemq", "消息中间件")),

    # ---------------- 前端开发 ----------------
    SkillEntry("React", CAT_FRONTEND, ("react", "react.js", "reactjs", "hooks")),
    SkillEntry("Vue", CAT_FRONTEND, ("vue", "vue.js", "vue3", "vue2")),
    SkillEntry("Angular", CAT_FRONTEND, ("angular",)),
    SkillEntry("Next.js", CAT_FRONTEND, ("next.js", "nextjs")),
    SkillEntry("HTML", CAT_FRONTEND, ("html", "html5")),
    SkillEntry("CSS", CAT_FRONTEND, ("css", "css3", "sass", "scss", "less")),
    SkillEntry("Tailwind CSS", CAT_FRONTEND, ("tailwind", "tailwindcss")),
    SkillEntry("Webpack", CAT_FRONTEND, ("webpack",)),
    SkillEntry("Vite", CAT_FRONTEND, ("vite",)),
    SkillEntry("前端工程化", CAT_FRONTEND, ("前端工程化", "前端性能优化", "构建工具", "组件化开发")),

    # ---------------- 数据库 ----------------
    SkillEntry("MySQL", CAT_DATABASE, ("mysql",)),
    SkillEntry("PostgreSQL", CAT_DATABASE, ("postgresql", "postgres", "pgsql")),
    SkillEntry("Redis", CAT_DATABASE, ("redis",)),
    SkillEntry("MongoDB", CAT_DATABASE, ("mongodb", "mongo")),
    SkillEntry("SQLite", CAT_DATABASE, ("sqlite",)),
    SkillEntry("Oracle", CAT_DATABASE, ("oracle", "oracle数据库")),
    SkillEntry("ClickHouse", CAT_DATABASE, ("clickhouse", "ck数据库")),
    SkillEntry("HBase", CAT_DATABASE, ("hbase",)),
    SkillEntry("数据库设计", CAT_DATABASE, ("数据库设计", "表结构设计", "sql优化", "索引优化", "慢查询")),

    # ---------------- 云原生与运维 ----------------
    SkillEntry("Docker", CAT_CLOUD, ("docker", "容器化", "dockerfile")),
    SkillEntry("Kubernetes", CAT_CLOUD, ("kubernetes", "k8s", "k8s集群")),
    SkillEntry("CI/CD", CAT_CLOUD, ("ci/cd", "cicd", "持续集成", "持续交付", "持续部署")),
    SkillEntry("GitHub Actions", CAT_CLOUD, ("github actions", "gh actions")),
    SkillEntry("Jenkins", CAT_CLOUD, ("jenkins",)),
    SkillEntry("Linux", CAT_CLOUD, ("linux", "ubuntu", "centos", "unix", "linux系统")),
    SkillEntry("Nginx", CAT_CLOUD, ("nginx", "反向代理")),
    SkillEntry("AWS", CAT_CLOUD, ("aws", "ec2", "s3存储", "amazon web services")),
    SkillEntry("阿里云", CAT_CLOUD, ("阿里云", "aliyun", "ali cloud", "阿里云服务")),
    SkillEntry("腾讯云", CAT_CLOUD, ("腾讯云", "tencent cloud")),
    SkillEntry("华为云", CAT_CLOUD, ("华为云", "huawei cloud")),
    SkillEntry("Serverless", CAT_CLOUD, ("serverless", "云函数", "无服务器")),
    SkillEntry("监控告警", CAT_CLOUD, ("监控告警", "prometheus", "grafana", "链路追踪", "日志采集")),

    # ---------------- 移动端 ----------------
    SkillEntry("HarmonyOS", CAT_MOBILE, ("harmonyos", "鸿蒙", "鸿蒙系统", "harmony os", "openharmony")),
    SkillEntry("ArkTS", CAT_MOBILE, ("arkts", "arkui", "ark ts")),
    SkillEntry("Android", CAT_MOBILE, ("android", "安卓")),
    SkillEntry("iOS", CAT_MOBILE, ("ios开发", "swiftui", "ios 开发")),
    SkillEntry("Flutter", CAT_MOBILE, ("flutter",)),
    SkillEntry("React Native", CAT_MOBILE, ("react native", "rn开发")),
    SkillEntry("小程序", CAT_MOBILE, ("小程序", "微信小程序", "miniprogram")),

    # ---------------- 数据处理 ----------------
    SkillEntry("Pandas", CAT_DATA, ("pandas",)),
    SkillEntry("NumPy", CAT_DATA, ("numpy",)),
    SkillEntry("Spark", CAT_DATA, ("spark", "pyspark", "sparksql")),
    SkillEntry("Flink", CAT_DATA, ("flink", "实时计算")),
    SkillEntry("Hadoop", CAT_DATA, ("hadoop", "hdfs", "mapreduce")),
    SkillEntry("Kafka", CAT_DATA, ("kafka",)),
    SkillEntry("ETL", CAT_DATA, ("etl", "数据清洗", "数据抽取", "数据管道", "data pipeline")),
    SkillEntry("数据仓库", CAT_DATA, ("数据仓库", "数仓", "data warehouse", "维度建模", "hive")),
    SkillEntry("数据分析", CAT_DATA, ("数据分析", "数据挖掘", "数据可视化", "bi报表", "指标体系")),

    # ---------------- 开发工具 ----------------
    # ⚠️ 这里**不能**把 "github" 写进 Git 的别名。
    # 踩过的坑：Git 与 GitHub 是两条独立的词条，如果两条都收录 "github"，
    # 那么按"长别名优先 + 区间占用"的匹配规则，先注册的 Git 会抢走
    # "GitHub" 这段文本的归属权，导致简历里明明写了 GitHub 却被识别成 Git
    # ——技能清单里少一项，匹配报告里也会误报"未覆盖开源经历"。
    # 现在 SkillNormalizer 会在初始化时对重复别名发出告警，防止再次踩到。
    SkillEntry("Git", CAT_TOOL, ("git", "gitlab", "gitee", "版本控制", "git 分支")),
    SkillEntry("Postman", CAT_TOOL, ("postman", "apifox", "接口测试工具")),
    SkillEntry("Jupyter", CAT_TOOL, ("jupyter", "notebook", "colab")),
    SkillEntry("PyCharm", CAT_TOOL, ("pycharm",)),
    SkillEntry("VS Code", CAT_TOOL, ("vs code", "vscode", "visual studio code")),
    SkillEntry("DevEco Studio", CAT_TOOL, ("deveco", "deveco studio")),
    SkillEntry("GitHub", CAT_TOOL, ("github", "github 开源", "开源项目", "开源贡献", "star 数")),

    # ---------------- 工程方法 ----------------
    SkillEntry("单元测试", CAT_METHOD, ("单元测试", "pytest", "unittest", "测试用例", "自动化测试")),
    SkillEntry("代码审查", CAT_METHOD, ("代码审查", "code review", "codeview")),
    SkillEntry("敏捷开发", CAT_METHOD, ("敏捷开发", "scrum", "agile", "迭代开发")),
    SkillEntry("TDD", CAT_METHOD, ("tdd", "测试驱动开发")),
    SkillEntry("MLOps", CAT_METHOD, ("mlops", "模型运维", "模型全生命周期")),
    SkillEntry("DevOps", CAT_METHOD, ("devops",)),
    SkillEntry("A/B 测试", CAT_METHOD, ("a/b测试", "ab测试", "abtest", "灰度发布")),
    SkillEntry("性能优化", CAT_METHOD, ("性能优化", "性能调优", "压测", "qps优化", "降低延迟")),
    SkillEntry("架构设计", CAT_METHOD, ("架构设计", "系统设计", "技术方案设计", "方案选型")),
    SkillEntry("技术文档", CAT_METHOD, ("技术文档", "文档编写", "需求文档", "设计文档")),
    SkillEntry("英文文献阅读", CAT_METHOD, ("英文文献", "英文文档", "英语读写", "cet-6", "cet6", "六级")),

    # ---------------- 领域方向 ----------------
    SkillEntry("金融科技", CAT_DOMAIN, ("金融科技", "fintech", "量化交易", "风控模型", "支付系统")),
    SkillEntry("电商", CAT_DOMAIN, ("电商", "电商平台", "交易系统", "订单系统", "供应链")),
    # 注意："教育" 单字词不能作为别名。简历里 "教育背景" 是章节标题，
    # 一旦收录裸的 "教育"，简历技能清单里就会莫名其妙多出一项"教育"。
    SkillEntry("在线教育", CAT_DOMAIN, ("在线教育", "教育科技", "智慧教育", "学习平台", "教育行业", "k12")),
    SkillEntry("医疗健康", CAT_DOMAIN, ("医疗健康", "智慧医疗", "医疗影像", "辅助诊断", "healthcare")),
    SkillEntry("智能客服", CAT_DOMAIN, ("智能客服", "对话系统", "问答机器人", "客服机器人")),
    SkillEntry("自动驾驶", CAT_DOMAIN, ("自动驾驶", "智能驾驶", "车联网", "感知算法")),
    SkillEntry("智能硬件", CAT_DOMAIN, ("智能硬件", "物联网", "iot", "嵌入式", "边缘计算")),
    SkillEntry("游戏开发", CAT_DOMAIN, ("游戏开发", "游戏引擎", "unity", "ue4", "ue5", "unreal")),
    SkillEntry("企业服务", CAT_DOMAIN, ("企业服务", "saas", "to b", "tob", "b端产品")),
)


# ---------------------------------------------------------------------------
# 掌握程度 / 需求强度词表
# ---------------------------------------------------------------------------

#: 简历侧：程度词 -> 0~5 的掌握度分值。
#: 用于回答"这个人说'了解 Python'和'精通 Python'，是不是一回事"。
LEVEL_WORDS: dict[str, int] = {
    "精通": 5,
    "专家": 5,
    "资深": 5,
    "深入理解": 5,
    "底层原理": 5,
    "熟练": 4,
    "熟悉": 4,
    "掌握": 4,
    "扎实": 4,
    "擅长": 4,
    "会用": 3,
    "能够使用": 3,
    "使用过": 3,
    "实践过": 3,
    "了解": 2,
    "基本了解": 2,
    "有所了解": 2,
    "初步": 2,
    "入门": 2,
    "接触过": 2,
    "听说过": 1,
    "正在学习": 2,
    "学习中": 2,
}

#: JD 侧：需求强度词 -> "must"（硬性）/ "nice"（加分）
#:
#: 这份词表是按真实招聘语言的使用习惯定的：
#: - 「精通 / 熟练掌握 / 熟悉 / 掌握」在 JD 里基本都是**硬性**门槛，
#:   不会有人写"熟悉 Python（可不会）"；
#: - 「了解 / 加分 / 优先 / 更佳」才是真正的"有则更好"。
#: 把「熟悉」错判成 nice 会让匹配分虚高，这是很常见的坑。
DEMAND_WORDS: dict[str, str] = {
    # ---- 硬性 ----
    "必须": "must",
    "必需": "must",
    "要求": "must",
    "精通": "must",
    "熟练": "must",
    "扎实": "must",
    "深入": "must",
    "掌握": "must",
    "熟悉": "must",
    "具备": "must",
    "负责": "must",
    # ---- 加分 ----
    "加分": "nice",
    "优先": "nice",
    "更佳": "nice",
    "了解": "nice",
    "有所了解": "nice",
    "有经验": "nice",
    "锦上添花": "nice",
}

# ---------------------------------------------------------------------------
# 强度词分层
# ---------------------------------------------------------------------------
# 为什么"硬性词"还要再分两层？
#
#   真实 JD 里有两类句子长得几乎一样，但意思完全不同：
#     A. 「熟悉 Kubernetes 者优先」   -> 加分项，不熟悉也能投
#     B. 「必须熟悉 Python」          -> 硬门槛，不熟直接没戏
#   两者都同时包含 must 词（熟悉）与 nice 词（优先）。
#   如果按"must 一律压过 nice"处理，A 会被误判成硬要求 ——
#   结果是匹配分虚低、系统劝退本可以一试的岗位。
#
# 所以这里把标记词拆成四档，按"表达强度 + 具体性"排序：
#   1. 强硬性词（必须/必需/要求/精通/深入/扎实）：明确的门槛用语，最强；
#   2. 显式加分词（加分/优先/更佳）：**对整句的限定**，"熟悉 X 者优先"
#      里的"熟悉"只是搭配动词，真正的语义由"优先"决定；
#   3. 普通硬性词（熟悉/掌握/熟练/具备/负责）：JD 的常规要求动词；
#   4. 普通加分词（了解/有所了解/有经验/锦上添花）：常规的降级用语。
#
# 判定时按 1 > 2 > 3 > 4 取第一个命中的那一档。

#: 第一档：明确的"这是硬门槛"用语
STRONG_MUST_WORDS: tuple[str, ...] = ("必须", "必需", "要求", "精通", "深入", "扎实")

#: 第二档：对整句做"降级"限定的用语（"者优先""加分项"）
EXPLICIT_NICE_WORDS: tuple[str, ...] = ("加分", "优先", "更佳")

#: 第三档：JD 的常规硬性动词
SOFT_MUST_WORDS: tuple[str, ...] = ("熟悉", "掌握", "熟练", "具备", "负责")

#: 第四档：常规降级用语
GENERAL_NICE_WORDS: tuple[str, ...] = ("有所了解", "了解", "有经验", "锦上添花")

#: 从 DEMAND_WORDS 派生的两组标记词，供解析器做行级强度判定。
#: ``MUST_MARKERS`` / ``NICE_MARKERS`` 是"是否出现过某类词"的粗筛集合，
#: 真正的强弱判定在 ``parser.judge_demand`` 里用上面四档做。
MUST_MARKERS: tuple[str, ...] = tuple(w for w, kind in DEMAND_WORDS.items() if kind == "must")
NICE_MARKERS: tuple[str, ...] = tuple(w for w, kind in DEMAND_WORDS.items() if kind == "nice")

#: 一致性自检：四档词表必须与 DEMAND_WORDS 完全对齐。
#: 漏一个词会让"粗筛"和"细判"对不上（典型症状：某技能被扫出来了，
#: 但强度判定永远落不到正确的那一档）。这里用断言把口子焊死。
_TIERED_MUST = STRONG_MUST_WORDS + SOFT_MUST_WORDS
_TIERED_NICE = EXPLICIT_NICE_WORDS + GENERAL_NICE_WORDS
assert set(_TIERED_MUST) == set(MUST_MARKERS), (
    f"硬性词分层与 MUST_MARKERS 不一致：{set(MUST_MARKERS) ^ set(_TIERED_MUST)}"
)
assert set(_TIERED_NICE) == set(NICE_MARKERS), (
    f"加分词分层与 NICE_MARKERS 不一致：{set(NICE_MARKERS) ^ set(_TIERED_NICE)}"
)

#: 学历等级映射：数字越大越高。用于做"学历是否达标"的比较。
EDUCATION_LEVELS: dict[str, int] = {
    "不限": 0,
    "学历不限": 0,
    "大专": 1,
    "专科": 1,
    "高职": 1,
    "本科": 2,
    "学士": 2,
    "统招本科": 2,
    "硕士": 3,
    "研究生": 3,
    "硕士及以上": 3,
    "master": 3,
    "博士": 4,
    "phd": 4,
    "doctor": 4,
}


# ---------------------------------------------------------------------------
# 别名编译
# ---------------------------------------------------------------------------

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

# 英文别名的左右边界。
#
# 为什么不用 \b？因为 \b 判定的是"单词字符/非单词字符"的切换，而技能名里
# 大量出现 + # . 这类非单词字符（C++ / C# / .NET / Node.js），\b 在它们
# 周围的行为完全不符合直觉，会出现 "C++ 里匹配不到 C++" 这种怪事。
#
# 所以这里自定义边界：
#   左侧：不允许前面紧贴字母、数字、下划线、+、#、.
#         —— 拦住 "Django" 里的 "go"、"notjavascript" 里的 "javascript"
#   右侧：不允许后面紧贴字母、下划线、+、#、.
#         —— 拦住 "JavaScript" 里的 "Java"、"Google" 里的 "Go"
#         注意**故意放开了数字**，否则 "Python3"、"Java8"、"C++11" 都会漏报
_LEFT_BOUND = r"(?<![A-Za-z0-9_+#.])"
_RIGHT_BOUND = r"(?![A-Za-z_+#.])"


@dataclass(frozen=True)
class _CompiledAlias:
    """一条编译好的别名规则。"""

    pattern: re.Pattern[str]
    canonical: str
    category: str
    alias: str
    length: int
    is_cjk: bool


def _build_alias_pattern(alias: str) -> re.Pattern[str]:
    """把一条别名字符串编译成带边界的正则。

    中文别名与英文别名的边界策略不同：
    - 中文没有词间空格，加 \\b 毫无意义，直接子串匹配；
    - 英文/符号别名用上面自定义的边界。
    """
    escaped = re.escape(alias)
    if _CJK_RE.search(alias):
        return re.compile(escaped, re.IGNORECASE)
    return re.compile(_LEFT_BOUND + escaped + _RIGHT_BOUND, re.IGNORECASE)


# ---------------------------------------------------------------------------
# 扫描结果
# ---------------------------------------------------------------------------


@dataclass
class SkillMention:
    """文本中一次技能命中。

    Attributes:
        canonical: 归一化后的技能名。
        raw:       原文里实际出现的写法（如 "js"），用于向用户展示"你写的是这个"。
        start/end: 在原文中的字符区间，前端据此高亮。
        category:  技能大类。
        source:    命中来源（"dict" 表示词典命中，后续 LLM 抽取会标 "llm"）。
    """

    canonical: str
    raw: str
    start: int
    end: int
    category: str
    source: str = "dict"

    def to_dict(self) -> dict[str, object]:
        return {
            "canonical": self.canonical,
            "raw": self.raw,
            "start": self.start,
            "end": self.end,
            "category": self.category,
            "source": self.source,
        }


class SkillNormalizer:
    """技能归一化器：词典加载 + 别名索引 + 全文扫描。

    用法::

        nz = SkillNormalizer()
        nz.canonical_of("js")        # -> "JavaScript"
        nz.scan("熟悉 Python 与 js")  # -> [SkillMention("Python", ...), SkillMention("JavaScript", ...)]
    """

    def __init__(self, entries: Sequence[SkillEntry] = SKILL_DB) -> None:
        self._alias_to_canonical: dict[str, str] = {}
        self._categories: dict[str, str] = {}
        self._aliases_by_canonical: dict[str, list[str]] = {}
        self._parents: dict[str, tuple[str, ...]] = {}
        compiled: list[_CompiledAlias] = []

        seen: set[str] = set()
        for entry in entries:
            if entry.canonical in seen:
                # 词典里重复定义同一个 canonical 是维护事故，直接报错而不是静默覆盖
                raise ValueError(f"技能词典中存在重复的 canonical: {entry.canonical}")
            seen.add(entry.canonical)
            self._categories[entry.canonical] = entry.category
            if entry.parents:
                self._parents[entry.canonical] = entry.parents

            alias_list: list[str] = []
            for alias in entry.aliases:
                key = alias.strip().lower()
                if not key:
                    continue
                alias_list.append(key)
                self._register_alias(key, entry.canonical, alias=alias.strip())
                compiled.append(
                    _CompiledAlias(
                        pattern=_build_alias_pattern(alias.strip()),
                        canonical=entry.canonical,
                        category=entry.category,
                        alias=alias.strip(),
                        length=len(alias.strip()),
                        is_cjk=bool(_CJK_RE.search(alias)),
                    )
                )

            # canonical 自身也是别名（用户可能直接写标准名）。
            # 但 auto_alias=False 的词条跳过这一步 —— 见 SkillEntry.auto_alias 的说明。
            canon_key = entry.canonical.strip().lower()
            if entry.auto_alias and canon_key not in alias_list:
                alias_list.append(canon_key)
                self._register_alias(canon_key, entry.canonical, alias=entry.canonical, setdefault=True)
                compiled.append(
                    _CompiledAlias(
                        pattern=_build_alias_pattern(entry.canonical),
                        canonical=entry.canonical,
                        category=entry.category,
                        alias=entry.canonical,
                        length=len(entry.canonical),
                        is_cjk=bool(_CJK_RE.search(entry.canonical)),
                    )
                )
            self._aliases_by_canonical[entry.canonical] = alias_list

        # **长别名优先**：先匹配 "javascript" 再匹配 "java"，否则短别名会抢先占掉位置。
        # 这是本模块最关键的一个排序 —— 去掉它，"JavaScript" 会被识别成 "Java"。
        compiled.sort(key=lambda item: item.length, reverse=True)
        self._compiled: tuple[_CompiledAlias, ...] = tuple(compiled)
        self._max_alias_len = max((c.length for c in compiled), default=1)

    def _register_alias(self, key: str, canonical: str, *, alias: str, setdefault: bool = False) -> None:
        """登记一个别名，并在发现冲突时告警。

        为什么要在登记处做冲突检测？因为冲突的后果非常隐蔽：
        两条词条共用同一个别名时，最终只有"排序靠前"的那条能匹配到文本，
        另一条形同虚设 —— 而且**不会有任何报错**，只会在匹配报告里
        表现为"某项技能莫名其妙没被识别出来"。

        这个告警就是为了把这类问题从"跑出来才发现"提前到"启动即暴露"。
        """
        existing = self._alias_to_canonical.get(key)
        if existing is not None and existing != canonical:
            logger.warning(
                "技能别名冲突：%r 同时指向 %r 与 %r。由于长别名优先匹配，"
                "只有排序靠前的一条能生效，请检查是否应合并词条或改掉重复别名。",
                alias,
                existing,
                canonical,
            )
        if setdefault:
            self._alias_to_canonical.setdefault(key, canonical)
        else:
            self._alias_to_canonical[key] = canonical

    # -- 查询 ---------------------------------------------------------------

    @property
    def canonical_names(self) -> list[str]:
        """词典中所有标准技能名（按字典序）。"""
        return sorted(self._categories)

    def is_known(self, name: str) -> bool:
        """判断一个写法是否在词典里。"""
        return name.strip().lower() in self._alias_to_canonical

    def canonical_of(self, name: str) -> str | None:
        """把任意写法归一化；词典里没有则返回 None。"""
        return self._alias_to_canonical.get(name.strip().lower())

    def category_of(self, canonical: str) -> str:
        """取标准技能名所属大类；未知技能归为「其它」。"""
        return self._categories.get(canonical, "其它")

    def aliases_of(self, canonical: str) -> list[str]:
        """取某标准技能的全部已知写法。"""
        return list(self._aliases_by_canonical.get(canonical, []))

    def parents_of(self, canonical: str) -> tuple[str, ...]:
        """取某技能的上位技能。

        例：``parents_of("FAISS")`` → ``("向量数据库",)``

        用途：JD 写「了解向量数据库（FAISS / Milvus）」，简历只写了 FAISS。
        按字面匹配会判成"向量数据库缺失"，但常识上会用 FAISS 就是会用向量数据库。
        有了这层关系，匹配时可以把子技能的掌握**向上继承**一级。
        """
        return self._parents.get(canonical, ())

    def expand_with_parents(self, names: Iterable[str]) -> list[str]:
        """把技能列表扩展一级上位技能（保持原顺序，父技能追加在其子技能之后）。

        只扩展**一级**，不做传递闭包 —— 词典很小，两级以上的继承关系
        容易变成"会 Python 就等于会一切"这种荒谬结论。
        """
        out: list[str] = []
        seen: set[str] = set()
        for name in names:
            if name not in seen:
                seen.add(name)
                out.append(name)
            for parent in self.parents_of(name):
                if parent not in seen:
                    seen.add(parent)
                    out.append(parent)
        return out

    def expand_skill_set(self, names: Iterable[str]) -> set[str]:
        """把技能集合扩展出上位技能，返回集合（便于做交并运算）。"""
        return set(self.expand_with_parents(names))

    def normalize(self, name: str) -> str | None:
        """归一化单个名字（``canonical_of`` 的语义化别名）。"""
        return self.canonical_of(name)

    def normalize_list(self, names: Iterable[str]) -> list[str]:
        """批量归一化并去重（保持首次出现的顺序）。

        词典里查不到的技能**不会被丢弃**，而是原样保留 —— 真实 JD 里
        总有词典覆盖不到的新技术，静默丢掉会让匹配分虚高。
        调用方可以用 ``is_known`` 区分它们。
        """
        out: list[str] = []
        seen: set[str] = set()
        for raw in names:
            if not raw or not raw.strip():
                continue
            canon = self.canonical_of(raw) or raw.strip()
            if canon not in seen:
                seen.add(canon)
                out.append(canon)
        return out

    # -- 扫描 ---------------------------------------------------------------

    def scan(self, text: str, *, limit: int | None = None) -> list[SkillMention]:
        """在文本里扫描全部技能命中（带字符区间）。

        实现要点：**区间占用**。按"别名长度从长到短"的顺序匹配，
        一旦某个区间被占用就不再接受新命中。这样处理的直接效果是
        ``JavaScript`` 只会产出一条命中，而不会额外产出一条 ``Java``。

        Args:
            text: 待扫描文本。
            limit: 最多返回多少条（按位置排序后截断），None 表示不限。

        Returns:
            按 ``start`` 升序排列的命中列表。
        """
        if not text:
            return []

        occupied = bytearray(len(text))
        hits: list[SkillMention] = []

        for item in self._compiled:
            for m in item.pattern.finditer(text):
                s, e = m.start(), m.end()
                if e <= s:
                    continue
                # 任一位被占用就跳过这条命中（长别名已优先占位）
                if any(occupied[s:e]):
                    continue
                for i in range(s, e):
                    occupied[i] = 1
                hits.append(
                    SkillMention(
                        canonical=item.canonical,
                        raw=text[s:e],
                        start=s,
                        end=e,
                        category=item.category,
                    )
                )

        hits.sort(key=lambda h: (h.start, -(h.end - h.start)))
        if limit is not None:
            hits = hits[:limit]
        return hits

    def scan_canonicals(self, text: str) -> list[str]:
        """扫描并只返回去重后的标准名（按首次出现位置排序）。"""
        out: list[str] = []
        seen: set[str] = set()
        for hit in self.scan(text):
            if hit.canonical not in seen:
                seen.add(hit.canonical)
                out.append(hit.canonical)
        return out

    # -- 分级 ---------------------------------------------------------------

    @staticmethod
    def detect_level(text: str, start: int, end: int, *, window: int = 14) -> tuple[str | None, int | None]:
        """判断某个技能命中点附近的**掌握程度**。

        做法：取命中位置前后 ``window`` 个字符作为窗口（"精通 Python" 的
        程度词在前，"Python（熟练）"的程度词在后，所以两侧都要看），
        在窗口内找程度词。若窗口内出现多个程度词，**取距离命中点最近的那个**，
        因为「熟悉 Java，精通 Python」里 Python 的程度应该由更近的"精通"决定。

        Returns:
            ``(程度词, 分值)``；窗口内没有程度词时返回 ``(None, None)``。
        """
        if not text:
            return None, None

        lo = max(0, start - window)
        hi = min(len(text), end + window)
        window_text = text[lo:hi]
        if not window_text:
            return None, None

        center = (start + end) / 2
        candidates: list[tuple[float, int, str]] = []

        # 只在"连续汉字游程"内部找程度词。为什么要先取游程而不是直接全文找？
        # 因为程度词本身全是汉字，直接全文 find 会把「熟悉业务，了解金融」
        # 里的「了解」也算到附近的技能头上，而它其实修饰的是别的东西。
        for m in re.finditer(r"[\u4e00-\u9fff]+", window_text):
            run_start, run_end = m.start(), m.end()
            for lvl_word in LEVEL_WORDS:
                idx = window_text.find(lvl_word, run_start, run_end)
                if idx < 0:
                    continue
                abs_center = lo + idx + len(lvl_word) / 2
                dist = abs(abs_center - center)
                # 排序键：先比距离（越近越可靠），距离相同则比长度（越长越具体）
                candidates.append((dist, -len(lvl_word), lvl_word))

        if not candidates:
            return None, None

        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        best_word = candidates[0][2]
        return best_word, LEVEL_WORDS[best_word]


#: 全局默认归一化器。词典是只读数据，建一次即可复用。
DEFAULT_NORMALIZER = SkillNormalizer()


def normalize_skills(names: Iterable[str]) -> list[str]:
    """便捷函数：用默认词典做批量归一化。"""
    return DEFAULT_NORMALIZER.normalize_list(names)


def scan_skills(text: str) -> list[SkillMention]:
    """便捷函数：用默认词典扫描文本。"""
    return DEFAULT_NORMALIZER.scan(text)


def group_by_category(names: Iterable[str], normalizer: SkillNormalizer = DEFAULT_NORMALIZER) -> dict[str, list[str]]:
    """把技能按大类分组，供前端做「技能栈结构」展示。"""
    grouped: dict[str, list[str]] = {}
    for name in names:
        cat = normalizer.category_of(name)
        grouped.setdefault(cat, []).append(name)
    return grouped


__all__ = [
    "SkillEntry",
    "SkillMention",
    "SkillNormalizer",
    "SKILL_DB",
    "LEVEL_WORDS",
    "DEMAND_WORDS",
    "MUST_MARKERS",
    "NICE_MARKERS",
    "STRONG_MUST_WORDS",
    "EXPLICIT_NICE_WORDS",
    "SOFT_MUST_WORDS",
    "GENERAL_NICE_WORDS",
    "EDUCATION_LEVELS",
    "DEFAULT_NORMALIZER",
    "normalize_skills",
    "scan_skills",
    "group_by_category",
    "CAT_LANGUAGE",
    "CAT_AI",
    "CAT_LLM",
    "CAT_VECTOR_DB",
    "CAT_BACKEND",
    "CAT_FRONTEND",
    "CAT_DATABASE",
    "CAT_CLOUD",
    "CAT_MOBILE",
    "CAT_DATA",
    "CAT_TOOL",
    "CAT_METHOD",
    "CAT_DOMAIN",
]
