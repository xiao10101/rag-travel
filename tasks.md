# RAG Travel 企业级优化任务清单

## 阶段一：工程化基础（P0）

### 1. API 服务化

- [ ] 引入 FastAPI，暴露 RESTful API（`/chat`、`/upload`、`/health`）
- [ ] 异步化全部 IO 操作（embedding 调用、Milvus 查询、LLM 调用）
- [ ] 使用 Pydantic 做请求/响应模型校验
- [ ] 统一错误响应格式（错误码 + 错误信息）

### 2. 配置管理

- [ ] 引入 pydantic-settings 替代裸 `os.getenv`
- [ ] 分离 dev / staging / prod 多环境配置文件
- [ ] 敏感信息（API Key）走环境变量或 Secret Manager，禁止硬编码
- [ ] 配置项文档化（每个配置的含义、默认值、影响范围）

### 3. 结构化日志

- [ ] 引入 loguru 或 structlog
- [ ] 全链路统一 request_id 贯穿
- [ ] 日志分级：INFO（正常流）/ WARNING（降级）/ ERROR（故障）
- [ ] 日志格式统一为 JSON，便于日志平台采集

### 4. 容器化

- [ ] 编写 Dockerfile（非 root 用户运行）
- [ ] 编写 docker-compose.yml（Milvus + 应用 + Redis）
- [ ] 健康检查端点 `/health`
- [ ] 优雅关闭（graceful shutdown）

---

## 阶段二：检索质量提升（P0）

### 5. ReRank 引入

- [ ] 粗排（混合检索 Top 20-30）→ 精排（Cross-Encoder ReRank）→ 返回 Top 3-5
- [ ] 可选模型：bge-reranker-v2-m3、Cohere Rerank
- [ ] ReRank 模型本地部署或 API 调用，做好 fallback

### 6. Chunking 策略优化

- [ ] Semantic Chunking：基于 embedding 相似度断点切分，替代固定长度
- [ ] Small-to-Big / Sentence Window 检索：索引小粒度，返回大粒度上下文
- [ ] 多粒度索引：chunk 级 + 摘要级，分层检索
- [ ] chunk_size 和 overlap 参数化，支持不同文档类型不同策略

### 7. Query 理解增强

- [ ] 在 Query Rewrite 之外增加 HyDE（假设文档生成）
- [ ] Query 意图分类：事实查询 / 摘要 / 对比 / 多跳推理
- [ ] Query 分解：复杂问题拆成子问题，逐个子问题检索后合并

### 8. Metadata 过滤与多模态

- [ ] 文档级元数据：来源、日期、类别、权限标签
- [ ] Milvus 分区键 + 标量过滤（`subject` 字段已预留，需真正使用）
- [ ] 支持图片描述生成（多模态 Embedding），OCR 图片文字提取

---

## 阶段三：数据管道（P0）

### 9. 文档处理 Pipeline

- [ ] 多格式支持：Word、Excel、PPT、HTML、Markdown
- [ ] 表格提取 + 结构化保留（不要丢失表格语义）
- [ ] 图片 OCR + 多模态描述生成
- [ ] 文档解析器插件化，方便扩展新格式

### 10. 增量索引

- [ ] 文档哈希去重，避免重复入库
- [ ] 增量 upsert 而非全量重建（按文档粒度）
- [ ] Milvus Partition 按时间 / 来源 / 文档 ID 隔离
- [ ] 索引版本管理，支持回滚

### 11. Pipeline 编排

- [ ] 引入 Prefect / Temporal / Celery 做异步任务编排
- [ ] 完整 DAG：文档上传 → 解析 → 切分 → Embedding → 入库 → 索引更新
- [ ] 任务失败重试 + 告警
- [ ] 任务状态可查询（处理中 / 完成 / 失败）

---

## 阶段四：评估体系（P1）

### 12. RAG 评估框架

- [ ] 引入 RAGAS 或自建评估
- [ ] 评估指标：
  - Context Precision（检索精度）
  - Context Recall（检索召回）
  - Faithfulness（生成忠实度）
  - Answer Relevancy（回答相关性）
  - 端到端延迟（P50 / P95 / P99）
- [ ] 评估结果可视化（Dashboard）

### 13. 黄金数据集

- [ ] 构建 100+ Q&A 对作为 ground truth
- [ ] 配合 `subject` 分类做分场景评估
- [ ] 每次检索策略变更后自动回归测试
- [ ] 数据集持续更新机制

---

## 阶段五：可观测性与运维（P1）

### 14. Tracing 链路追踪

- [ ] 引入 Langfuse / Phoenix / OpenTelemetry
- [ ] 追踪完整链路：用户问题 → Query Rewrite → 检索 → RRF → ReRank → LLM 生成
- [ ] 可视化每次召回的 embedding 分数和 BM25 分数
- [ ] 支持按 trace_id 搜索单次请求全链路

### 15. 监控与告警

- [ ] 关键指标采集：
  - QPS
  - P95 延迟
  - Embedding API 错误率
  - LLM Token 用量 / 费用
- [ ] Milvus 健康监控（集合大小、索引状态、查询延迟）
- [ ] 成本监控（DashScope API 调用量 / 费用，按天汇总）
- [ ] 告警规则：错误率 > 阈值、延迟 > 阈值、费用 > 预算

### 16. 缓存层

- [ ] 精确问题缓存（Redis）：相同问题直接返回缓存结果
- [ ] 语义缓存（GPTCache）：相似问题命中缓存
- [ ] 缓存过期策略（按文档更新频率）
- [ ] Embedding 计算结果缓存（避免重复 embedding）

---

## 阶段六：安全与多租户（P1）

### 17. 安全加固

- [ ] API 认证：JWT 或 API Key 认证
- [ ] 速率限制（slowapi 或 Redis 令牌桶）
- [ ] 输入安全：
  - Prompt Injection 防护
  - 文本长度限制
  - SQL / NoSQL 注入防护
- [ ] Milvus 开启认证（用户名 / 密码）
- [ ] 敏感信息脱敏（日志中不输出 API Key、用户问题脱敏）

### 18. 多租户

- [ ] 知识库隔离（每个租户独立 Milvus Collection 或 Partition）
- [ ] 权限控制（RBAC：不同用户可见不同文档）
- [ ] 租户级配额管理（QPS、存储量、文档数）
- [ ] 租户级配置（自定义 chunk 策略、模型选择）

---

## 阶段七：高级能力（P2 - 按需引入）

### 19. 对话记忆与多轮对话

- [ ] 引入 LangGraph 做多轮对话状态管理
- [ ] 对话历史窗口 + 摘要压缩（长对话自动摘要）
- [ ] 上下文窗口管理（Token 计数 + 智能截断策略）
- [ ] 会话持久化（Redis / DB）

### 20. Agentic RAG

- [ ] 多跳推理：跨文档关联检索
- [ ] 工具调用：计算器、日期解析、外部 API
- [ ] 自反思路由：检索结果不够时自动改策略重新检索
- [ ] Self-RAG / Corrective RAG 模式
- [ ] LangGraph Agent + Tool 定义

### 21. 流式输出

- [ ] FastAPI StreamingResponse
- [ ] SSE（Server-Sent Events）逐 Token 推送
- [ ] 首 Token 延迟优化（TTFT）
- [ ] 支持流式 + 非流式切换

### 22. A/B 测试框架

- [ ] 不同检索策略 / 模型版本的流量分流
- [ ] 基于评估指标自动选择最优策略
- [ ] 实验配置管理（流量比例、生效时间）
- [ ] 实验结果统计与对比

---

## 架构演进目标态

```
┌──────────────────────────────────────────────────────┐
│                    Gateway Layer                      │
│   FastAPI + Auth + RateLimit + Request Validation     │
└────────────────────┬─────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────┐
│              Conversation Manager                     │
│   Memory | Context Window | Multi-turn               │
└────────────────────┬─────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────┐
│                 Query Processing                      │
│   Rewrite → HyDE → Decompose → Intent Classify       │
└────────────────────┬─────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────┐
│             Hybrid Retrieval Pipeline                 │
│   BM25 + Dense Vector → RRF → ReRank → Filter        │
│          (Milvus)    (BM25 Index)                     │
└────────────────────┬─────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────┐
│               Generation + Citation                   │
│   LLM (qwen-plus) + Post-processing + Guardrails     │
└──────────────────────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────┐
│              Observability Layer                      │
│   Tracing | Metrics | Logs | Eval | Cost Tracking    │
└──────────────────────────────────────────────────────┘
```

---

## 优先级说明

| 优先级 | 含义                                     |
| ------ | ---------------------------------------- |
| P0     | 必须做，直接影响系统可用性和核心体验     |
| P1     | 应该做，不阻塞上线但影响长期稳定性和质量 |
| P2     | 可以做，按业务需求选择性引入             |

## 建议执行顺序

按投入产出比排序：

1. **FastAPI + 异步化 + Pydantic 配置** — 立即可做，工程化基础
2. **ReRank + Semantic Chunking** — 检索质量最大提升点
3. **日志 + Docker** — 可运维的底线
4. **评估数据集 + RAGAS** — 后续所有优化的度量基础
5. **Tracing（Langfuse）** — 排查问题的眼睛，半天即可集成
6. **增量索引 + 多格式支持** — 数据管道核心
7. **缓存层** — 降本增效关键
8. **安全加固** — 对外暴露前必须完成
9. **对话记忆** — 按产品需求引入
10. **Agentic RAG** — 仅在单一检索无法解决时引入
