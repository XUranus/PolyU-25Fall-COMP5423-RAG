---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-family: 'Noto Sans SC', 'PingFang SC', 'Microsoft YaHei', sans-serif;
    font-size: 22px;
  }
  section.lead {
    text-align: center;
  }
  section.lead h1 {
    font-size: 2.0em;
  }
  section.lead h2 {
    font-size: 1.2em;
    color: #555;
  }
  h1 { font-size: 1.5em; }
  h2 { font-size: 1.2em; color: #2c3e50; }
  h3 { font-size: 1.05em; }
  p, li { font-size: 0.92em; }
  table { font-size: 0.72em; margin: 0 auto; }
  code { font-size: 0.85em; }
  blockquote { font-size: 0.88em; }
  pre { font-size: 0.78em; }
---

<!-- _class: lead -->

# RAG42

## 基于检索增强生成的多跳问答系统

**PolyU COMP5423 · 25Fall 课程项目**

---

# 目录

1. 项目概述与问题定义
2. 系统架构总览
3. 检索模块 — 5 种检索器
4. 混合检索与重排序
5. 生成模块与工作流
6. Agentic 多跳推理流程
7. 评估框架与实验结果

---

# 1. 项目概述

## 问题：多跳问答 (Multi-hop QA)

> 给定一个需要**跨多个文档推理**的复杂问题，系统需检索相关信息并生成准确答案。

**示例问题：**
*“电影《绿皮书》的导演出生在哪个城市？”*

需要两步推理：
1. 《绿皮书》的导演是谁？ → Peter Farrelly
2. Peter Farrelly 出生在哪里？ → Cumberland, Rhode Island

## 数据集

- **HotpotQA** — 经典多跳问答基准
- 课程提供的子集 `izhx/COMP5423-25Fall-HQ-small`
- 验证集 300 条多跳问题

---

# 2. 系统架构总览

```
┌─────────────────────────────────────────────────────────┐
│                   Frontend (React + TS)                  │
│              ChatPanel │ ThinkingPanel │ Sidebar         │
└──────────────────────────┬──────────────────────────────┘
                           │ HTTP POST /api/chat/{id}/messages
┌──────────────────────────▼──────────────────────────────┐
│                  Backend (Flask + SQLite)                 │
│          Chat History │ Session Management                │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                    RAG Pipeline Core                      │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  Retriever   │→ │   Workflow   │→ │   Generator   │  │
│  │  (Hybrid)    │  │ (Agentic)    │  │ (Qwen/OpenAI) │  │
│  └──────────────┘  └──────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

# 3. 检索模块 — 5 种检索器

- **SparseRetriever**：经典 BM25 算法，基于词频-逆文档频率，擅长精确关键词匹配
- **DenseRetriever**：BAAI/bge-large-en-v1.5 编码为 1024 维向量，FAISS 近邻搜索，捕获语义相似性
- **StaticEmbeddingRetriever**：Word2Vec 100 维静态词向量取平均作为文档表示
- **ColBERTRetriever**：token 级多向量检索，MaxSim 评分，精细度高但计算成本更大
- **HybridRetriever**：核心方案，融合 BM25 + BGE + 交叉编码器重排

所有检索器继承 `BaseRetriever`，统一接口：
```python
retrieve(query, k=20) -> List[Tuple[doc_id, doc_text, score]]
```

---

# 4. 混合检索 — 核心方案

## Reciprocal Rank Fusion (RRF)

$$\text{score}(d) = \frac{1}{k + \text{rank}_{\text{BM25}}(d)} + \frac{1}{k + \text{rank}_{\text{BGE}}(d)}$$

其中 $k = 60$（标准常数）

## 工作流程

```
Query ──┬──→ BM25 检索 top-3k ──┐
        │                         ├─→ RRF 融合排序 ──→ top-3k 候选
        └──→ BGE 检索 top-3k ────┘                                  │
                                                                      ▼
                                                              Cross-Encoder 重排
                                                          (BAAI/bge-reranker-v2-m3)
                                                                      │
                                                                      ▼
                                                              最终 top-k 结果
```

**效果：** BM25 擅长关键词匹配，BGE 捕获语义，RRF 互补融合，交叉编码器精排。

---

# 5. 生成模块

## 两种 Generator

| | HuggingfaceGenerator | OpenAIGenerator |
|---|---|---|
| **模型** | Qwen2.5-0.5B-Instruct (本地) | Qwen 系列 (远程 API) |
| **推理方式** | HuggingFace Transformers | Aliyun DashScope API |
| **解码策略** | 确定性 (`do_sample=False`) | API 默认 |
| **适用场景** | 本地部署/无网络 | 更大模型/更好效果 |

## System Prompt 设计原则

- 要求输出**短语、实体名或 yes/no**
- 禁止完整句子和解释
- 适配 HotpotQA 答案格式

---

# 6. Agentic 多跳推理流程

```
                      用户问题
                         │
                    ┌────▼────┐
                    │ 查询改写 │  (多轮对话时，消解指代)
                    └────┬────┘
                         │
                    ┌────▼────────┐
                    │ 问题分解     │  LLM 将复杂问题拆为子问题
                    │ (Decompose)  │  最多 10 步
                    └────┬────────┘
                         │
              ┌──────────▼──────────┐
              │ 多跳 or 单跳？       │
              └──┬──────────────┬───┘
          多跳   │              │ 单跳
     ┌──────────▼───┐    ┌─────▼────────┐
     │ 逐步检索+推理 │    │ 单次检索+生成  │
     │ 链式传递答案  │    │ + 答案验证    │
     └──────┬───────┘    └──────┬───────┘
            │                   │
     ┌──────▼───────┐           │
     │ 综合子答案     │◄──────────┘
     │ 生成最终答案   │
     └──────────────┘
```

---

# 6.1 多跳推理 — 链式推理

**子问题 1：** *“《绿皮书》的导演是谁？”*
- 检索相关文档 → 生成答案：**Peter Farrelly**

**子问题 2：** *“Peter Farrelly 出生在哪里？”*
- 检索 + **注入子答案 1 作为上下文** → 生成答案：**Cumberland, Rhode Island**

**综合：** 将所有子答案传给 LLM，生成最终答案。

### 关键机制

- **链式上下文传递** — 后续子问题可利用前序子答案
- **原始问题补充检索** — 避免分解后丢失全局信息
- **答案后处理** — 去除冗余前缀（"The answer is:"），统一格式

---

# 6.2 单跳路径 — 答案验证

当问题被判定为单跳时，采用**带验证的生成**：

```
检索 → 生成答案 → LLM 验证 → 通过 → 返回
                          │
                          × 不通过 → 重新生成（最多 1 次重试）
```

**验证 Prompt 示例：**
> Verify whether the following answer is directly supported by the evidence.
> Question: ... Answer: ... Evidence: ...
> Respond with ONLY 'yes' or 'no'.

---

# 7. 评估框架

## 检索指标

| 指标 | 含义 |
|------|------|
| **nDCG@k** | 归一化折扣累积增益 — 排序质量 |
| **MAP@k** | 平均精度均值 — 检索准确性 |
| **Recall@k** | 召回率 — 覆盖度 |
| **Precision@k** | 精确率 |

## QA 指标

| 指标 | 含义 |
|------|------|
| **EM** | 精确匹配 — 标准化后是否完全一致 |
| **F1** | 预测与标准答案的 token 重叠度 |
| **Joint EM/F1** | 答案 × 支撑文档 的联合得分 |

---

# 7.1 实验结果

### Dense Retrieval (BGE-large) + qwen2.5-7b-instruct

**检索指标：**

| 指标 | @2 | @5 | @10 |
|------|----|----|-----|
| nDCG | 0.609 | 0.737 | **0.764** |
| MAP | 0.539 | 0.654 | 0.671 |
| Recall | 0.572 | 0.799 | **0.866** |

**QA 指标：**

| EM | F1 | Recall | Joint F1 |
|----|----|--------|----------|
| 0.335 | 0.470 | 0.570 | 0.229 |

> Dense 检索 nDCG@10 达到 **0.764**，Recall@10 达到 **0.866**

---

<!-- _class: lead -->

# 谢谢

## Q & A

项目地址: `PolyU-25Fall-COMP5423-RAG`
