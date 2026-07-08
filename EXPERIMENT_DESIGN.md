# 不同知识表示形式对LLM检索与推理影响：完整实验设计

## 1. 研究目标

本实验研究同一组知识采用不同表示形式时，对大语言模型问答性能产生的影响，并进一步定位影响主要发生在哪个阶段。

研究的三种知识表示为：

1. **Document**：自然语言事实句。
2. **Triple**：无序三元组。
3. **Graph**：带实体ID和实体名称的有向关系边。

核心研究问题包括：

- RQ1：知识表示形式是否影响LLM在Gold证据完整给定时的推理能力？
- RQ2：知识表示形式是否影响LLM在含噪上下文中识别并使用证据的能力？
- RQ3：知识表示形式是否影响检索器召回Gold事实和完整推理路径？
- RQ4：知识表示形式对最终端到端RAG准确率、token消耗和延迟有什么影响？
- RQ5：上述影响是否随推理路径从1-hop增加到2-hop、3-hop而变化？

---

## 2. 总体实验框架

实验由E0、E1、E2、E3四个阶段组成。

```text
E0 Closed-book
问题 → LLM
测量模型参数知识基线

E1 Gold-only
问题 + 完整Gold证据 → LLM
测量知识表示对纯证据推理的影响

E2 Forced-Gold with retrieved noise
问题 + E3真实检索噪声 + 强制补齐Gold → LLM
测量检索成功后，表示形式对证据选择和使用的影响

E3 Atomic RAG
问题 → 完整KB检索 → Top-100事实 → LLM
测量检索与证据使用的端到端综合影响
```

四个阶段应使用完全相同的题目集合、Gold答案和hop划分。

### 2.1 阶段差值的含义

在其他变量控制成立时，可以使用下列差值进行初步定位：

| 对比 | 主要反映的影响 |
|---|---|
| E1 − E0 | 外部Gold知识相对于模型参数知识带来的收益 |
| E1 − E2 | 加入大量无关事实后，证据筛选和抗噪能力造成的损失 |
| E2 − E3 | Gold路径缺失或不完整召回造成的损失 |
| E1 − E3 | 端到端RAG相对于理想Gold条件的总损失 |

这些差值是阶段性诊断，不应在小样本下直接解释为严格因果效应。

---

## 3. 数据集与问题选择

### 3.1 数据集

实验使用KQA Pro：

- `kb.json`：完整知识库。
- `train.json`：问题、答案、候选项和KoPL程序。
- canonical statements：由完整KB规范化得到的统一事实层。

完整KB包含约409,518条canonical statements。三种知识表示都由同一canonical statement生成，并通过`statement_id`对齐。

### 3.2 当前Pilot题集

当前流程固定使用15题：

| Hop | 题目数 |
|---|---:|
| 1-hop | 5 |
| 2-hop | 5 |
| 3-hop | 5 |
| 合计 | 15 |

题目文件：

```text
knowledge_representation_processing/output/oracle_hops_15/oracle_questions.jsonl
```

### 3.3 Hop定义

Hop数根据KoPL程序依赖关系和实际关系链计算，而不是简单使用KoPL函数总数。

每道入选题满足：

- 可以执行并追踪KoPL程序；
- 可以定位支持答案的Gold statements；
- Gold statement数量与标注hop数一致；
- 当前Pilot属于可追踪的单关系链问题。

因此：

```text
1-hop = 1条Gold关系事实
2-hop = 2条Gold关系事实
3-hop = 3条Gold关系事实
```

### 3.4 正式实验题集

正式实验计划扩展为：

```text
1-hop × 200
2-hop × 200
3-hop × 200
总计600题
```

题目只能抽取一次并保存为共享题集。E0、E1、E2、E3不得分别独立抽题。

---

## 4. 统一知识事实层

每条知识首先转换为canonical statement，包含：

- `statement_id`
- subject ID和label
- predicate
- object ID、label或literal
- qualifiers

三种表示只改变序列化形式，不改变底层事实。

示例事实：

```text
subject: Q90 / Paris
predicate: capital of
object: Q142 / France
```

对应三种表示：

### 4.1 Document

```text
Paris has the relation 'capital of' to France.
```

### 4.2 Triple

```text
(Paris | capital of | France)
```

### 4.3 Graph

```text
EDGE Q90 (Paris) --[capital of]--> Q142 (France)
```

如果存在qualifier，则统一附加在对应事实或边后，例如：

```text
[qualifiers: point in time=2020]
```

Graph不再先输出独立NODE表，以避免节点列表占据上下文并使关键边位于截断区域。

---

## 5. Atomic Chunk设计

### 5.1 设计原则

主实验采用Atomic Chunk：

```text
1 canonical statement = 1 retrieval chunk
```

同一事实的Document、Triple、Graph版本共享：

- 相同`statement_id`；
- 相同`chunk_id`；
- 相同事实边界；
- 不同的文本表示。

Atomic语料规模：

| 表示 | Chunk数 |
|---|---:|
| Document | 409,518 |
| Triple | 409,518 |
| Graph | 409,518 |

### 5.2 Atomic Chunk控制的混杂变量

旧Native设计中，Document和Graph每块最多20条事实，而Triple每块只有1条事实。这会让表示形式与chunk大小、chunk长度、事实密度同时变化。

Atomic设计消除了以下混杂：

- chunk包含的statement数量不同；
- 大chunk一次召回多条事实；
- BM25长度归一化受到不同chunk粒度影响；
- 不同表示在相同Top-K下获得不同数量的事实。

### 5.3 上下文选择

E3对每种表示独立检索，并固定选择：

```text
Top 100 statements
```

不强制相同token数。固定的是事实数量上限，而实际prompt token、completion token和延迟作为因变量记录。

---

## 6. E0：Closed-book基线

### 6.1 输入

```text
问题
```

不提供：

- KB知识；
- Gold证据；
- 检索结果；
- 原始候选项。

### 6.2 目的

测量模型仅依赖参数知识和语言先验时的问答能力，为后续外部知识实验提供基线。

### 6.3 主要指标

- Accuracy
- unmatched/invalid rate
- 按hop准确率
- prompt、completion和total tokens
- 模型延迟

---

## 7. E1：Gold-only实验

### 7.1 输入

```text
问题 + 对应题目的完整Gold statements
```

每道题的三种表示共享完全相同的Gold statement IDs。

### 7.2 目的

在检索误差和无关噪声被移除的条件下，测量知识表示对LLM理解证据和执行推理的影响。

### 7.3 控制条件

- 同一道题；
- 相同Gold statement IDs；
- 相同Gold事实数量；
- 相同模型；
- 相同temperature；
- 相同最大输出长度；
- 不展示候选项；
- 只改变Document、Triple、Graph序列化。

---

## 8. E3：Atomic RAG实验

E3先于E2运行，因为E2需要复用E3的真实检索结果。

### 8.1 检索流程

```text
原始问题
  → 对对应表示的完整KB Atomic语料执行BM25
  → 排序最多1000条statement
  → 选取Top 100 statements
  → 拼接为上下文
  → 调用LLM回答
```

### 8.2 检索器配置

当前三种表示统一使用：

| 参数 | 配置 |
|---|---|
| Retriever | BM25 |
| Query | 原始问题文本 |
| k1 | 1.5 |
| b | 0.75 |
| Tokenizer | Unicode word regex + casefold |
| 最大排序数 | 1000 statements |
| 最终上下文 | Top 100 statements |
| Tie break | 共享chunk ID稳定排序 |

Gold statement IDs不参与检索、打分、重排或上下文选择，只在检索完成后计算指标。

### 8.3 目的

测量知识表示对以下环节的综合影响：

- BM25匹配和排序；
- Gold事实召回；
- 完整推理路径召回；
- LLM从真实检索结果中回答问题。

---

## 9. E2：Forced-Gold with Retrieved Noise

### 9.1 数据来源

E2不重新运行检索器，而是直接读取E3保存的：

```text
rag_full_kb_processing/output/rag_atomic_hops_15/retrieval_results.jsonl
```

因此每道题、每种表示的噪声来自该表示在E3中的真实检索结果。

### 9.2 Gold补齐算法

对于每道题和每种表示：

1. 读取E3 Top-100 statement IDs；
2. 检查已召回的Gold statements；
3. 找出缺失的Gold statements；
4. 每缺失一条Gold，就删除一条最低排名的非Gold statement；
5. 在被删除statement的位置插入缺失Gold；
6. 保留原本已经召回的Gold；
7. 验证最终Gold完整路径召回率为100%；
8. 验证替换前后statement数量仍为100。

示意：

```text
E3上下文：98条噪声 + 2条已召回Gold
题目需要3条Gold

删除最低排名的1条噪声
插入缺失的1条Gold

E2上下文：97条噪声 + 3条完整Gold
总数仍为100
```

### 9.3 目的

在完整Gold路径保证存在、检索噪声仍然保留的条件下，测量LLM：

- 能否定位相关证据；
- 能否忽略无关证据；
- 能否连接多跳Gold路径；
- 是否对某种知识表示更敏感。

### 9.4 与E3的配对关系

同一表示下，E2和E3应满足：

- 相同问题；
- 相同检索器来源；
- 相同Top-100上下文规模；
- 相同真实检索噪声主体；
- 相同模型和生成参数；
- 唯一核心差异是E2强制补齐缺失Gold。

---

## 10. 模型与生成控制

当前正式对齐条件使用：

| 参数 | 配置 |
|---|---|
| 模型权重 | Qwen3 4B |
| Ollama模型名 | `qwen3-4b-16k` |
| Context window | 16,384 tokens |
| Temperature | 0 |
| Max output tokens | 4,096 |
| 回答形式 | 开放式精确答案 |
| 候选项是否展示 | 否 |

不向模型展示KQA Pro原始候选项，避免模型通过排除干扰选项猜测答案。

评分阶段可以使用原始候选项辅助规范化匹配，但候选项不得进入prompt。

---

## 11. 独立变量、控制变量与因变量

### 11.1 独立变量

- 知识表示：Document / Triple / Graph；
- 实验阶段：E0 / E1 / E2 / E3；
- 推理跳数：1 / 2 / 3-hop。

### 11.2 主要控制变量

| 控制对象 | 控制方法 |
|---|---|
| 问题集合 | 四阶段读取同一固定题集 |
| Gold答案 | 使用同一KQA Pro答案 |
| Hop划分 | 使用同一KoPL追踪结果 |
| 底层知识 | 三种表示共享canonical statement IDs |
| 语料范围 | 三种表示覆盖相同完整KB |
| Chunk粒度 | 一条statement一个Atomic chunk |
| Chunk ID | 三种表示共享相同chunk ID |
| 检索方法 | 相同BM25实现和参数 |
| Query | 只使用原始问题 |
| 上下文规模 | 固定Top-100 statements |
| 模型 | `qwen3-4b-16k` |
| 上下文窗口 | 16K |
| Temperature | 0 |
| 最大输出 | 4096 tokens |
| 候选项 | 不提供给模型 |
| Gold使用 | E3仅离线评估；E2仅在检索结束后补齐 |
| Graph格式 | 三阶段统一内联实体边格式 |
| 随机性 | 固定题目抽样种子和语料生成规则 |

### 11.3 不进行等量控制的变量

本实验不强制三种表示具有相同token数，因为token效率本身是研究对象。

固定的是：

- 相同事实粒度；
- 相同Top-100事实数量上限；
- 相同模型上下文上限。

测量的是：

- 实际prompt tokens；
- completion tokens；
- total tokens；
- 模型延迟。

### 11.4 因变量

#### 问答指标

- Accuracy
- Correct count
- Invalid/unmatched count和rate
- Accuracy on valid outputs
- 按hop准确率

#### 检索指标

- Statement Recall
- Hit Rate
- Complete-path Recall
- Retrieval MRR
- Context MRR
- First Gold rank

#### 效率指标

- Prompt tokens
- Completion tokens
- Total tokens
- 模型生成延迟
- 检索延迟（正式扩展时应补充）
- 端到端延迟（正式扩展时应补充）

---

## 12. 推荐运行顺序

```text
步骤1：固定共享问题集和Gold statements
步骤2：构建完整KB canonical statements
步骤3：构建三种Atomic语料
步骤4：运行E0 Closed-book
步骤5：运行E1 Gold-only
步骤6：运行E3 Atomic RAG并保存retrieval_results
步骤7：E2读取E3 retrieval_results并补齐Gold
步骤8：统一汇总E0/E1/E2/E3
```

必须先运行E3再运行E2，因为E2依赖E3的真实检索噪声。

---

## 13. 当前文件与结果位置

### E0

```text
closed_book_processing/
closed_book_processing/output/e0_hops_15/
```

### E1

```text
knowledge_representation_processing/run_oracle_hops_15.py
knowledge_representation_processing/output/oracle_hops_15_graph_inline_16k/
```

### E3 Atomic

```text
rag_full_kb_processing/build_atomic_kb_corpora.py
rag_full_kb_processing/run_rag_atomic_hops_15.py
rag_full_kb_processing/output/atomic_kb/
rag_full_kb_processing/output/rag_atomic_hops_15/
```

### E2 Atomic Forced-Gold

```text
e2_forced_gold_processing/run_e2_forced_gold_hops_15.py
e2_forced_gold_processing/output/e2_atomic_hops_15/
```

每个实验通常包含：

- `metrics.json`：总体和按hop汇总；
- `predictions.jsonl`：逐题回答和原始响应；
- `retrieval_results.jsonl`：E3逐题检索结果；
- `configuration.json`：E3配置；
- `bundle_validation.json`：E1/E2数据校验；
- `e2_contexts.jsonl`：E2替换前后statement和最终上下文。

---

## 14. 结果解释规则

### 14.1 E1的解释

E1中的表示差异主要反映Gold证据在不同序列化形式下的可读性和推理难度。

### 14.2 E2的解释

同一表示下比较E1和E2，可以观察加入真实检索噪声后的性能损失。

同一表示下比较E2和E3，可以观察Gold路径缺失对性能的影响。

### 14.3 E3的解释

E3同时包含：

- 检索排序能力；
- 完整路径召回能力；
- LLM证据使用能力；
- 表示带来的token和延迟成本。

因此不能只根据E3 Accuracy判断某种表示的纯推理能力，必须结合E1、E2和检索指标解释。

---

## 15. 当前仍需注意的控制问题

### 15.1 E0模型名尚未统一

当前E0结果来自`qwen3:4b`，E1/E2/E3来自`qwen3-4b-16k`。两者底层权重相同，但上下文配置和实验标识不同。

正式实验应使用`qwen3-4b-16k`重新运行E0，使配置记录完全一致。

### 15.2 E2跨表示比较仍包含噪声内容差异

E2保留每种表示在E3中各自检索到的真实噪声。因此：

- 同一表示的E2与E3比较较干净；
- E2中Document、Triple、Graph横向比较时，噪声statement集合不完全相同。

如果需要严格研究“相同噪声下的表示使用能力”，应增加E2-Shared-Noise补充实验：三种表示共享同一组statement IDs，再分别序列化。

### 15.3 E2与E3文本拼接模板需完全一致

正式扩大前应校验：

- Document的statement连接符一致；
- Triple的顺序和换行一致；
- Graph标题、边格式和顺序一致；
- 除Gold替换外，E2和E3 prompt逐字符或逐token一致。

### 15.4 当前样本量过小

15题仅用于流程验证。每道题会使总体准确率变化约6.67个百分点，每个hop内单题会造成20个百分点变化。

当前结果不得直接作为最终统计结论。

### 15.5 问题类型覆盖有限

当前题目主要是可追踪的单关系链。KQA Pro还包含过滤、比较、计数、集合交并和限定符推理。正式研究应明确：

- 只研究关系链多跳问题；或
- 扩大到其他KoPL类型，并单独分层报告。

### 15.6 Invalid输出需要细分

目前`prediction is None`主要表示回答没有匹配原始候选答案。正式评估建议细分：

- `UNKNOWN`
- 空输出
- 格式错误
- 未匹配实体别名
- 匹配但答案错误
- 正确答案

---

## 16. 正式600题实验要求

扩大到600题时，应增加：

- 共享题集文件，禁止各阶段重新抽题；
- 每次调用后立即写JSONL；
- 以`question_id + representation`作为唯一运行键；
- 断点续跑；
- API失败重试；
- 保存原始响应；
- 保存模型、上下文、prompt模板和语料哈希；
- 记录检索、生成和端到端延迟；
- 配对bootstrap置信区间；
- 同题McNemar检验或其他配对显著性检验；
- 按hop、问题类型、Gold完整性分层报告。

预计调用量：

| 实验 | 调用数 |
|---|---:|
| E0 | 600 |
| E1 | 1,800 |
| E2 | 1,800 |
| E3 | 1,800 |
| 合计 | 6,000 |

为了保持延迟可比性，正式实验不应对同一个本地模型并发发送多个请求。

---

## 17. 最终实验报告结构

正式结果建议按以下顺序报告：

1. E0闭卷基线；
2. E1三种表示的Gold-only准确率；
3. E2三种表示的含噪证据使用准确率；
4. E3三种表示的检索指标；
5. E3端到端准确率；
6. E1/E2/E3按hop分层结果；
7. 完整路径召回条件下的准确率；
8. prompt、completion、total tokens和延迟；
9. 配对显著性检验和置信区间；
10. E2-Shared-Noise与Native RAG补充实验。

该结构能够分别回答“表示是否影响推理”“表示是否影响含噪证据使用”“表示是否影响检索”以及“最终综合影响如何”。
