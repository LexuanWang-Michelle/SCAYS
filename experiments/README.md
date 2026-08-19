# Experiments

SCAYS 数据质量验证与方法论实验代码。

## 实验列表

| 脚本 | 对应方法 | 功能 |
|------|---------|------|
| `colander_full.py` | Colander (Sala, NeurIPS 2024) | 标注质量审计：提取 BERT 内部特征，训练质检模型，识别 LLM 标注盲区 |
| `alchemist_experiment.py` | AlCHEmist (Sala, NeurIPS 2024) | 多规则标注覆盖率实验：10 条标注函数投票，量化隐含情绪深度 |
| `pruning_colander.py` | Colander + Superfiltering | 数据剪枝对比实验：Colander 置信度选择 vs 全量 vs 随机 |

## 运行方法

### 1. Colander 标注质量审计

```bash
# 提取 BERT 内部特征
python3 colander_full.py step2

# 生成校准模板（需人工标注 500 条）
python3 colander_full.py gen_calib

# 训练 Colander
python3 colander_full.py step3

# 全量扫描
python3 colander_full.py step4
```

### 2. AlCHEmist 多规则覆盖率实验

```bash
python3 alchemist_experiment.py
```

### 3. 数据剪枝对比实验

```bash
python3 pruning_colander.py
```

## 核心发现

- **重叠密度**：整体 24.4%，仅 1/4 有情绪数据能被关键词覆盖
- **隐含情绪深度**：64.9% 的情绪数据为 hard-only，无任何表面模式可匹配
- **LLM 盲区**：Colander 识别出 11.5% 的 LLM 标注不可靠数据
- **剪枝实验**：去除低置信度数据反而损害性能（-6.4%），证明困难数据是训练核心

## 依赖

```
torch >= 1.13
transformers >= 4.20
pandas
numpy
scikit-learn
```

## 参考论文

- Sala et al., *Pearls from Pebbles* (NeurIPS 2024)
- Sala et al., *The AlCHEmist* (NeurIPS 2024)
- Sala et al., *Weak-to-Strong Generalization* (ICLR 2025)
- Ming Li et al., *Superfiltering* (ACL 2024)
