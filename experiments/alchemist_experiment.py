"""
AlCHEmist 规则实验
==================
模仿 Sala 的 AlCHEmist 论文：用多条标注函数（LFs）投票，
对比单规则 vs 多规则投票 vs LLM 标签的一致率。

实验设计：
  A组（baseline）：单条情绪关键词规则
  B组（AlCHEmist）：10条规则投票聚合
  对比对象：LLM 的三分类标签（label_3）

运行：python3 alchemist_experiment.py
"""

import pandas as pd
import numpy as np
import re
import os

# ============================================================
# 数据加载
# ============================================================
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "..", "data", "训练数据集.csv")

df = pd.read_csv(DATA_PATH, low_memory=False)
df = df.dropna(subset=["Sentence", "label"])
texts = df["Sentence"].astype(str).tolist()

# 生成三分类 label_3（和 bert_3class.py 一致）
POSITIVE_WORDS = {"正面"}
NEGATIVE_WORDS = {"焦虑", "愤怒", "自卑", "委屈", "无助", "孤独", "抑郁", "不赞同"}

def map_label(row) -> int:
    emotion = str(row.get("隐含情绪", "")).strip()
    if emotion in ("", "nan"):
        if str(row.get("label", 0)) == "1":
            return 1
        return 0
    if emotion == "中性":
        return 0
    if emotion in POSITIVE_WORDS:
        return 2
    parts = set(p.strip() for p in emotion.replace("，", "/").split("/"))
    has_positive = bool(parts & POSITIVE_WORDS)
    has_negative = bool(parts & NEGATIVE_WORDS)
    if has_negative:
        return 1
    elif has_positive:
        return 2
    else:
        return 0

df["label_3"] = df.apply(map_label, axis=1)
# 简化为二分类对比：0=中性, 1=有情绪（负面+正面合并）
df["label_binary"] = (df["label_3"] != 0).astype(int)


# ============================================================
# 10 条标注函数（Labeling Functions）
# ============================================================

# --- LF1: 情绪关键词 ---
NEG_KEYWORDS = set("焦虑 崩溃 难过 烦 累 痛苦 害怕 委屈 绝望 无助 抑郁 愤怒 生气 恨 哭 心疼 心酸 难受 郁闷 窒息 想死 自卑 孤独 压抑 内耗 破防 emo".split())
POS_KEYWORDS = set("开心 高兴 期待 激动 幸福 满足 感动 骄傲 幸运 快乐 兴奋 感恩 庆幸".split())

def lf_emotion_keywords(text):
    for w in NEG_KEYWORDS | POS_KEYWORDS:
        if w in text:
            return 1
    return -1  # 弃权

# --- LF2: 否定+动词组合 ---
NEG_PATTERNS = [
    r"不想.{0,4}(上学|活|回家|写|去|动|说话|学|做|干|来|努力)",
    r"没有.{0,4}(朋友|人|意义|希望|出路|办法)",
    r"受不了|忍不了|撑不住|顶不住|扛不住|坚持不下去",
]

def lf_negation(text):
    for pat in NEG_PATTERNS:
        if re.search(pat, text):
            return 1
    return -1

# --- LF3: 求助/无助句式 ---
HELP_WORDS = ["怎么办", "咋办", "咋整", "该怎样", "救命", "谁能帮", "怎么解决", "怎么处理"]

def lf_helpless(text):
    for w in HELP_WORDS:
        if w in text:
            return 1
    return -1

# --- LF4: 程度词+负面评价 ---
DEGREE_NEG_PATTERNS = [
    r"(真的|太|好|超级|特别|非常).{0,3}(累|烦|难|怕|慌|痛苦|焦虑|崩溃|难受|绝望|恶心|气|恨)",
]

def lf_degree_negative(text):
    for pat in DEGREE_NEG_PATTERNS:
        if re.search(pat, text):
            return 1
    return -1

# --- LF5: 身体/生理信号 ---
BODY_WORDS = ["失眠", "哭了", "哭", "想哭", "胃疼", "头疼", "心慌", "手抖", "吐了", "恶心", "喘不过气", "心跳加速", "眼泪"]

def lf_body_signal(text):
    for w in BODY_WORDS:
        if w in text:
            return 1
    return -1

# --- LF6: 学业压力场景 ---
EXAM_WORDS = ["月考", "一模", "二模", "期末", "高考", "中考", "成绩", "分数", "排名", "挂科"]
BAD_RESULT = ["完了", "炸了", "凉了", "挂了", "废了", "来不及", "考砸", "没考好", "退步", "垫底", "倒数"]

def lf_academic_pressure(text):
    has_exam = any(w in text for w in EXAM_WORDS)
    has_bad = any(w in text for w in BAD_RESULT)
    if has_exam and has_bad:
        return 1
    return -1

# --- LF7: 人际冲突 ---
CONFLICT_WORDS = ["被骂", "被打", "吵架", "孤立", "冷暴力", "没朋友", "被排挤", "被针对", "被欺负", "霸凌", "被说闲话"]

def lf_interpersonal(text):
    for w in CONFLICT_WORDS:
        if w in text:
            return 1
    return -1

# --- LF8: 家庭情绪 ---
FAMILY_PATTERNS = [
    r"(爸|妈|父母|家长|爸妈).{0,8}(吵架|打我|骂我|不理解|逼我|不让|偏心)",
    r"(离婚|家暴|重男轻女|原生家庭)",
]

def lf_family(text):
    for pat in FAMILY_PATTERNS:
        if re.search(pat, text):
            return 1
    return -1

# --- LF9: 自我否定 ---
SELF_NEG = ["我太差了", "我什么都不行", "废物", "没用", "不配", "我好差", "我是垃圾", "我真笨", "我不够好", "讨厌自己"]

def lf_self_negative(text):
    for w in SELF_NEG:
        if w in text:
            return 1
    return -1

# --- LF10: 正面情绪 ---
POS_PATTERNS = [
    r"(终于|成功|考上|上岸|录取|通过)",
    r"(好开心|太开心|超开心|特别开心|真开心)",
    r"(感动|骄傲|幸福|满足|庆幸|感恩|值得)",
]

def lf_positive(text):
    for pat in POS_PATTERNS:
        if re.search(pat, text):
            return 1
    return -1


# ============================================================
# 运行所有 LF
# ============================================================
ALL_LFS = [
    ("LF1_情绪关键词", lf_emotion_keywords),
    ("LF2_否定表达", lf_negation),
    ("LF3_求助无助", lf_helpless),
    ("LF4_程度+负面", lf_degree_negative),
    ("LF5_身体信号", lf_body_signal),
    ("LF6_学业压力", lf_academic_pressure),
    ("LF7_人际冲突", lf_interpersonal),
    ("LF8_家庭情绪", lf_family),
    ("LF9_自我否定", lf_self_negative),
    ("LF10_正面情绪", lf_positive),
]

print("=" * 60)
print("AlCHEmist 规则实验：10 条标注函数投票")
print("=" * 60)

# 对每条数据跑 10 个函数
results = {}
for name, func in ALL_LFS:
    votes = [func(t) for t in texts]
    results[name] = votes

lf_df = pd.DataFrame(results)

# ============================================================
# 统计每条 LF 的覆盖率和准确率
# ============================================================
print("\n=== 各标注函数统计 ===")
print(f"{'函数':<16} {'覆盖率':>8} {'投1数':>8} {'投0数':>8} {'弃权数':>8} {'准确率(vs LLM)':>14}")
print("-" * 70)

for name in results:
    votes = np.array(results[name])
    covered = (votes != -1).sum()
    vote_1 = (votes == 1).sum()
    vote_0 = (votes == 0).sum()
    abstain = (votes == -1).sum()
    coverage = covered / len(votes)

    # 准确率：在投了票的数据上，和 label_binary 一致的比例
    covered_mask = votes != -1
    if covered_mask.sum() > 0:
        acc = (votes[covered_mask] == df["label_binary"].values[covered_mask]).mean()
    else:
        acc = 0
    print(f"{name:<16} {coverage:>7.1%} {vote_1:>8,} {vote_0:>8,} {abstain:>8,} {acc:>13.1%}")


# ============================================================
# 多数投票聚合
# ============================================================
print("\n\n=== 多数投票聚合 ===")

# 方法：对每条数据，统计 10 个 LF 中投 1 的数量
# 如果 >=1 个 LF 投了 1 → 最终判定：有情绪
# 如果 0 个 LF 投 1 → 最终判定：中性（所有 LF 都弃权或没命中）
vote_counts = lf_df.apply(lambda row: (row == 1).sum(), axis=1)
majority_label = (vote_counts >= 1).astype(int)

# 统计
print(f"多数投票结果：有情绪={majority_label.sum():,}, 中性={len(majority_label)-majority_label.sum():,}")

# 和 LLM 对比
agree = (majority_label == df["label_binary"]).sum()
total = len(df)
print(f"和 LLM 标签一致率: {agree:,}/{total:,} = {agree/total:.1%}")

# 更严格：至少 2 个 LF 投 1
majority_2 = (vote_counts >= 2).astype(int)
agree_2 = (majority_2 == df["label_binary"]).sum()
print(f"至少2个LF投1: 一致率 = {agree_2/total:.1%}")

# 至少 3 个
majority_3 = (vote_counts >= 3).astype(int)
agree_3 = (majority_3 == df["label_binary"]).sum()
print(f"至少3个LF投1: 一致率 = {agree_3/total:.1%}")


# ============================================================
# 对比实验：单规则 vs 多规则
# ============================================================
print("\n\n=== 实验对比 ===")

# Baseline：只用 LF1（情绪关键词）
lf1_votes = np.array(results["LF1_情绪关键词"])
# LF1 不弃权时的准确率
lf1_covered = lf1_votes != -1
lf1_acc = (lf1_votes[lf1_covered] == df["label_binary"].values[lf1_covered]).mean()
lf1_coverage = lf1_covered.mean()

# 多规则投票（>=1）
multi_acc = agree / total
multi_coverage = 1.0  # 多规则投票对所有数据都有结果

print(f"{'方法':<20} {'覆盖率':>8} {'一致率(vs LLM)':>14}")
print("-" * 50)
print(f"{'单规则(LF1关键词)':<20} {lf1_coverage:>7.1%} {lf1_acc:>13.1%}")
print(f"{'10规则投票(>=1)':<20} {multi_coverage:>7.1%} {multi_acc:>13.1%}")
print(f"{'10规则投票(>=2)':<20} {multi_coverage:>7.1%} {agree_2/total:>13.1%}")
print(f"{'10规则投票(>=3)':<20} {multi_coverage:>7.1%} {agree_3/total:>13.1%}")


# ============================================================
# 分析规则之间的互补性
# ============================================================
print("\n\n=== 规则互补性分析 ===")
# 每条数据被多少个 LF 覆盖
covered_per_sample = lf_df.apply(lambda row: (row != -1).sum(), axis=1)
print(f"每条数据平均被 {covered_per_sample.mean():.1f} 条规则覆盖")
print(f"0条规则覆盖: {(covered_per_sample == 0).sum():,} 条 ({(covered_per_sample == 0).mean():.1%})")
print(f"1条规则覆盖: {(covered_per_sample == 1).sum():,} 条 ({(covered_per_sample == 1).mean():.1%})")
print(f"2+条规则覆盖: {(covered_per_sample >= 2).sum():,} 条 ({(covered_per_sample >= 2).mean():.1%})")
print(f"5+条规则覆盖: {(covered_per_sample >= 5).sum():,} 条 ({(covered_per_sample >= 5).mean():.1%})")

# 那些 LLM 说有情绪但所有规则都没覆盖的 = hard-only
has_emotion = df["label_binary"] == 1
no_rule_hit = vote_counts == 0
hard_only = has_emotion & no_rule_hit
print(f"\nLLM标有情绪但10条规则都没命中(hard-only): {hard_only.sum():,} 条 ({hard_only.sum()/has_emotion.sum():.1%} of 有情绪数据)")

# 保存结果
output = os.path.join(BASE, "colander_output", "alchemist_results.csv")
df["vote_count"] = vote_counts.values
df["majority_label"] = majority_label.values
df.to_csv(output, index=False, encoding="utf-8-sig")
print(f"\n结果已保存: {output}")
