"""
Colander 数据剪枝实验（简化版）
================================
用 Colander 置信度选数据，对比全量训练效果。

策略：
  A：全量训练（baseline）
  B：Colander 高置信度 Top 50%
  C：Colander 高置信度 Top 30%
  D：随机 50%（对照组）

运行：python3 pruning_colander.py
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "..", "best_model_3class")
BERT_BASE_PATH = os.path.join(BASE, "..", "my_bert")
COLANDER_PATH = os.path.join(BASE, "colander_output", "all_confidence.csv")
OUTPUT_DIR = os.path.join(BASE, "pruning_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 16
MAX_LEN = 128
EPOCHS = 3
LR = 2e-5

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"设备: {device}")


# ============================================================
# 标签映射
# ============================================================
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
    if bool(parts & NEGATIVE_WORDS):
        return 1
    elif bool(parts & POSITIVE_WORDS):
        return 2
    return 0


# ============================================================
# Dataset
# ============================================================
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]), padding="max_length",
            truncation=True, max_length=MAX_LEN, return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ============================================================
# 训练+评估函数
# ============================================================
def train_and_eval(train_texts, train_labels, test_texts, test_labels, name=""):
    """从头训练一个 BERT 三分类，返回准确率"""
    tokenizer = BertTokenizer.from_pretrained(BERT_BASE_PATH, local_files_only=True)
    model = BertForSequenceClassification.from_pretrained(
        BERT_BASE_PATH, num_labels=3, local_files_only=True
    )
    model.to(device)

    train_ds = TextDataset(train_texts, train_labels, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"    [{name}] Epoch {epoch+1}/{EPOCHS} Loss: {avg_loss:.4f}")

    # 评估
    model.eval()
    test_ds = TextDataset(test_texts, test_labels, tokenizer)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=ids, attention_mask=mask)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())

    acc = accuracy_score(test_labels, all_preds)
    return acc


# ============================================================
# 主实验
# ============================================================
def main():
    print("=" * 60)
    print("Colander 数据剪枝对比实验")
    print("=" * 60)

    # 加载数据 + Colander 分数
    df = pd.read_csv(COLANDER_PATH, low_memory=False)
    df["label_3"] = df.apply(map_label, axis=1)

    print(f"总数据: {len(df):,} 条")
    print(f"Colander 置信度: 平均{df['colander_confidence'].mean():.3f}, "
          f"中位数{df['colander_confidence'].median():.3f}")

    # 固定测试集（20%）
    np.random.seed(42)
    n = len(df)
    test_idx = np.random.choice(n, size=int(n * 0.2), replace=False)
    train_mask = np.ones(n, dtype=bool)
    train_mask[test_idx] = False

    test_df = df.iloc[test_idx]
    train_df = df.iloc[train_mask].reset_index(drop=True)

    test_texts = test_df["Sentence"].astype(str).tolist()
    test_labels = test_df["label_3"].tolist()

    print(f"测试集: {len(test_texts):,} 条")
    print(f"训练池: {len(train_df):,} 条")

    # ========== 定义策略 ==========
    strategies = {}

    # A: 全量训练
    strategies["A_全量(100%)"] = train_df

    # B: Colander 高置信度 Top 50%
    top50 = train_df.nlargest(int(len(train_df) * 0.5), "colander_confidence")
    strategies["B_Colander高50%"] = top50

    # C: Colander 高置信度 Top 30%
    top30 = train_df.nlargest(int(len(train_df) * 0.3), "colander_confidence")
    strategies["C_Colander高30%"] = top30

    # D: 随机 50%（对照组）
    random50 = train_df.sample(frac=0.5, random_state=42)
    strategies["D_随机50%"] = random50

    # ========== 跑实验 ==========
    results = {}
    for name, subset in strategies.items():
        s_texts = subset["Sentence"].astype(str).tolist()
        s_labels = subset["label_3"].tolist()
        print(f"\n{'='*40}")
        print(f"策略: {name} ({len(s_texts):,} 条)")
        print(f"{'='*40}")
        acc = train_and_eval(s_texts, s_labels, test_texts, test_labels, name=name)
        results[name] = {"size": len(s_texts), "accuracy": acc}
        print(f"  → 准确率: {acc:.4f} ({acc:.1%})")

    # ========== 汇总 ==========
    print("\n\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)
    print(f"\n{'策略':<20} {'训练数据量':>10} {'准确率':>8} {'对比全量':>10}")
    print("-" * 52)
    baseline_acc = results["A_全量(100%)"]["accuracy"]
    for name, r in results.items():
        diff = r["accuracy"] - baseline_acc
        diff_str = f"{diff:+.2%}" if name != "A_全量(100%)" else "baseline"
        print(f"{name:<20} {r['size']:>10,} {r['accuracy']:>7.1%} {diff_str:>10}")

    # 保存
    result_df = pd.DataFrame([
        {"策略": name, "训练数据量": r["size"], "准确率": round(r["accuracy"], 4),
         "对比全量": round(r["accuracy"] - baseline_acc, 4)}
        for name, r in results.items()
    ])
    result_path = os.path.join(OUTPUT_DIR, "colander_pruning_results.csv")
    result_df.to_csv(result_path, index=False, encoding="utf-8-sig")
    print(f"\n结果保存至: {result_path}")

    # 关键发现
    print("\n" + "=" * 60)
    print("关键发现")
    print("=" * 60)
    b_acc = results["B_Colander高50%"]["accuracy"]
    d_acc = results["D_随机50%"]["accuracy"]
    print(f"\n  Colander高50% vs 随机50%: {b_acc:.1%} vs {d_acc:.1%} (差距: {b_acc-d_acc:+.2%})")
    if b_acc > d_acc:
        print(f"  → Colander 选数据比随机选更好！")
    print(f"\n  Colander高50% vs 全量: {b_acc:.1%} vs {baseline_acc:.1%} (差距: {b_acc-baseline_acc:+.2%})")
    if abs(b_acc - baseline_acc) < 0.01:
        print(f"  → 用50%数据就接近全量效果！数据剪枝有效！")
    elif b_acc > baseline_acc:
        print(f"  → 去掉烂数据反而更好！")


if __name__ == "__main__":
    main()
