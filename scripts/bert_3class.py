"""
SCAYS — BERT 三分类情绪分类器
判断句子性质：正面情绪 / 负面情绪 / 中性

标签映射：
  0 = 中性     (neutral)
  1 = 负面情绪  (negative) — 焦虑/愤怒/自卑/委屈/无助/孤独/抑郁
  2 = 正面情绪  (positive)

使用方法：
  训练：python3 bert_3class.py --train
  预测：python3 bert_3class.py --predict --input to_predict.csv
  评估：python3 bert_3class.py --eval
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from sklearn.metrics import classification_report, confusion_matrix
import os
import argparse
import json
from datetime import datetime

# ============ 配置 ============
DATA_DIR    = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE  = os.path.join(DATA_DIR, "训练数据集.csv")
MODEL_DIR   = os.path.join(DATA_DIR, "my_bert")       # 本地 BERT-base-Chinese
OUTPUT_DIR  = os.path.join(DATA_DIR, "best_model_3class")

# 超参数
MAX_LEN    = 128
BATCH_SIZE = 16
EPOCHS     = 10
LR         = 2e-5
WEIGHT_DECAY   = 0.01
WARMUP_RATIO   = 0.1
EARLY_STOP     = 3

# 三个类别的名称（对应 label 0/1/2）
LABEL_NAMES = ["中性", "负面", "正面"]

# 离线模式（不联网下载模型）
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"设备: {device}")


# ============ 标签映射 ============
def map_label(row) -> int:
    """
    根据 隐含情绪 和 情绪类型 映射到三分类标签
      0 = 中性
      1 = 负面
      2 = 正面
    """
    emotion = str(row.get("隐含情绪", "")).strip()
    emo_type = str(row.get("情绪类型", "")).strip()

    # 如果是 NaN / 空
    if emotion in ("", "nan"):
        # 显性情绪但没有细分 → 根据 label 字段粗判
        if str(row.get("label", 0)) == "1":
            return 1  # 有情绪但没细分，保守归负面
        return 0

    # 中性
    if emotion == "中性":
        return 0

    # 正面（单纯正面，不含负面混合）
    pure_positive = {"正面"}
    if emotion in pure_positive:
        return 2

    # 混合情绪（如"焦虑/正面""正面/焦虑"）→ 取主要成分
    # 正面词集合
    positive_words = {"正面"}
    # 负面词集合
    negative_words = {"焦虑", "愤怒", "自卑", "委屈", "无助", "孤独", "抑郁", "不赞同"}

    parts = set(p.strip() for p in emotion.replace("，", "/").split("/"))
    has_positive = bool(parts & positive_words)
    has_negative = bool(parts & negative_words)

    if has_negative and has_positive:
        # 混合情绪：负面优先（情绪风险更重要）
        return 1
    elif has_negative:
        return 1
    elif has_positive:
        return 2
    else:
        # 未知标签 → 中性
        return 0


def load_and_map(filepath: str) -> pd.DataFrame:
    """加载训练数据，生成 三分类标签"""
    df = pd.read_csv(filepath)
    df["label_3"] = df.apply(map_label, axis=1)

    print("\n标签分布（三分类）：")
    dist = df["label_3"].value_counts().sort_index()
    for lbl, cnt in dist.items():
        print(f"  {LABEL_NAMES[lbl]} ({lbl}): {cnt} 条")
    return df


# ============ Dataset ============
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ============ 训练 ============
def train():
    print("=" * 55)
    print("BERT 三分类训练：正面 / 负面 / 中性")
    print("=" * 55)

    df = load_and_map(TRAIN_FILE)
    texts  = df["Sentence"].astype(str).tolist()
    labels = df["label_3"].tolist()

    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_DIR, num_labels=3
    ).to(device)

    full_ds = TextDataset(texts, labels, tokenizer, MAX_LEN)
    val_size   = max(1, int(len(full_ds) * 0.2))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(
        full_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"\n训练集: {train_size} 条 | 验证集: {val_size} 条")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # 类别权重：中性样本多，正面/负面样本少 → 给少数类更高权重
    counts = [labels.count(i) for i in range(3)]
    total  = sum(counts)
    # 逆频率权重
    weights = torch.tensor(
        [total / (3 * c) if c > 0 else 1.0 for c in counts],
        dtype=torch.float32
    ).to(device)
    print(f"类别权重: {[f'{LABEL_NAMES[i]}={weights[i]:.2f}' for i in range(3)]}")
    loss_fn = CrossEntropyLoss(weight=weights)

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    print(f"\n开始训练 (epochs={EPOCHS}, batch={BATCH_SIZE}, lr={LR})...\n")

    for epoch in range(EPOCHS):
        # --- 训练 ---
        model.train()
        total_loss = 0
        for batch in train_loader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            lbls  = batch["label"].to(device)

            out  = model(input_ids=ids, attention_mask=mask)
            loss = loss_fn(out.logits, lbls)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_train = total_loss / len(train_loader)

        # --- 验证 ---
        model.eval()
        val_loss = 0
        preds_all, true_all = [], []
        with torch.no_grad():
            for batch in val_loader:
                ids   = batch["input_ids"].to(device)
                mask  = batch["attention_mask"].to(device)
                lbls  = batch["label"].to(device)

                out  = model(input_ids=ids, attention_mask=mask)
                loss = loss_fn(out.logits, lbls)
                val_loss += loss.item()

                preds_all.extend(torch.argmax(out.logits, dim=-1).cpu().numpy())
                true_all.extend(lbls.cpu().numpy())

        avg_val = val_loss / len(val_loader)
        acc = np.mean(np.array(preds_all) == np.array(true_all))

        history.append({"epoch": epoch+1, "train_loss": avg_train,
                         "val_loss": avg_val, "val_acc": acc})

        print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
              f"训练Loss: {avg_train:.4f} | "
              f"验证Loss: {avg_val:.4f} | "
              f"验证Acc: {acc:.2%}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print(f"  >> 验证Loss改善，模型已保存到 {OUTPUT_DIR}")
        else:
            patience_counter += 1
            print(f"  >> 验证Loss未改善 ({patience_counter}/{EARLY_STOP})")
            if patience_counter >= EARLY_STOP:
                print(f"  >> 早停！")
                break

    # 保存训练历史
    with open(os.path.join(DATA_DIR, "training_history_3class.json"), "w") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    # 最终报告
    print("\n" + "=" * 55)
    print("最终模型评估（验证集）")
    print("=" * 55)
    best = BertForSequenceClassification.from_pretrained(OUTPUT_DIR).to(device)
    best.eval()
    preds_all, true_all = [], []
    with torch.no_grad():
        for batch in val_loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out  = best(input_ids=ids, attention_mask=mask)
            preds_all.extend(torch.argmax(out.logits, dim=-1).cpu().numpy())
            true_all.extend(batch["label"].numpy())

    print(classification_report(true_all, preds_all, target_names=LABEL_NAMES))
    print("混淆矩阵（行=真实，列=预测）：")
    print(f"       {'  '.join(f'{n:4s}' for n in LABEL_NAMES)}")
    cm = confusion_matrix(true_all, preds_all)
    for i, row in enumerate(cm):
        print(f"  {LABEL_NAMES[i]:4s}  {row}")


# ============ 预测 ============
def predict(input_file: str):
    print("=" * 55)
    print("BERT 三分类预测")
    print("=" * 55)

    if not os.path.exists(OUTPUT_DIR):
        print(f"找不到训练好的模型 {OUTPUT_DIR}")
        print("请先运行: python3 bert_3class.py --train")
        return

    df = pd.read_csv(input_file)
    print(f"待预测: {len(df)} 条")

    # 兼容不同列名
    text_col = "Sentence" if "Sentence" in df.columns else "raw_text"
    texts = df[text_col].astype(str).tolist()

    tokenizer = BertTokenizer.from_pretrained(OUTPUT_DIR)
    model = BertForSequenceClassification.from_pretrained(OUTPUT_DIR).to(device)
    model.eval()

    all_preds, all_probs = [], []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=MAX_LEN, return_tensors="pt").to(device)
        with torch.no_grad():
            out   = model(**inputs)
            probs = F.softmax(out.logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())

    df["预测标签"]   = all_preds
    df["预测情绪"]   = [LABEL_NAMES[p] for p in all_preds]
    df["中性概率"]   = [round(p[0], 4) for p in all_probs]
    df["负面概率"]   = [round(p[1], 4) for p in all_probs]
    df["正面概率"]   = [round(p[2], 4) for p in all_probs]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_file = os.path.join(DATA_DIR, f"预测结果_三分类_{timestamp}.csv")
    df.to_csv(out_file, index=False, encoding="utf-8-sig")

    print(f"\n预测分布：")
    dist = pd.Series(all_preds).value_counts().sort_index()
    for lbl, cnt in dist.items():
        print(f"  {LABEL_NAMES[lbl]}: {cnt} 条 ({cnt/len(all_preds):.1%})")
    print(f"\n结果已保存到 {out_file}")


# ============ 主入口 ============
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SCAYS BERT 三分类（正面/负面/中性）")
    parser.add_argument("--train",   action="store_true", help="训练模型")
    parser.add_argument("--predict", action="store_true", help="预测新数据")
    parser.add_argument("--input",   type=str, default="to_predict.csv",
                        help="预测输入文件（默认 to_predict.csv）")
    args = parser.parse_args()

    if args.train:
        train()
    elif args.predict:
        input_path = args.input if os.path.isabs(args.input) \
                     else os.path.join(DATA_DIR, args.input)
        predict(input_path)
    else:
        parser.print_help()
        print("\n使用流程：")
        print("  1. python3 bert_3class.py --train")
        print("  2. python3 bert_3class.py --predict --input to_predict.csv")
