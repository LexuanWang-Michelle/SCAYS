"""
Colander 实验 - 完整流程
=========================
基于 Sala 的 Pearls from Pebbles (NeurIPS 2024)
目标：找出 LLM 标注中"嘴上自信、内部犹豫"的数据点

流程：
  Step 1: 训 Baseline BERT（或跳过，如果你已经有了）
  Step 2: 提取 BERT 内部特征
  Step 3: 人工校准 500 条数据（你需要手动做的）
  Step 4: 训练 Colander
  Step 5: 全量扫描，产出每条数据的置信度

使用方法：
  python colander_full.py step1    # 训练 baseline BERT
  python colander_full.py step2    # 提取特征
  # --- 手动标注 calibration_500.csv ---
  python colander_full.py step3    # 训练 Colander
  python colander_full.py step4    # 全量扫描
"""

import sys
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.model_selection import train_test_split

# ============================================================
# 配置
# ============================================================
DATA_PATH = "../data/训练数据集.csv"
MODEL_NAME = "bert-base-chinese"
# 你已有的三分类 BERT（跳过 step1）
EXISTING_MODEL_PATH = "../best_model_3class"
BATCH_SIZE = 32
MAX_LEN = 128
CALIBRATION_SIZE = 500
OUTPUT_DIR = "./colander_output"
FEATURE_PATH = os.path.join(OUTPUT_DIR, "features.npy")
CALIB_PATH = os.path.join(OUTPUT_DIR, "calibration_500.csv")
SCAN_PATH = os.path.join(OUTPUT_DIR, "all_confidence.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 标签映射：LLM 原始标注 → 三分类（与 bert_3class.py 保持一致）
#   0 = 中性
#   1 = 负面情绪（焦虑/愤怒/自卑/委屈/无助/孤独/抑郁）
#   2 = 正面情绪
# ============================================================
POSITIVE_WORDS = {"正面"}
NEGATIVE_WORDS = {"焦虑", "愤怒", "自卑", "委屈", "无助", "孤独", "抑郁", "不赞同"}


def map_label(row) -> int:
    """根据 隐含情绪 和 情绪类型 映射到三分类标签"""
    emotion = str(row.get("隐含情绪", "")).strip()
    emo_type = str(row.get("情绪类型", "")).strip()

    if emotion in ("", "nan"):
        # 显性情绪但没有细分 → 根据 label 字段粗判
        if str(row.get("label", 0)) == "1":
            return 1  # 有情绪但没细分，保守归负面
        return 0

    if emotion == "中性":
        return 0

    if emotion in POSITIVE_WORDS:
        return 2

    parts = set(p.strip() for p in emotion.replace("，", "/").split("/"))
    has_positive = bool(parts & POSITIVE_WORDS)
    has_negative = bool(parts & NEGATIVE_WORDS)

    if has_negative:
        return 1  # 含负面（含负面+正面混合）→ 负面优先
    elif has_positive:
        return 2
    else:
        return 0  # 未知标签 → 中性


# ============================================================
# Step 1: 训练 Baseline BERT
# ============================================================
def step1_train_baseline():
    """三分类 BERT：0=中性 1=有情绪"""
    print("=" * 50)
    print("Step 1: 训练 Baseline BERT")
    print("=" * 50)

    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df.dropna(subset=["Sentence", "label"])
    df["label"] = df["label"].astype(int)

    texts = df["Sentence"].tolist()
    labels = df["label"].tolist()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.15, random_state=42
    )

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    class SCAYSDataset(Dataset):
        def __init__(self, texts, labels):
            self.encodings = tokenizer(
                texts, truncation=True, padding="max_length",
                max_length=MAX_LEN, return_tensors="pt"
            )
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                "input_ids": self.encodings["input_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx],
                "labels": self.labels[idx],
            }

    train_dataset = SCAYSDataset(train_texts, train_labels)
    val_dataset = SCAYSDataset(val_texts, val_labels)

    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )

    training_args = TrainingArguments(
        output_dir=MODEL_PATH,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        load_best_model_at_end=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    trainer.save_model(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)

    val_result = trainer.evaluate()
    print(f"\n验证集准确率: {val_result.get('eval_accuracy', 'N/A')}")
    print(f"模型保存至: {MODEL_PATH}")


# ============================================================
# Step 2: 提取 BERT 最后两层的内部特征 + 模型预测
# ============================================================
def step2_extract_features():
    """提取所有句子的倒数两层 CLS 特征"""
    print("=" * 50)
    print("Step 2: 提取 BERT 内部特征")
    print("=" * 50)

    # 加载你已有的三分类 BERT
    if os.path.exists(EXISTING_MODEL_PATH):
        model = BertForSequenceClassification.from_pretrained(
            EXISTING_MODEL_PATH, output_hidden_states=True
        )
        tokenizer = BertTokenizer.from_pretrained(EXISTING_MODEL_PATH)
        print(f"从已有模型加载: {EXISTING_MODEL_PATH}")
    else:
        print(f"找不到模型 {EXISTING_MODEL_PATH}，重新训练...")
        step1_train_baseline()
        model = BertForSequenceClassification.from_pretrained(
            MODEL_PATH, output_hidden_states=True
        )
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"使用设备: {device}")

    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df.dropna(subset=["Sentence", "label"])
    texts = df["Sentence"].tolist()

    all_features = []
    all_predictions = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i : i + BATCH_SIZE]
        encodings = tokenizer(
            batch_texts, truncation=True, padding="max_length",
            max_length=MAX_LEN, return_tensors="pt"
        )
        encodings = {k: v.to(device) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = model(**encodings)

        # 取倒数第一层和倒数第二层的 CLS 向量
        layer_minus2 = outputs.hidden_states[-2][:, 0, :]  # [batch, 768]
        layer_minus1 = outputs.hidden_states[-1][:, 0, :]  # [batch, 768]
        features = torch.cat([layer_minus2, layer_minus1], dim=1)  # [batch, 1536]

        all_features.append(features.cpu().numpy())

        # 同时记录 BERT 的预测
        preds = outputs.logits.argmax(dim=1).cpu().numpy()
        all_predictions.extend(preds.tolist())

        if (i // BATCH_SIZE) % 50 == 0:
            print(f"  处理中... {i}/{len(texts)}")

    features_array = np.concatenate(all_features, axis=0)
    np.save(FEATURE_PATH, features_array)

    # 保存预测结果到 CSV，用于与你原有的 LLM 标签对比
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df.dropna(subset=["Sentence", "label"])
    df["bert_prediction"] = all_predictions
    df.to_csv(os.path.join(OUTPUT_DIR, "predictions_bert.csv"), index=False)

    print(f"特征矩阵: {features_array.shape}")
    print(f"保存至: {FEATURE_PATH}")
    print(f"\nBERT 预测结果保存至: {os.path.join(OUTPUT_DIR, 'predictions_bert.csv')}")


# ============================================================
# Step 3: 准备校准数据 & 训练 Colander
# ============================================================
class Colander(nn.Module):
    """两层小神经网络"""
    def __init__(self, input_dim=1536, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def step3_train_colander():
    """用校准数据训练 Colander"""
    print("=" * 50)
    print("Step 3: 训练 Colander")
    print("=" * 50)

    if not os.path.exists(CALIB_PATH):
        print(f"\n⚠️ 校准文件不存在: {CALIB_PATH}")
        print("请先手动标注校准数据！")
        print("\n操作步骤：")
        print("  1. 打开 colander_output/predictions_bert.csv")
        print("  2. 复制出前 500 行")
        print("  3. 增加一列 'is_correct'，逐条判断 LLM 的标签对不对")
        print("     (1=标对了, 0=标错了)")
        print("  4. 保存为 colander_output/calibration_500.csv")
        print("\n或者运行如下命令生成校准模板：")
        print("  python colander_full.py gen_calib")
        return

    # 加载特征
    features = np.load(FEATURE_PATH)
    cal_df = pd.read_csv(CALIB_PATH)

    # 检查 is_correct 是否填好
    cal_df["is_correct"] = cal_df["is_correct"].astype(str).str.strip()
    empty_mask = ~cal_df["is_correct"].isin(["0", "1"])
    if empty_mask.any():
        print(f"⚠️ 有 {empty_mask.sum()} 条 is_correct 没填或填错（必须是 0 或 1）")
        print(f"未填的行号（0-indexed）: {cal_df[empty_mask].index.tolist()[:20]}")
        print("请先填完这些行再跑 step3")
        return

    # 用 original_index 列定位到原始 features 的位置
    if "original_index" in cal_df.columns:
        cal_indices = cal_df["original_index"].values
        print(f"使用 original_index 列定位原始特征")
    else:
        cal_indices = cal_df.index.values
        print(f"未找到 original_index，使用默认 index（可能是错的）")

    cal_features = features[cal_indices]
    cal_labels = cal_df["is_correct"].astype(int).values.astype(np.float32)

    print(f"校准数据量: {len(cal_df)}")
    print(f"标对的比例: {cal_labels.mean():.1%}")

    # 收窄序列
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Colander(input_dim=features.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    X = torch.tensor(cal_features, dtype=torch.float32).to(device)
    y = torch.tensor(cal_labels, dtype=torch.float32).unsqueeze(1).to(device)

    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            acc = ((pred > 0.5).float() == y).float().mean().item()
            print(f"  Epoch {epoch:3d} | Loss: {loss.item():.4f} | Acc: {acc:.3f}")

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "colander.pt"))
    print(f"\nColander 模型保存至: {os.path.join(OUTPUT_DIR, 'colander.pt')}")


# ============================================================
# Step 4: 全量扫描
# ============================================================
def step4_scan_all():
    """用 Colander 扫描全量数据，输出每条的置信度"""
    print("=" * 50)
    print("Step 4: 全量扫描")
    print("=" * 50)

    features = np.load(FEATURE_PATH)
    cola_path = os.path.join(OUTPUT_DIR, "colander.pt")
    model = Colander(input_dim=features.shape[1])
    model.load_state_dict(torch.load(cola_path))
    model.eval()

    # 扫描
    X = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        confidences = model(X).squeeze().numpy()

    # 保存结果
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df.dropna(subset=["Sentence", "label"])
    df["colander_confidence"] = confidences
    df.to_csv(SCAN_PATH, index=False)

    print(f"保存至: {SCAN_PATH}")
    print(f"\n置信度统计：")
    print(f"  平均: {confidences.mean():.3f}")
    print(f"  中位数: {np.median(confidences):.3f}")
    print(f"  最小值: {confidences.min():.3f}")
    print(f"  最大值: {confidences.max():.3f}")

    # 按阈值推荐审核量
    for threshold in [0.3, 0.4, 0.5, 0.6]:
        suspicious = (confidences < threshold).sum()
        print(f"  置信度 < {threshold:.1f}: {suspicious:>5} 条可疑 (占 {suspicious/len(confidences)*100:.1f}%)")


# ============================================================
# 生成校准模板
# ============================================================
def gen_calib_template():
    """生成 500 条数据让你人工标对错"""
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "predictions_bert.csv"), low_memory=False)

    # 用三分类标签（与 bert_3class.py 一致）：LLM 的 label_3 vs BERT 的 bert_prediction
    if "label_3" not in df.columns:
        df["label_3"] = df.apply(map_label, axis=1)
    df["disagree"] = (df["label_3"] != df["bert_prediction"]).astype(int)

    # 尽量均匀采样：一半 BERT 和 LLM 一致的，一半不一致的
    agree_samples = df[df["disagree"] == 0].sample(250, random_state=42)
    disagree_samples = df[df["disagree"] == 1].sample(min(250, df["disagree"].sum()), random_state=42)

    calib = pd.concat([agree_samples, disagree_samples]).sample(frac=1, random_state=42).reset_index(drop=False)

    # 干净改名，并加入语义化的列名方便人工判断
    calib["is_correct"] = ""
    label_names = {0: "中性", 1: "负面", 2: "正面"}
    calib["LLM三分类标签"] = calib["label_3"].map(label_names)
    calib["BERT三分类预测"] = calib["bert_prediction"].map(label_names)
    calib = calib[["index", "Sentence", "label_3", "LLM三分类标签",
                   "bert_prediction", "BERT三分类预测", "情绪类型", "隐含情绪", "is_correct"]]
    calib.rename(columns={
        "label_3": "LLM_label(0/1/2)",
        "index": "original_index"
    }, inplace=True)

    calib.to_csv(CALIB_PATH, index=False, encoding="utf-8-sig")
    print(f"校准模板已生成: {CALIB_PATH}")
    print(f"共 {len(calib)} 条（{len(agree_samples)} 条一致 + {len(disagree_samples)} 条不一致）")
    print("\n请打开此文件，逐条填写 is_correct 列：1=LLM标对了, 0=LLM标错了")
    print("填写完成后，重新运行：python colander_full.py step3")


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("\n可用命令：step1, step2, gen_calib, step3, step4")
    elif sys.argv[1] == "step1":
        step1_train_baseline()
    elif sys.argv[1] == "step2":
        step2_extract_features()
    elif sys.argv[1] == "gen_calib":
        gen_calib_template()
    elif sys.argv[1] == "step3":
        step3_train_colander()
    elif sys.argv[1] == "step4":
        step4_scan_all()
    else:
        print(f"未知命令: {sys.argv[1]}")
