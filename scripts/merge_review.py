"""
人工审核后，将审核结果合并回 invisible_labeled.csv，并重新生成训练数据集
使用方法: python3 merge_review.py
"""

import pandas as pd
import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def merge():
    labeled_file = os.path.join(DATA_DIR, "invisible_labeled.csv")
    review_file = os.path.join(DATA_DIR, "invisible_需人工审核.csv")

    if not os.path.exists(review_file):
        print("❌ 找不到审核文件，请先完成人工审核")
        return

    df_labeled = pd.read_csv(labeled_file)
    df_review = pd.read_csv(review_file)

    print(f"标注数据: {len(df_labeled)} 条")
    print(f"审核数据: {len(df_review)} 条")

    # 把审核结果覆盖回标注数据
    # 用 (Note_ID, Sentence) 作为匹配键
    review_dict = {}
    for _, row in df_review.iterrows():
        key = (str(row["Note_ID"]), str(row["Sentence"]))
        review_dict[key] = {
            "隐含情绪": row["隐含情绪"],
            "情绪动机": row["情绪动机"],
            "置信度": row["置信度"],
        }

    updated = 0
    for idx, row in df_labeled.iterrows():
        key = (str(row["Note_ID"]), str(row["Sentence"]))
        if key in review_dict:
            df_labeled.at[idx, "隐含情绪"] = review_dict[key]["隐含情绪"]
            df_labeled.at[idx, "情绪动机"] = review_dict[key]["情绪动机"]
            df_labeled.at[idx, "置信度"] = review_dict[key]["置信度"]
            updated += 1

    print(f"已更新: {updated} 条")

    # 保存合并后的标注文件
    df_labeled.to_csv(labeled_file, index=False, encoding="utf-8-sig")
    print(f"✅ 已更新 {labeled_file}")

    # ---- 重新生成训练数据集 ----
    normal = pd.read_csv(os.path.join(DATA_DIR, "normal.csv"))
    normal["label"] = 1
    normal["情绪类型"] = "显性情绪"

    inv_emotion = df_labeled[df_labeled["隐含情绪"] != "中性"].copy()
    inv_emotion["label"] = 1
    inv_emotion["情绪类型"] = "隐含情绪"

    inv_neutral = df_labeled[df_labeled["隐含情绪"] == "中性"].copy()
    inv_neutral["label"] = 0
    inv_neutral["情绪类型"] = "中性"

    train_data = pd.concat([normal, inv_emotion, inv_neutral], ignore_index=True)
    train_data = train_data[["Keyword", "Note_ID", "Source", "Sentence", "label", "情绪类型",
                              "隐含情绪", "情绪动机", "置信度"]]

    # 完整版
    train_path = os.path.join(DATA_DIR, "训练数据集.csv")
    train_data.to_csv(train_path, index=False, encoding="utf-8-sig")

    # BERT精简版
    simple = train_data[["Sentence", "label"]].rename(columns={"Sentence": "raw_text"})
    simple_path = os.path.join(DATA_DIR, "bert训练集.csv")
    simple.to_csv(simple_path, index=False, encoding="utf-8-sig")

    print(f"\n📊 合并后统计:")
    print(f"  label=1 (焦虑/负面): {(train_data['label']==1).sum()} 条")
    print(f"  label=0 (正常):      {(train_data['label']==0).sum()} 条")
    print(f"\n✅ 训练数据集 → {train_path}")
    print(f"✅ BERT训练集 → {simple_path}")


if __name__ == "__main__":
    merge()
