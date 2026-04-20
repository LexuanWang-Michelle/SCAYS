"""
清洗 normal.csv 和 invisible.csv 中的用户名
- 删除 "回复 xxx :" 中的用户名 → "回复："
- 删除 @xxx 中的用户名

使用方法: python3 remove_usernames.py
"""

import pandas as pd
import re
import os

DATA_DIR = "/Users/yangchao/Desktop/ai"
FILES = [
    os.path.join(DATA_DIR, "normal.csv"),
    os.path.join(DATA_DIR, "invisible.csv"),
]


def remove_usernames(sentence):
    """清洗一句话中的用户名"""
    s = str(sentence)

    # 1. 删除 "回复 xxx :" / "回复 xxx:" 中的用户名 → "回复："
    s = re.sub(r'回复\s+\S+\s*[:：]', '回复：', s)

    # 2. 删除 @用户名
    # 用户名可能含：中英日文、数字、.·_、括号、emoji等
    # 匹配策略：@ 后面取到空格或字符串结尾
    # 但要区分 "@用户名 实际内容" 和 "@用户名@下一个用户名"
    # 先处理连续@：@A@B@C → 全删
    # 再处理 @用户名 后跟空格+内容 → 只删@用户名，保留内容
    s = re.sub(r'@[\w\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff．.·\-_%✨🎧😡💭]+', '', s)

    # 3. 清理多余空格和标点
    s = re.sub(r'\s+', ' ', s).strip()
    # 清理 "回复：" 后面可能多余的空格
    s = re.sub(r'回复：\s+', '回复：', s)

    return s


def main():
    for filepath in FILES:
        filename = os.path.basename(filepath)
        df = pd.read_csv(filepath)
        print(f"\n{'='*40}")
        print(f"📄 {filename}: {len(df)} 条")

        # 统计含用户名的句子
        reply_mask = df['Sentence'].astype(str).str.contains(r'回复\s+\S+\s*[:：]')
        at_mask = df['Sentence'].astype(str).str.contains(r'@')
        print(f"  含'回复xxx:': {reply_mask.sum()} 条")
        print(f"  含'@xxx':     {at_mask.sum()} 条")

        # 清洗
        df['Sentence'] = df['Sentence'].astype(str).apply(remove_usernames)

        # 清洗后可能产生空句，去掉
        empty_before = len(df)
        df = df[df['Sentence'].str.strip() != '']
        empty_removed = empty_before - len(df)
        if empty_removed > 0:
            print(f"  清洗后空句删除: {empty_removed} 条")

        # 保存
        df.to_csv(filepath, index=False, encoding="utf-8-sig")
        print(f"  ✅ 已保存: {filepath}")

        # 抽样展示
        print(f"\n  清洗样例:")
        samples = df[df['Sentence'].str.contains('回复：|回复：')].head(3)
        for _, row in samples.iterrows():
            print(f"    {row['Sentence'][:80]}")


if __name__ == "__main__":
    main()
