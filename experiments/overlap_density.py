"""
重叠密度分析 (Overlap Density Analysis)
========================================
基于 Sala et al. (ICLR 2025) 的重叠密度理论，
分析 SCAYS 各维度数据中"显性情绪信号与隐性情绪信号同时出现"的比例。

重叠密度 = 同时具备关键词信号（easy feature）和情境语义信号（hard feature）的数据占比。
高重叠密度 → 关键词采集命中率高、废料少。

使用方法：
  python3 overlap_density.py
"""

import pandas as pd
import os
import re

# ============================================================
# 配置
# ============================================================
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "训练数据集.csv")

# 情绪关键词列表（显性信号）
EMOTION_KEYWORDS = [
    # 负面
    "焦虑", "崩溃", "难过", "抑郁", "绝望", "痛苦", "害怕", "恐惧",
    "愤怒", "生气", "委屈", "无助", "孤独", "自卑", "烦", "累",
    "哭", "泪", "emo", "破防", "裂开", "窒息", "受不了", "想死",
    "不想活", "寄了", "废了", "完蛋", "崩了", "麻了", "烦死",
    # 正面
    "开心", "快乐", "幸福", "激动", "兴奋", "期待", "喜欢", "爱",
    "感动", "骄傲", "自豪", "满足", "舒服", "爽", "太好了", "耶",
]


def compute_overlap_density(df, dimension_name="整体"):
    """
    计算重叠密度：在有情绪的数据中，有多少同时包含情绪关键词。
    
    有情绪 = label != 0（即 label=1 负面 或 label=2 正面）
    包含关键词 = 句子中含有 EMOTION_KEYWORDS 中的任一词
    
    重叠密度 = (有情绪 且 含关键词) / (有情绪)
    """
    # 筛选有情绪的数据
    emotional = df[df["label"] != 0].copy()
    total_emotional = len(emotional)
    
    if total_emotional == 0:
        return 0.0, 0, 0
    
    # 检查是否包含情绪关键词
    pattern = "|".join(re.escape(kw) for kw in EMOTION_KEYWORDS)
    has_keyword = emotional["Sentence"].astype(str).str.contains(pattern, na=False)
    overlap_count = has_keyword.sum()
    
    density = overlap_count / total_emotional
    return density, overlap_count, total_emotional


def main():
    print("=" * 60)
    print("SCAYS 重叠密度分析 (Overlap Density)")
    print("=" * 60)
    
    # 加载数据
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"\n总数据量: {len(df)} 条")
    print(f"有情绪: {(df['label'] != 0).sum()} 条")
    print(f"中性: {(df['label'] == 0).sum()} 条")
    
    # 整体重叠密度
    density, overlap, total = compute_overlap_density(df)
    print(f"\n{'=' * 40}")
    print(f"整体重叠密度: {density:.1%} ({overlap}/{total})")
    print(f"{'=' * 40}")
    
    # 按维度（Keyword 所属维度）分析
    # 根据关键词判断维度
    identity_keywords = [
        "准高三", "准初三", "高三党", "初三党", "复读生",
        "住校生", "住宿生", "走读生", "理科生", "文科生",
        "美术生", "体育生", "学渣", "学霸", "卷王", "小透明", "班干部"
    ]
    event_keywords = [
        "月考", "周测", "摸底考", "开学考", "一模", "二模", "百日誓师",
        "发成绩单", "晚自习", "早读", "跑操", "课间操", "拖堂", "占课",
        "大扫除", "没收手机", "剪头发", "查宿舍", "请家长", "留堂",
        "写检讨", "全校通报", "补作业", "赶作业", "开学综合征", "补课",
        "高考出分", "高考失利", "出分焦虑"
    ]
    relationship_keywords = [
        "偏心", "查手机", "道德绑架", "重男轻重", "原生家庭",
        "爸妈吵架", "留守", "单亲家庭", "亲子关系", "被针对",
        "班主任", "老师偏心", "被罚", "办公室谈话",
        "被孤立", "校园冷暴力", "被排挤", "没有朋友"
    ]
    
    dimensions = {
        "身份锚点": identity_keywords,
        "周期节律": event_keywords,
        "关系场域": relationship_keywords,
    }
    
    print(f"\n{'维度':<10} {'重叠密度':<10} {'重叠数':<8} {'有情绪总数':<10}")
    print("-" * 45)
    
    for dim_name, keywords in dimensions.items():
        dim_df = df[df["Keyword"].isin(keywords)]
        if len(dim_df) == 0:
            print(f"{dim_name:<10} {'无数据':<10}")
            continue
        density, overlap, total = compute_overlap_density(dim_df, dim_name)
        print(f"{dim_name:<10} {density:<10.1%} {overlap:<8} {total:<10}")
    
    # 按关键词细分（Top 10 高/低密度）
    print(f"\n{'=' * 40}")
    print("按关键词细分（重叠密度 Top/Bottom）")
    print(f"{'=' * 40}")
    
    keyword_results = []
    for kw in df["Keyword"].unique():
        kw_df = df[df["Keyword"] == kw]
        d, o, t = compute_overlap_density(kw_df)
        if t >= 50:  # 至少 50 条有情绪数据才有统计意义
            keyword_results.append({"关键词": kw, "重叠密度": d, "有情绪数": t})
    
    kw_results = pd.DataFrame(keyword_results).sort_values("重叠密度", ascending=False)
    
    print("\n重叠密度最高的 10 个关键词（规则最有效）:")
    for _, r in kw_results.head(10).iterrows():
        print(f"  {r['关键词']:<12} {r['重叠密度']:.1%}  (n={r['有情绪数']})")
    
    print("\n重叠密度最低的 10 个关键词（规则最无效，需要 LLM）:")
    for _, r in kw_results.tail(10).iterrows():
        print(f"  {r['关键词']:<12} {r['重叠密度']:.1%}  (n={r['有情绪数']})")
    
    print("\n" + "=" * 60)
    print("结论：重叠密度越低的维度/关键词，越依赖 LLM 或人工标注。")
    print("=" * 60)


if __name__ == "__main__":
    main()
