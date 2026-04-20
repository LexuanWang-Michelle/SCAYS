"""
invisible.csv 隐含情绪标注脚本

基于规则引擎对 invisible 中的句子进行：
  1. 隐含情绪类别标注（焦虑/压力/悲伤/愤怒/无助/自卑/委屈/孤独/正面/中性）
  2. 情绪动机标注（学业/人际/自我/家庭/未来/生活/身体/其他）
  3. 置信度标注（high/medium/low）— low 的需要人工审核

使用方法: python3 label_invisible.py
"""

import pandas as pd
import re
import os

DATA_DIR = "/Users/yangchao/Desktop/ai"
INPUT_FILE = os.path.join(DATA_DIR, "invisible.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "invisible_labeled.csv")
REVIEW_FILE = os.path.join(DATA_DIR, "invisible_需人工审核.csv")

# ============ 隐含情绪识别规则 ============

# 每个规则: (情绪类别, 动机, 正则模式, 置信度)
# 置信度: high=明确隐含, medium=较可能, low=需人工判断

EMOTION_RULES = [
    # --- 学业压力/焦虑（隐含） ---
    ("焦虑", "学业", r"还有.{0,4}天就", "high"),
    ("焦虑", "学业", r"只剩.{0,4}天", "high"),
    ("焦虑", "学业", r"距离.{0,4}还有", "high"),
    ("焦虑", "学业", r"倒计时", "high"),
    ("焦虑", "学业", r"马上就要.{0,4}考", "high"),
    ("焦虑", "学业", r"来不及", "high"),
    ("焦虑", "学业", r"还来得及", "high"),
    ("焦虑", "学业", r"提分|涨分|冲刺|逆袭|突击", "high"),
    ("焦虑", "学业", r"基础.{0,4}薄弱|基础.{0,4}差", "high"),
    ("焦虑", "学业", r"偏科", "high"),
    ("焦虑", "学业", r"不及格|没及格|挂科", "high"),
    ("焦虑", "学业", r"跟不上|听不懂|学不会", "high"),
    ("焦虑", "学业", r"复习.{0,6}不完|刷题.{0,6}不完", "high"),
    ("焦虑", "学业", r"时间.{0,4}不够|时间.{0,4}紧", "high"),
    ("焦虑", "学业", r"成绩.{0,4}下滑|成绩.{0,4}掉", "high"),
    ("焦虑", "学业", r"一模|二模|三模|期中|期末|月考|中考|高考", "medium"),
    ("焦虑", "学业", r"排名|名次|倒数", "medium"),
    ("焦虑", "学业", r"错题|薄弱|漏洞|短板", "medium"),
    ("焦虑", "学业", r"还有.{0,6}没有[学复]完", "medium"),
    ("焦虑", "学业", r"怎么.{0,6}提高|怎么.{0,6}提分|怎么.{0,6}学", "medium"),
    ("焦虑", "学业", r"有没有.{0,6}方法|有没有.{0,6}建议", "low"),
    ("焦虑", "学业", r"复习.{0,4}怎么", "low"),

    # --- 无助/迷茫（隐含） ---
    ("无助", "学业", r"不知道.{0,6}从哪.{0,4}开始", "high"),
    ("无助", "学业", r"不知道.{0,6}怎么办", "high"),
    ("无助", "学业", r"无从下手", "high"),
    ("无助", "自我", r"不知道.{0,6}选", "medium"),
    ("无助", "未来", r"不知道.{0,6}以后|不知道.{0,6}未来", "medium"),
    ("无助", "学业", r"找不到.{0,6}方向|没有.{0,6}方向", "medium"),
    ("无助", "学业", r"学了.{0,4}还是.{0,4}(不会|不行|没用)", "high"),
    ("无助", "学业", r"怎么.{0,6}也.{0,6}(不会|不行|提高)", "high"),
    ("无助", "自我", r"迷茫", "high"),

    # --- 自我否定/自卑（隐含） ---
    ("自卑", "自我", r"比我.{0,6}(好|强|厉害|优秀)", "high"),
    ("自卑", "自我", r"别人.{0,6}(都|也).{0,6}(会|能|行)", "high"),
    ("自卑", "自我", r"就我.{0,6}(不会|不行|差)", "high"),
    ("自卑", "自我", r"只有我.{0,6}(不会|不行|差)", "high"),
    ("自卑", "自我", r"天赋.{0,4}(差|不行|没有)", "high"),
    ("自卑", "自我", r"童子功|天赋怪|经济怪", "high"),
    ("自卑", "学业", r"才.{0,4}(分|名)", "high"),  # 才400分
    ("自卑", "学业", r"只能考.{0,4}(分|名)", "high"),
    ("自卑", "自我", r"比不上|不如|赶不上", "high"),
    ("自卑", "自我", r"怎么学都.{0,4}(不|没)", "medium"),
    ("自卑", "学业", r"我.{0,4}(分|名次).{0,4}(低|差|少)", "medium"),

    # --- 委屈/不满（隐含） ---
    ("委屈", "人际", r"不让我|不让穿|不让带|不让用|不让回", "high"),
    ("委屈", "学校", r"学校.{0,6}(不让|不准|禁止|必须)", "high"),
    ("委屈", "学校", r"老师.{0,6}(不让|不准|骂|批评|罚)", "high"),
    ("委屈", "学校", r"宿管.{0,6}(不让|不准|骂|凶)", "high"),
    ("委屈", "人际", r"被.{0,4}(骂|说|排挤|孤立|忽略|无视)", "high"),
    ("委屈", "人际", r"凭什么", "high"),
    ("委屈", "学校", r"还要.{0,6}(上|写|做|交)", "medium"),
    ("委屈", "学校", r"又要.{0,6}(上|写|做|交)", "medium"),
    ("委屈", "人际", r"为什么.{0,6}(我|总是|老是)", "medium"),
    ("委屈", "学校", r"请假.{0,4}(不给|不批|不请)", "medium"),
    ("委屈", "生活", r"都不给.{0,4}(报|买|吃)", "medium"),

    # --- 愤怒/不满（隐含） ---
    ("愤怒", "人际", r"有没有病|有病吧|脑子有病", "high"),
    ("愤怒", "社会", r"这不就是.{0,6}吗", "high"),
    ("愤怒", "人际", r"真烦人|真恶心|真受不了", "high"),
    ("愤怒", "社会", r"搞什么|什么鬼|什么破", "high"),
    ("愤怒", "人际", r"你管得着|关你什么事|关你啥事", "high"),
    ("愤怒", "社会", r"无语|服了|真的服", "medium"),
    ("愤怒", "社会", r"离谱|太离谱|好离谱", "medium"),
    ("愤怒", "人际", r"笑死|笑死我了|太好笑", "low"),

    # --- 孤独（隐含） ---
    ("孤独", "人际", r"一个人.{0,6}(住|待|吃|走|学|睡)", "high"),
    ("孤独", "人际", r"没有.{0,6}(朋友|人陪|人聊)", "high"),
    ("孤独", "人际", r"没人.{0,6}(理|管|说|陪|找)", "high"),
    ("孤独", "人际", r"融不进|融入不了|合不来", "high"),
    ("孤独", "人际", r"孤零零|孤伶伶", "high"),
    ("孤独", "人际", r"交不到.{0,4}(朋友|对象)", "medium"),
    ("孤独", "人际", r"没有.{0,6}说话", "medium"),

    # --- 身体/生活困扰（隐含负面） ---
    ("焦虑", "身体", r"失眠|睡不着|醒太早|睡不好|熬夜", "high"),
    ("焦虑", "身体", r"头疼|头痛|胃痛|肚子痛|腰痛|颈椎", "high"),
    ("焦虑", "身体", r"感冒.{0,4}(不|没|还)", "high"),
    ("焦虑", "身体", r"发炎|过敏|生病", "high"),
    ("焦虑", "生活", r"吃不好|睡不好|住不好", "high"),
    ("焦虑", "生活", r"噪音|吵|吵死|隔壁.{0,4}(吵|闹)", "high"),
    ("焦虑", "生活", r"热水.{0,4}(没有|不够|不热)", "high"),
    ("焦虑", "生活", r"空调.{0,4}(没有|坏了|不)", "medium"),
    ("焦虑", "生活", r"食堂.{0,4}(难吃|不好|贵)", "medium"),

    # --- 正面情绪（隐含） ---
    ("正面", "学业", r"提分|涨了.{0,4}(分|名)", "high"),
    ("正面", "学业", r"上岸|录取|考上了|过了", "high"),
    ("正面", "生活", r"好爽|好幸福|好开心|好快乐|好舒服", "high"),
    ("正面", "生活", r"太棒了|太好了|太香了", "high"),
    ("正面", "人际", r"室友.{0,6}(好|不错|人好|友善)", "high"),
    ("正面", "生活", r"还不错|也还行|挺好的", "medium"),
    ("正面", "学业", r"加油|冲|努力|拼搏", "low"),

    # --- 家庭压力（隐含） ---
    ("焦虑", "家庭", r"我妈|我爸|家长.{0,6}(让|要|逼|催|管)", "high"),
    ("焦虑", "家庭", r"家里.{0,6}(不让|不给|要求|必须)", "high"),
    ("委屈", "家庭", r"父母.{0,6}(不|不让|不理解)", "high"),
    ("焦虑", "家庭", r"报班|补课|请家教", "medium"),
    ("委屈", "家庭", r"不给.{0,6}(钱|买|报)", "medium"),

    # --- 宿舍/人际冲突（隐含） ---
    ("委屈", "人际", r"室友.{0,6}(吵|闹|脏|乱|不|偷|用)", "high"),
    ("愤怒", "人际", r"室友.{0,6}(过分|过分了|太过分)", "high"),
    ("焦虑", "人际", r"关系.{0,6}(紧张|不好|差|僵)", "high"),
    ("委屈", "人际", r"被.{0,4}(说|讲|议论|嘲笑|看不起)", "high"),
    ("焦虑", "人际", r"社交|社恐|不敢.{0,6}(说话|交流|主动)", "high"),

    # --- 对未来的担忧（隐含） ---
    ("焦虑", "未来", r"就业|找工作|毕业.{0,4}(怎么办|之后)", "high"),
    ("焦虑", "未来", r"考不上.{0,6}(怎么办|就)", "high"),
    ("焦虑", "未来", r"前途|出路|以后.{0,6}(怎么办|做啥)", "high"),
    ("焦虑", "未来", r"没有.{0,6}(出路|前途|希望)", "high"),
    ("无助", "未来", r"不知道.{0,6}(干嘛|做什么|能干嘛)", "medium"),

    # --- 自嘲/调侃（隐含负面） ---
    ("自卑", "自我", r"牛马|打工人|韭菜|炮灰|分母|陪跑", "high"),
    ("自卑", "自我", r"废物|垃圾|菜鸡|学渣", "high"),
    ("自卑", "自我", r"躺平|摆烂|摆了|开摆", "high"),
    ("自卑", "学业", r"重在参与|佛系|随缘", "medium"),
    ("自卑", "自我", r"摆烂|摆了|开摆", "high"),

    # --- 语气词隐含情绪 ---
    ("焦虑", "学业", r"救命", "high"),
    ("无助", "学业", r"求[救帮助推荐分享]", "high"),
    ("愤怒", "其他", r"真的服了|无语死了|我醉了", "medium"),
]

# 编译正则
COMPILED_RULES = []
for emotion, motive, pattern, confidence in EMOTION_RULES:
    try:
        COMPILED_RULES.append((emotion, motive, re.compile(pattern), confidence))
    except:
        pass


def label_sentence(sentence):
    """
    对一句话进行隐含情绪标注
    返回: (情绪列表, 动机列表, 置信度)
    """
    s = str(sentence)
    emotions = []
    motives = []
    max_confidence = "low"

    for emotion, motive, pattern, confidence in COMPILED_RULES:
        if pattern.search(s):
            if emotion not in emotions:
                emotions.append(emotion)
            if motive not in motives:
                motives.append(motive)
            if confidence == "high":
                max_confidence = "high"
            elif confidence == "medium" and max_confidence == "low":
                max_confidence = "medium"

    if not emotions:
        return "中性", "无", "low"

    return "/".join(emotions), "/".join(motives), max_confidence


def main():
    print("=" * 50)
    print("🏷️  invisible.csv 隐含情绪标注")
    print("=" * 50)

    df = pd.read_csv(INPUT_FILE)
    print(f"原始数据: {len(df)} 条\n")

    # 标注
    results = df["Sentence"].astype(str).apply(label_sentence)
    df["隐含情绪"] = results.apply(lambda x: x[0])
    df["情绪动机"] = results.apply(lambda x: x[1])
    df["置信度"] = results.apply(lambda x: x[2])

    # 统计
    print("📊 标注统计:")
    print(f"\n隐含情绪分布:")
    print(df["隐含情绪"].value_counts().to_string())
    print(f"\n情绪动机分布:")
    print(df["情绪动机"].value_counts().to_string())
    print(f"\n置信度分布:")
    print(df["置信度"].value_counts().to_string())

    # 保存全部标注结果
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n✅ 全部标注结果 → {OUTPUT_FILE}")

    # 单独保存需人工审核的（置信度=low 且非中性）
    review_df = df[(df["置信度"] == "low") & (df["隐含情绪"] != "中性")]
    neutral_df = df[df["隐含情绪"] == "中性"]

    print(f"\n📋 标注结果分类:")
    print(f"  高/中置信度标注: {len(df[df['置信度'] != 'low'])} 条")
    print(f"  低置信度需审核: {len(review_df)} 条")
    print(f"  中性（无隐含情绪）: {len(neutral_df)} 条")

    if len(review_df) > 0:
        review_df.to_csv(REVIEW_FILE, index=False, encoding="utf-8-sig")
        print(f"\n✅ 需人工审核 → {REVIEW_FILE}")

    # 样例展示
    print(f"\n{'='*50}")
    print("📋 各类别样例:")

    for emotion in ["焦虑", "无助", "自卑", "委屈", "愤怒", "孤独", "正面"]:
        subset = df[df["隐含情绪"].str.contains(emotion, na=False)]
        if len(subset) > 0:
            print(f"\n  【{emotion}】 (共{len(subset)}条):")
            for _, row in subset.head(3).iterrows():
                print(f"    [{row['置信度']}] {row['Sentence'][:70]}")

    print(f"\n  【中性】 (共{len(neutral_df)}条):")
    for _, row in neutral_df.head(3).iterrows():
        print(f"    {row['Sentence'][:70]}")


if __name__ == "__main__":
    main()
