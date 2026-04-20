"""
小红书语料库：去广告 + 情绪分类
输出两个文件：
  - normal.csv   → 含情绪词汇的句子（情绪外显）
  - invisible.csv → 无情绪词汇的句子（情绪隐匿/中性）

使用方法: python3 filter_emotion.py
"""

import pandas as pd
import re
import os

# ============ 配置 ============
DATA_DIR = "/Users/yangchao/Desktop/ai"
INPUT_FILE = os.path.join(DATA_DIR, "xhs_sentences_corpus.csv")
NORMAL_FILE = os.path.join(DATA_DIR, "normal.csv")
INVISIBLE_FILE = os.path.join(DATA_DIR, "invisible.csv")


# ============ 第一步：去广告 ============
def is_ad(sentence):
    """判断是否为广告/营销/社交水贴"""
    s = sentence.strip()

    # 1. 明确的广告/营销话术
    ad_keywords = [
        '后台滴滴', '私信我', '私信发', '私信获取', '私聊', '私我',
        '加微', '加V', '加Q', '微信号', 'QQ群', '群号',
        '无小号', '无助理', '马扁', '小心被马扁',
        '1对1训练课', '一对一课', '陪学弟学妹',
        '关注私建议', '提分方法笔记已备好',
        '个性化提升计划', '无偿.*诊断',
        '约稿', '接稿', '接单', '出物', '收物',
        '闲鱼', '淘宝下单', '优惠码',
    ]
    for kw in ad_keywords:
        if re.search(kw, s):
            return True

    # 2. "已关注+求方法/初几" 模式（典型的求资料水贴）
    if re.match(r'^已[关官]➕?[初高]', s):
        return True
    if re.match(r'^已[关官][，,]?\s*(初|高)', s):
        return True
    if re.match(r'^已[关官]注?\s*[，,]?\s*(初|高)', s):
        return True
    if re.match(r'^已[关官]注.*求方法', s):
        return True

    # 3. "看下💌(无小号" 模式
    if re.search(r'看下.*无小号|无小号.*无助理', s):
        return True

    # 4. 极短社交水贴（≤6字且无实质内容）
    short_social = ['求方法', '求分享', '求推荐', '来啦', '好滴', '冲',
                    '加油', '码住', '蹲', '同', '同求', '求带', '求教']
    if len(s) <= 6 and s in short_social:
        return True

    # 5. "好的，后台滴滴你辣哦" 之类的引流回复
    if re.search(r'后台.*滴滴|滴滴.*看下', s):
        return True

    return False


def is_empty_reply(sentence):
    """判断是否为空的回复碎片（'回复 xxx :' 后面没有实质内容）"""
    s = sentence.strip()
    m = re.match(r'^回复\s+\S+\s*[:：]\s*(.*)', s)
    if m:
        rest = m.group(1).strip()
        # 回复后面内容太短（≤3字），视为碎片
        if len(rest) <= 3:
            return True
    return False


# ============ 第二步：情绪词分类 ============

# 情绪词表（覆盖青少年常见负面/正面情绪表达）
EMOTION_WORDS = {
    # --- 焦虑/压力 ---
    '焦虑', '紧张', '害怕', '担心', '恐惧', '慌', '慌张', '心慌', '慌了',
    '压力', '压抑', '崩溃', '窒息', '喘不过气', '喘不过',
    '烦', '烦死', '烦死了', '好烦', '很烦', '太烦', '超烦',
    '累', '好累', '很累', '太累', '累死', '累死了', '疲惫', '疲倦', '心力交瘁',
    '怕', '好怕', '很怕', '害怕', '恐惧', '畏缩', '不敢',
    '急', '着急', '焦急', '急死', '急死了',

    # --- 悲伤/低落 ---
    '难过', '伤心', '悲伤', '痛苦', '心酸', '心碎', '心寒',
    '哭', '哭了', '想哭', '好想哭', '泪', '落泪', '流泪', '泪目', '抹泪',
    '绝望', '崩溃', '崩溃了', '崩溃边缘', '万念俱灰',
    '低落', '消沉', '沮丧', '颓废', '丧', '很丧', '好丧',
    '抑郁', '郁闷', '郁结', '抑郁了',

    # --- 愤怒/不满 ---
    '气', '生气', '气愤', '愤怒', '恼火', '火大', '气死', '气死了',
    '恨', '讨厌', '厌恶', '恶心', '烦人', '可恶', '该死',
    '骂', '怒', '怒了', '暴怒', '炸了', '气炸',

    # --- 无助/迷茫 ---
    '迷茫', '困惑', '无助', '无奈', '没办法', '不知所措',
    '不知道怎么办', '咋办', '怎么办', '怎么办呀', '不知咋办',
    '没方向', '没有方向', '找不着方向', '看不到希望', '看不到头',
    '不知道怎么学', '不知道怎么学', '不会做', '不会写',

    # --- 自我否定/自卑 ---
    '废物', '垃圾', '没用', '没用的', '没救了', '没救',
    '笨', '太笨', '好笨', '蠢', '太蠢', '好蠢', '愚蠢',
    '差', '太差', '好差', '很差', '差劲', '太差劲',
    '不行', '我不行', '做不到', '做不好', '学不会',
    '考不上', '完蛋', '完蛋了', '完了', '完了完了',
    '放弃', '想放弃', '放弃了', '摆烂', '躺平', '摆了',
    '学渣', '差生', '后进生', '垫底',

    # --- 委屈/不甘 ---
    '委屈', '不甘', '不甘心', '不公平', '凭什么',
    '被骂', '被批评', '被训', '被罚',

    # --- 孤独/寂寞 ---
    '孤独', '寂寞', '孤单', '一个人', '没人', '没有朋友',
    '被孤立', '被排挤', '小透明',

    # --- 正面情绪（也保留） ---
    '开心', '高兴', '快乐', '幸福', '满足', '欣慰',
    '激动', '兴奋', '期待', '盼望', '希望',
    '感动', '温暖', '感恩', '感谢', '谢谢',
    '骄傲', '自豪', '自信', '勇敢', '坚强',
    '喜欢', '爱', '热爱', '珍惜', '幸福',

    # --- 常见情绪短语/口语 ---
    '撑不住', '扛不住', '受不了', '坚持不住', '坚持不了',
    '想哭', '想逃', '想跑', '想回家', '想休息',
    '不想学', '不想去', '不想上', '不想考', '不想努力',
    '累了', '够了', '真的够了', '撑不下去',
    '心累', '心塞', '心烦', '头疼', '头痛',
    '受不了了', '崩溃了', '炸了', '疯了', '要疯了',
    '太难了', '好难', '太难', '太难了吧', '太难熬',
    '熬不下去', '熬不住', '熬不过', '熬不了',
    '好绝望', '绝望了', '没希望', '没指望',
    '想死', '死了算了', '不想活', '活不下去',
    '自闭', '自闭了', '社恐', '社恐了',
    'emo', 'emo了', '破防', '破防了',
    '无语', '无语了', '醉了', '醉了醉了',
    '羡慕', '嫉妒', '眼红', '酸了', '酸',
    '后悔', '遗憾', '可惜', '悔恨',

    # --- 感叹词/语气词（表达强烈情绪的） ---
    '救命', '天哪', '我的天', '妈呀', '哇', '啊这',
    '呜呜', '嘤嘤', '555', '😭', '🥹', '😢', '😤', '😡', '😩', '😫',
    '💔', '😞', '😔', '🥺', '😖', '😣',
}

# 情绪正则模式（匹配"太X了"、"好X"、"X死"等情绪句式）
EMOTION_PATTERNS = [
    r'太.{0,4}了',          # 太累了/太难了/太烦了
    r'好.{0,2}[累烦怕难慌急气委屈]',  # 好累/好烦/好怕
    r'[累烦怕气急].*死.*了',   # 累死了/烦死了/气死了
    r'真.{0,2}[累烦怕难]',    # 真累/真烦/真怕
    r'想[哭逃跑回休息放弃]',   # 想哭/想逃/想放弃
    r'不[想愿敢] ',          # 不想/不愿/不敢
    r'怎么.{0,6}[办啊呀呢]',  # 怎么办/怎么办啊
    r'撑[不]住',            # 撑不住
    r'受[不]了',            # 受不了
    r'扛[不]住',            # 扛不住
    r'熬[不].*[去下过]',     # 熬不下去/熬不过
    r'考[不]上',            # 考不上
    r'完[蛋了]',            # 完蛋/完了
    r'没[有]?[希望救]',      # 没希望/没救
    r'坚[持][不]住',        # 坚持不住
    r'想[要]?[死放弃]',      # 想死/想放弃
]

emotion_pattern_re = re.compile('|'.join(EMOTION_PATTERNS))


def has_emotion(sentence):
    """判断句子是否包含情绪表达"""
    s = str(sentence)

    # 1. 检查是否包含情绪词
    for word in EMOTION_WORDS:
        if word in s:
            return True

    # 2. 检查是否匹配情绪句式
    if emotion_pattern_re.search(s):
        return True

    return False


# ============ 主流程 ============
def main():
    print("=" * 50)
    print("🧹 小红书语料库：去广告 + 情绪分类")
    print("=" * 50)

    # 加载数据
    df = pd.read_csv(INPUT_FILE)
    print(f"原始数据: {len(df)} 条\n")

    # --- 去广告 ---
    ad_mask = df['Sentence'].astype(str).apply(is_ad)
    empty_reply_mask = df['Sentence'].astype(str).apply(is_empty_reply)
    remove_mask = ad_mask | empty_reply_mask

    print(f"广告/营销: {ad_mask.sum()} 条")
    print(f"空回复碎片: {empty_reply_mask.sum()} 条")
    print(f"合计去除: {remove_mask.sum()} 条")

    df_clean = df[~remove_mask].copy()
    print(f"清洗后剩余: {len(df_clean)} 条\n")

    # --- 情绪分类 ---
    emotion_mask = df_clean['Sentence'].astype(str).apply(has_emotion)

    df_normal = df_clean[emotion_mask].copy()
    df_invisible = df_clean[~emotion_mask].copy()

    print(f"📊 情绪分类结果:")
    print(f"  normal   (含情绪词): {len(df_normal)} 条")
    print(f"  invisible (无情绪词): {len(df_invisible)} 条")

    # 保存
    df_normal.to_csv(NORMAL_FILE, index=False, encoding="utf-8-sig")
    df_invisible.to_csv(INVISIBLE_FILE, index=False, encoding="utf-8-sig")
    print(f"\n✅ normal   → {NORMAL_FILE}")
    print(f"✅ invisible → {INVISIBLE_FILE}")

    # --- 抽样展示 ---
    print(f"\n{'='*50}")
    print("📋 normal 样例（含情绪词）:")
    for i, row in df_normal.sample(min(10, len(df_normal)), random_state=42).iterrows():
        print(f"  [{row['Keyword']}] {row['Sentence'][:70]}")

    print(f"\n📋 invisible 样例（无情绪词）:")
    for i, row in df_invisible.sample(min(10, len(df_invisible)), random_state=42).iterrows():
        print(f"  [{row['Keyword']}] {row['Sentence'][:70]}")


if __name__ == "__main__":
    main()
