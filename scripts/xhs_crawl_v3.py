from DrissionPage import ChromiumPage
import time
import random
import os
import csv
from datetime import date

# ==========================================
# 1. 配置区
# ==========================================
KEYWORDS = [
    # --- 高考热点（最新优先） ---
    "高考出分", "高考失利", "出分焦虑",
    # --- 维度2：周期节律 - 日常场景 ---
    "早读", "跑操", "课间操", "拖堂", "占课", "大扫除",
    # --- 维度2：周期节律 - 纪律惩罚 ---
    "没收手机", "剪头发", "查宿舍", "请家长", "留堂", "写检讨", "全校通报",
    # --- 维度2：周期节律 - 学业压力 ---
    "补作业", "赶作业", "开学综合征", "补课",
]

MAX_NOTES_PER_KEYWORD = 30
OUTPUT_FILE = f"/Users/yangchao/Desktop/ai/xhs_corpus_v3_{date.today().strftime('%Y%m%d')}.csv"

MIN_BETWEEN_NOTES = 6
MAX_BETWEEN_NOTES = 14
LONG_REST_EVERY = 5
LONG_REST_MIN = 25
LONG_REST_MAX = 50
KEYWORD_REST_MIN = 35
KEYWORD_REST_MAX = 70


def save_to_csv(data, filename=OUTPUT_FILE):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Keyword", "Note_ID", "Title", "Content", "Comment"])
        kw = data.get('keyword', '')
        note_id = data.get('note_id', '')
        title = data.get('title', '')
        content = data.get('content', '')
        comments = data.get('comments_list', [])
        if comments:
            for comment in comments:
                writer.writerow([kw, note_id, title, content, comment])
        else:
            writer.writerow([kw, note_id, title, content, ''])


def get_already_crawled_ids(filename=OUTPUT_FILE):
    ids = set()
    if os.path.isfile(filename):
        with open(filename, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                nid = row.get('Note_ID', '').strip()
                if nid:
                    ids.add(nid)
    return ids


def human_delay(min_s=3, max_s=7):
    delay = random.uniform(min_s, max_s)
    if random.random() < 0.10:
        delay += random.uniform(3, 8)
    time.sleep(delay)


def slow_scroll_down(page, times=3):
    for _ in range(times):
        page.scroll.down(random.randint(250, 600))
        time.sleep(random.uniform(1.0, 2.5))


def simulate_reading(page):
    time.sleep(random.uniform(1.5, 3.0))
    for _ in range(random.randint(2, 4)):
        page.scroll.down(random.randint(200, 500))
        time.sleep(random.uniform(1.5, 3.0))
    if random.random() < 0.25:
        page.scroll.up(random.randint(100, 300))
        time.sleep(random.uniform(1, 2))
    for _ in range(random.randint(1, 2)):
        page.scroll.down(random.randint(200, 400))
        time.sleep(random.uniform(1, 2))


def is_ad_post(title, content, comments):
    """判断帖子是否为广告/营销帖"""
    ad_patterns = [
        '招生', '报名', '课程', '辅导班', '试听', '免费领', '资料分享',
        '➕', '加v', '加微信', 'vx', '私信', '戳我', '扣我',
        '限时', '名额', '优惠', '甩卖', '清仓', '下单',
        '专业一对一', '名师', '保过', '包过', '提分', '签约',
        '送资料', '免费资料', '电子版', '打印版',
    ]
    fb_patterns = [
        '求推', '求分享', '怎么联系', '怎么报名', 'dd', '扣1',
        '已私', '求拉', '怎么加', '拉我', '举手', '怎么买',
        '私我', '私信我', '求链接',
    ]

    # 标题和正文检查
    text = (title + ' ' + content).lower()
    for pat in ad_patterns:
        if pat in text:
            return True, f"标题/正文含广告词: {pat}"

    # 评论检查：如果大量评论都是"求推"类模式，说明是广告引流帖
    if len(comments) >= 3:
        fb_count = sum(1 for c in comments if any(p in c.lower() for p in fb_patterns))
        ratio = fb_count / len(comments)
        if ratio >= 0.4:  # 超过40%评论是求推类
            return True, f"评论区{ratio:.0%}为求推/问联系方式"

    return False, ""


def extract_note_id(href):
    if not href:
        return None
    for pattern in ['/search_result/', '/explore/']:
        if pattern in href:
            part = href.split(pattern)[-1]
            note_id = part.split('?')[0].split('/')[0]
            if len(note_id) >= 10:
                return note_id
    return None


def find_first_target(page, already_processed_ids, already_crawled):
    cover_links = page.eles('css:a.cover')
    if not cover_links:
        cover_links = page.eles('css:a.title')
    if not cover_links:
        cover_links = page.eles('css:.feeds-page section a')

    total_links = len(cover_links) if cover_links else 0
    candidates = []
    for el in cover_links:
        try:
            href = el.link
            note_id = extract_note_id(href)
            if not note_id:
                continue
            if note_id in already_processed_ids:
                continue
            if note_id in already_crawled:
                already_processed_ids.add(note_id)
                continue
            w, h = el.rect.size
            if w > 20 and h > 20:
                x, y = el.rect.location
                candidates.append((y, x, el, note_id))
        except:
            continue

    if not candidates:
        return None, None, total_links

    candidates.sort(key=lambda item: (item[0], item[1]))
    _, _, best_el, best_id = candidates[0]
    return best_el, best_id, total_links


def run_spider():
    print("🚀 启动爬虫 v3（高考热点 + 周期节律维度补完）...")

    already_crawled = get_already_crawled_ids()
    print(f"📋 已有 {len(already_crawled)} 条历史数据，将跳过重复帖子")

    try:
        page = ChromiumPage()
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        print("   请关闭所有正在运行的 Google Chrome 窗口再试。")
        return

    print("🌐 正在访问小红书首页...")
    page.get("https://www.xiaohongshu.com/explore")
    print("⏳ 等待 18 秒确保登录态加载（首次请扫码登录）...")
    time.sleep(18)

    print("📱 首页预热...")
    slow_scroll_down(page, times=random.randint(2, 4))
    time.sleep(random.uniform(2, 4))

    total_this_session = 0

    for kw_index, kw in enumerate(KEYWORDS):
        print(f"\n{'='*50}")
        print(f"🔍 [{kw_index+1}/{len(KEYWORDS)}] 开始搜索：【{kw}】")
        print(f"{'='*50}")

        if kw_index > 0:
            rest = random.uniform(KEYWORD_REST_MIN, KEYWORD_REST_MAX)
            print(f"  ☕ 切换关键词，休息 {rest:.0f} 秒...")
            time.sleep(rest)

        print(f"  ⏳ 5秒后开始搜索，可手动操作筛选...")
        time.sleep(5)

        search_url = f'https://www.xiaohongshu.com/search_result?keyword={kw}&source=web_search_result_notes'
        page.get(search_url)
        human_delay(5, 8)

        slow_scroll_down(page, times=random.randint(1, 2))
        time.sleep(random.uniform(1, 3))

        notes_processed = 0
        already_processed_ids = set()
        scroll_attempts = 0
        max_scrolls = 30
        consecutive_fail = 0
        no_target_consecutive = 0

        while notes_processed < MAX_NOTES_PER_KEYWORD and scroll_attempts < max_scrolls:
            target_el, target_id, total_links = find_first_target(
                page, already_processed_ids, already_crawled
            )

            if target_el:
                no_target_consecutive = 0
                scroll_attempts = 0
                already_processed_ids.add(target_id)

                print(f"  👆 点击帖子 [{target_id[:12]}...] (页面共{total_links}个)")

                try:
                    tab_ids_before = page.tab_ids

                    target_el.scroll.to_see(center=True)
                    time.sleep(random.uniform(0.8, 2.0))

                    page.actions.move_to(target_el).wait(random.uniform(0.5, 1.5)).click()
                    human_delay(4, 8)

                    tab_ids_after = page.tab_ids

                    is_new_tab = False
                    if len(tab_ids_after) > len(tab_ids_before):
                        new_tab_id = [t for t in tab_ids_after if t not in tab_ids_before][0]
                        work_tab = page.get_tab(new_tab_id)
                        is_new_tab = True
                    else:
                        work_tab = page

                    check_page = work_tab
                    if "404" in check_page.url or check_page.ele("text:当前笔记暂时无法浏览", timeout=2) or check_page.ele("text:扫码查看", timeout=2):
                        print(f"  🚨 被风控拦截！")
                        if is_new_tab:
                            work_tab.close()
                        else:
                            page.actions.key_down('Escape').key_up('Escape')
                            time.sleep(1)

                        consecutive_fail += 1
                        backoff = min(30 * (2 ** (consecutive_fail - 1)), 180)
                        print(f"  ⏳ 第{consecutive_fail}次拦截，等 {backoff} 秒...")
                        time.sleep(backoff)

                        if consecutive_fail >= 3:
                            print("  ⛔ 连续3次拦截，跳过该关键词，长休息2分钟")
                            time.sleep(120)
                            break
                        continue

                    consecutive_fail = 0

                    simulate_reading(check_page)

                    title_el = check_page.ele('#detail-title', timeout=2) or check_page.ele('.title', timeout=2) or check_page.ele('css:h1', timeout=1)
                    title = title_el.text.strip() if title_el else "无标题"

                    desc_el = check_page.ele('#detail-desc', timeout=2) or check_page.ele('.desc', timeout=2) or check_page.ele('css:.note-text', timeout=1)
                    content = desc_el.text.strip() if desc_el else "无正文"

                    for _ in range(random.randint(2, 5)):
                        expand_btn = check_page.ele('text:展开', timeout=1.5)
                        if expand_btn:
                            try:
                                time.sleep(random.uniform(0.8, 1.5))
                                expand_btn.click()
                                time.sleep(random.uniform(1.5, 2.5))
                            except:
                                break
                        else:
                            break

                    comments = []
                    MAX_COMMENTS = 150

                    for load_round in range(5):
                        check_page.scroll.down(random.randint(300, 500))
                        time.sleep(random.uniform(1.0, 1.8))
                        try:
                            more_btn = check_page.ele('text:共', timeout=0.8)
                            if more_btn and '评论' in more_btn.text:
                                pass
                            more_btn = check_page.ele('text:查看更多', timeout=0.8)
                            if more_btn:
                                more_btn.click()
                                time.sleep(random.uniform(1.5, 2.5))
                        except:
                            pass

                    for expand_round in range(3):
                        clicked = 0
                        try:
                            reply_btns = check_page.eles('text:条回复') or []
                            for btn in reply_btns:
                                try:
                                    btn.click()
                                    clicked += 1
                                    time.sleep(random.uniform(0.8, 1.5))
                                except:
                                    continue
                        except:
                            pass
                        if clicked == 0:
                            break
                        time.sleep(random.uniform(1.0, 1.5))

                    comment_nodes = check_page.eles('css:.comment-item') or check_page.eles('css:.comment-inner') or []
                    for node in comment_nodes:
                        try:
                            c_text_node = node.ele('.content', timeout=0.3) or node.ele('css:.note-text', timeout=0.3)
                            if c_text_node:
                                c_text = c_text_node.text.strip()
                                if c_text and c_text not in comments and len(c_text) > 1:
                                    comments.append(c_text)
                                    if len(comments) >= MAX_COMMENTS:
                                        break
                        except:
                            continue

                    data = {
                        "keyword": kw, "note_id": target_id,
                        "title": title, "content": content,
                        "comments_list": comments
                    }

                    # 🚫 广告帖子过滤
                    is_ad, ad_reason = is_ad_post(title, content, comments)
                    if is_ad:
                        print(f"  └─ 🚫 广告帖跳过: {ad_reason}")

                        if is_new_tab:
                            work_tab.close()
                        else:
                            close_btn = (
                                page.ele('css:.close-circle', timeout=1) or
                                page.ele('css:svg.close', timeout=1) or
                                page.ele('.close-box', timeout=1) or
                                page.ele('css:[aria-label="close"]', timeout=1)
                            )
                            if close_btn:
                                try:
                                    close_btn.click()
                                except:
                                    page.actions.key_down('Escape').key_up('Escape')
                            else:
                                page.actions.key_down('Escape').key_up('Escape')
                        continue

                    save_to_csv(data)
                    already_crawled.add(target_id)

                    notes_processed += 1
                    total_this_session += 1
                    print(f"  └─ ✅ ({notes_processed}/{MAX_NOTES_PER_KEYWORD}) {title[:20]}... | 评论: {len(comments)}")

                    if is_new_tab:
                        work_tab.close()
                    else:
                        close_btn = (
                            page.ele('css:.close-circle', timeout=1) or
                            page.ele('css:svg.close', timeout=1) or
                            page.ele('.close-box', timeout=1) or
                            page.ele('css:[aria-label="close"]', timeout=1)
                        )
                        if close_btn:
                            try:
                                close_btn.click()
                            except:
                                page.actions.key_down('Escape').key_up('Escape')
                        else:
                            page.actions.key_down('Escape').key_up('Escape')

                    human_delay(MIN_BETWEEN_NOTES, MAX_BETWEEN_NOTES)

                    if 'search_result' not in page.url:
                        page.back()
                        time.sleep(random.uniform(2, 4))

                    if total_this_session % LONG_REST_EVERY == 0 and total_this_session > 0:
                        rest = random.uniform(LONG_REST_MIN, LONG_REST_MAX)
                        print(f"  ☕ 已连续浏览 {LONG_REST_EVERY} 篇，休息 {rest:.0f} 秒...")
                        time.sleep(rest)

                except Exception as e:
                    print(f"  └─ ⚠️ 处理帖子出错: {e}")
                    try:
                        if is_new_tab:
                            work_tab.close()
                        else:
                            page.actions.key_down('Escape').key_up('Escape')
                    except:
                        pass
                    time.sleep(random.uniform(3, 6))

            else:
                no_target_consecutive += 1
                print(f"  📜 当前视口无新帖子(第{no_target_consecutive}次)，向下滚动... (已完成 {notes_processed} 篇)")
                slow_scroll_down(page, times=random.randint(2, 3))
                time.sleep(random.uniform(2, 4))
                scroll_attempts += 1

        print(f"\n  📊 【{kw}】完成，本轮爬取 {notes_processed} 篇")

    print(f"\n🎉 全部抓取完毕！本次共爬取 {total_this_session} 篇")
    print(f"📁 数据保存在: {OUTPUT_FILE}")
    page.quit()


if __name__ == "__main__":
    run_spider()
