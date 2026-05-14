from DrissionPage import ChromiumPage
import time
import random
import os
import csv

# ==========================================
# 1. 配置区
# ==========================================
KEYWORDS = [
    "体育生",  # 上次只爬了5条，补完
    "学渣", "学霸", "卷王", "小透明", "班干部"
]

MAX_NOTES_PER_KEYWORD = 20
OUTPUT_FILE = "/Users/yangchao/Desktop/ai/xhs_identity_corpus.csv"

# 反检测时间参数
MIN_BETWEEN_NOTES = 6
MAX_BETWEEN_NOTES = 14
LONG_REST_EVERY = 5
LONG_REST_MIN = 25
LONG_REST_MAX = 50
KEYWORD_REST_MIN = 35
KEYWORD_REST_MAX = 70


# ==========================================
# 2. 工具函数
# ==========================================
def save_to_csv(data, filename=OUTPUT_FILE):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Keyword", "Note_ID", "Title", "Content", "Comments"])
        writer.writerow([
            data.get('keyword', ''), data.get('note_id', ''),
            data.get('title', ''), data.get('content', ''), data.get('comments', '')
        ])


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


def extract_note_id(href):
    """
    从链接中提取 note_id
    新格式: /search_result/67d84351000000001c00786d?xsec_token=...
    旧格式: /explore/67d84351000000001c00786d
    """
    if not href:
        return None
    for pattern in ['/search_result/', '/explore/']:
        if pattern in href:
            part = href.split(pattern)[-1]
            note_id = part.split('?')[0].split('/')[0]
            if len(note_id) >= 10:  # note_id 一般至少10位hex
                return note_id
    return None


# ==========================================
# 3. 核心爬虫
# ==========================================
def run_spider():
    print("🚀 启动小红书身份语料爬虫（v3 - 适配新版页面结构）...")

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

    # 首页预热
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

        search_url = f'https://www.xiaohongshu.com/search_result?keyword={kw}&source=web_search_result_notes'
        page.get(search_url)
        human_delay(5, 8)

        # 滑一滑加载内容
        slow_scroll_down(page, times=random.randint(1, 2))
        time.sleep(random.uniform(1, 3))

        notes_processed = 0
        already_processed_ids = set()
        scroll_attempts = 0
        max_scrolls = 30
        consecutive_fail = 0

        while notes_processed < MAX_NOTES_PER_KEYWORD and scroll_attempts < max_scrolls:

            # ================================================
            # 核心改动：用 css:a.cover 定位帖子封面链接
            # 新版小红书搜索页：
            #   <a class="cover" href="/search_result/NOTE_ID?xsec_token=...">
            # ================================================
            cover_links = page.eles('css:a.cover')

            # 备选：如果 a.cover 找不到，用 a.title
            if not cover_links:
                cover_links = page.eles('css:a.title')

            # 再备选：从 .feeds-page 里找
            if not cover_links:
                cover_links = page.eles('css:.feeds-page section a')

            target_el = None
            target_id = None

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

                    # 确保元素有真实尺寸且可见
                    w, h = el.rect.size
                    if w > 20 and h > 20:
                        target_el = el
                        target_id = note_id
                        break
                except:
                    continue

            if target_el:
                already_processed_ids.add(target_id)
                scroll_attempts = 0

                print(f"  👆 点击帖子 [{target_id[:12]}...]")

                try:
                    tab_ids_before = page.tab_ids

                    # 滚动到可见位置
                    target_el.scroll.to_see(center=True)
                    time.sleep(random.uniform(0.8, 2.0))

                    # 移动鼠标 + 停顿 + 点击
                    page.actions.move_to(target_el).wait(random.uniform(0.5, 1.5)).click()
                    human_delay(4, 8)

                    tab_ids_after = page.tab_ids

                    # 判断新标签页 or 弹窗
                    is_new_tab = False
                    if len(tab_ids_after) > len(tab_ids_before):
                        new_tab_id = [t for t in tab_ids_after if t not in tab_ids_before][0]
                        work_tab = page.get_tab(new_tab_id)
                        is_new_tab = True
                    else:
                        work_tab = page

                    # 风控检测
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

                    # 模拟阅读
                    simulate_reading(check_page)

                    # 提取标题
                    title_el = check_page.ele('#detail-title', timeout=2) or check_page.ele('.title', timeout=2) or check_page.ele('css:h1', timeout=1)
                    title = title_el.text.strip() if title_el else "无标题"

                    # 提取正文
                    desc_el = check_page.ele('#detail-desc', timeout=2) or check_page.ele('.desc', timeout=2) or check_page.ele('css:.note-text', timeout=1)
                    content = desc_el.text.strip() if desc_el else "无正文"

                    # 点击"展开"
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

                    # 抓取评论
                    comments = []
                    comment_nodes = check_page.eles('.comment-item') or check_page.eles('css:.comment-inner')
                    for node in comment_nodes:
                        try:
                            c_text_node = node.ele('.content', timeout=0.5) or node.ele('css:.note-text', timeout=0.5)
                            if c_text_node:
                                c_text = c_text_node.text.strip()
                                if c_text:
                                    comments.append(c_text)
                        except:
                            continue

                    data = {
                        "keyword": kw, "note_id": target_id,
                        "title": title, "content": content,
                        "comments": " | ".join(comments)
                    }
                    save_to_csv(data)
                    already_crawled.add(target_id)

                    notes_processed += 1
                    total_this_session += 1
                    print(f"  └─ ✅ ({notes_processed}/{MAX_NOTES_PER_KEYWORD}) {title[:20]}... | 评论: {len(comments)}")

                    # 关闭帖子
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

                    # 帖子间延迟
                    human_delay(MIN_BETWEEN_NOTES, MAX_BETWEEN_NOTES)

                    # 确认回到搜索页
                    if 'search_result' not in page.url:
                        page.back()
                        time.sleep(random.uniform(2, 4))

                    # 每 N 篇长休息
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
                print(f"  📜 向下滚动加载更多... (已完成 {notes_processed} 篇)")
                slow_scroll_down(page, times=random.randint(2, 3))
                time.sleep(random.uniform(2, 4))
                scroll_attempts += 1

        print(f"\n  📊 【{kw}】完成，本轮爬取 {notes_processed} 篇")

    print(f"\n🎉 全部抓取完毕！本次共爬取 {total_this_session} 篇")
    print(f"📁 数据保存在: {OUTPUT_FILE}")
    page.quit()


if __name__ == "__main__":
    run_spider()
