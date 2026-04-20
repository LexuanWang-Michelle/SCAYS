from DrissionPage import ChromiumPage
import time
import random
import os
import csv

# ==========================================
# 1. 配置区
# ==========================================
KEYWORDS = [
    "学渣", "学霸", "卷王", "小透明", "班干部"
]

MAX_NOTES_PER_KEYWORD = 20
OUTPUT_FILE = "xhs_identity_corpus.csv"


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


# ==========================================
# 2. 核心爬虫模块
# ==========================================
def run_spider():
    print("启动小红书身份语料爬虫...")

    try:
        page = ChromiumPage()
    except Exception as e:
        print("启动失败。请关闭所有正在运行的 Google Chrome 窗口再试。")
        return

    print("正在访问小红书首页...")
    page.get("https://www.xiaohongshu.com/explore")
    print("等待 15 秒确保登录态加载完毕（若是首次请扫码登录）...")
    time.sleep(15)

    for kw in KEYWORDS:
        print(f"\n========== 开始搜索：【{kw}】 ==========")
        search_url = f'https://www.xiaohongshu.com/search_result?keyword={kw}&source=web_search_result_notes'
        page.get(search_url)
        time.sleep(random.uniform(4, 6))

        notes_processed = 0
        already_clicked_ids = set()
        scroll_attempts = 0
        max_scrolls = 30
        consecutive_fail = 0

        while notes_processed < MAX_NOTES_PER_KEYWORD and scroll_attempts < max_scrolls:

            # ====================================================
            # 核心改动：用 CSS 选择器精确定位搜索结果中的帖子卡片
            # 小红书搜索页的每个帖子卡片是 class 包含 "note-item" 的元素
            # ====================================================
            cards = page.eles('css:section.note-item') or page.eles('css:div.note-item') or page.eles('css:[data-note-id]')

            if not cards:
                # 备用：尝试用更宽泛的方式找卡片
                cards = page.eles('css:.feeds-page section a')

            target_card = None
            target_id = None

            for card in cards:
                try:
                    # 尝试获取帖子唯一标识
                    note_id = card.attr('data-note-id') or ''
                    if not note_id:
                        # 从内部链接中提取 ID
                        inner_a = card.ele('tag:a', timeout=0.5)
                        if inner_a:
                            href = inner_a.link or ''
                            if '/explore/' in href:
                                note_id = href.split('/explore/')[-1].split('?')[0]
                        if not note_id:
                            href = card.link or ''
                            if '/explore/' in href:
                                note_id = href.split('/explore/')[-1].split('?')[0]

                    if not note_id or note_id in already_clicked_ids:
                        continue

                    # 确保卡片在屏幕上可见且有真实尺寸
                    w, h = card.rect.size
                    if w > 50 and h > 50:
                        target_card = card
                        target_id = note_id
                        break
                except:
                    continue

            if target_card and target_id:
                already_clicked_ids.add(target_id)
                scroll_attempts = 0  # 找到了就重置滚动计数
                print(f"  👆 点击帖子卡片 [{target_id[:8]}...]")

                try:
                    # 滚动到卡片可见位置
                    target_card.scroll.to_see(center=True)
                    time.sleep(random.uniform(0.8, 1.5))

                    # 记录点击前的 URL
                    url_before = page.url

                    # 真实物理鼠标点击
                    page.actions.move_to(target_card).wait(random.uniform(0.3, 0.8)).click()
                    time.sleep(random.uniform(3, 5))

                    # ====================================================
                    # 核心改动：小红书搜索页点击帖子后是弹出浮层弹窗
                    # URL 会变成 /explore/xxxxx 但页面不跳转
                    # 弹窗内容直接在当前页面的 DOM 中
                    # ====================================================

                    # 检查是否被风控拦截
                    if "404" in page.url or page.ele("text:当前笔记暂时无法浏览", timeout=2):
                        print(f"  被风控拦截！等待 30 秒后继续...")
                        # 按 ESC 或后退关掉
                        page.actions.key_down('Escape').key_up('Escape')
                        time.sleep(2)
                        if "404" in page.url:
                            page.back()
                            time.sleep(3)
                        time.sleep(30)  # 被拦截后等久一点
                        consecutive_fail += 1
                        if consecutive_fail >= 3:
                            print("  连续 3 次被拦截，跳过该关键词")
                            break
                        continue

                    consecutive_fail = 0  # 没被拦截，重置计数

                    # 等待弹窗内容加载（尝试多种可能的选择器）
                    note_detail = (
                        page.ele('css:.note-detail-mask', timeout=3) or
                        page.ele('css:#noteContainer', timeout=2) or
                        page.ele('css:.note-container', timeout=2) or
                        page.ele('css:.content-container', timeout=2)
                    )

                    # 提取标题
                    title_el = (
                        page.ele('#detail-title', timeout=2) or
                        page.ele('css:.title', timeout=1) or
                        page.ele('css:h1', timeout=1)
                    )
                    title = title_el.text.strip() if title_el else "无标题"

                    # 提取正文
                    desc_el = (
                        page.ele('#detail-desc', timeout=2) or
                        page.ele('css:.desc', timeout=1) or
                        page.ele('css:.note-text', timeout=1)
                    )
                    content = desc_el.text.strip() if desc_el else "无正文"

                    # 模拟真人阅读：在弹窗内滚动
                    for _ in range(3):
                        page.scroll.down(500)
                        time.sleep(random.uniform(1, 2))

                    # 点击"展开"加载更多评论
                    for _ in range(5):
                        expand_btn = page.ele('text:展开', timeout=1.5)
                        if expand_btn:
                            try:
                                expand_btn.click()
                                time.sleep(random.uniform(1.5, 2.5))
                            except:
                                break
                        else:
                            break

                    # 抓取评论
                    comments = []
                    comment_nodes = page.eles('css:.comment-item') or page.eles('css:.comment-inner')
                    for node in comment_nodes:
                        try:
                            c_text_node = node.ele('css:.content', timeout=0.5) or node.ele('css:.note-text', timeout=0.5)
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

                    notes_processed += 1
                    print(f"  └─  ({notes_processed}/{MAX_NOTES_PER_KEYWORD}) {title[:15]}... | 评论: {len(comments)}")

                    # ====================================================
                    # 核心改动：关闭弹窗回到搜索列表
                    # 方法1: 点击弹窗外的遮罩层
                    # 方法2: 点击关闭按钮
                    # 方法3: 按 ESC
                    # 方法4: 浏览器后退
                    # ====================================================
                    closed = False

                    # 尝试点击关闭按钮
                    close_btn = (
                        page.ele('css:.close-circle', timeout=1) or
                        page.ele('css:.note-detail-mask .close', timeout=1) or
                        page.ele('css:svg.close', timeout=1) or
                        page.ele('css:[aria-label="close"]', timeout=1)
                    )
                    if close_btn:
                        try:
                            close_btn.click()
                            closed = True
                        except:
                            pass

                    if not closed:
                        # 按 ESC 键关闭弹窗
                        page.actions.key_down('Escape').key_up('Escape')

                    time.sleep(random.uniform(2, 4))

                    # 确认回到了搜索页
                    if 'search_result' not in page.url:
                        page.back()
                        time.sleep(2)

                except Exception as e:
                    print(f"  └─ 处理帖子出错: {e}")
                    # 无论出什么错，都确保回到搜索页
                    try:
                        page.actions.key_down('Escape').key_up('Escape')
                        time.sleep(1)
                        if 'search_result' not in page.url:
                            page.back()
                            time.sleep(2)
                    except:
                        pass
            else:
                # 没找到未处理的卡片，向下滚动
                print(f"  向下滚动加载更多... (当前已完成 {notes_processed} 篇)")
                page.scroll.down(800)
                time.sleep(random.uniform(3, 5))
                scroll_attempts += 1

    print("\n 全部抓取完毕！")
    page.quit()


if __name__ == "__main__":
    run_spider()
