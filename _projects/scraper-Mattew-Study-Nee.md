---
content_id: scraper-matthew-study-notes
language: zh
original_language: zh
reader_paths:
- projects-creation
- faith-spirituality
title_zh: "《马太福音查经记录》抓取器"
title_en: Scraper for Matthew Study Notes
summary_zh: 从倪柝声文集网站抓取《马太福音查经记录》的爬虫项目。
summary_en: A scraper that collects Watchman Nee's Matthew study notes from an online
  archive.
title: "Scraper for 马太福音查经记录 (Matthew Study Notes)"
collection: projects
category: scraper
excerpt: "a scraper to extract Matthew Gospel study notes from [倪柝声文集](http://lightinnj.org)"
permalink: "/games/scraper-mattew-study-nee"
date: 2025-08-25
---

# 马太福音查经记录在线抓取

[原文链接](http://lightinnj.org/%E5%80%AA%E6%9F%9D%E8%81%B2%E6%96%87%E9%9B%86/%E5%80%AA%E6%9F%9D%E8%81%B2%E6%96%87%E9%9B%86%E7%AC%AC%E4%B8%80%E8%BE%91/15%E9%A9%AC%E5%A4%AA%E7%A6%8F%E9%9F%B3%E6%9F%A5%E7%BB%8F%E8%AE%B0%E5%BD%95/%E9%A9%AC%E5%A4%AA%E7%A6%8F%E9%9F%B3%E6%9F%A5%E7%BB%8F%E8%AE%B0%E5%BD%95%E7%9B%AE%E5%BD%95.htm)

[代码链接](https://github.com/XavierOwen/Practicing-simple-spider/blob/main/scraper-Mattew-Study-Nee.py)

## 步骤

1. 从目录页中提取所有章节的标题和链接（通过检查"第"和"章"字符）
2. 遍历每个章节页面
3. 抽取第三个 `<p>` 标签的内容作为正文
4. 将 `<br>` 标签转换为换行符
5. 清理空格和多余的空行
6. 最终生成一个 **马太福音查经记录.md** 文件

## 脚本演进

### 初始步骤

- 使用 `requests` 请求网页，站点采用 `GB18030` 编码
- 用 `BeautifulSoup` 解析 `HTML`，提取章节链接

### 章节链接提取

- 从目录页遍历所有 `<a>` 标签
- 过滤出包含"第"和"章"的文本作为有效章节标题
- 使用 `urljoin` 统一为绝对 `URL`

### 正文抓取与处理

- 在每个章节页中，查找所有 `<p>` 标签
- 过滤出有实际内容的段落
- 优先选择第三个 `<p>`，如果不足三个则选择最长的段落
- 将 `<br>` 标签转换为换行符 `\n`

### 文本清理

- 移除行首和行尾的空格
- 删除空行
- 压缩多余的换行符（超过2个的换行压缩为2个）

### 汇总与输出

- 在 Markdown 中组织为：
- 一级标题：# 马太福音查经记录
- 二级标题：每个章节的标题
- 正文 Markdown 内容
- 原文链接（便于回源）

## 脚本函数主要功能

- `fetch_html(url, encoding="gb18030")`：抓取网页，设置编码和礼貌延迟（0.5-1.2秒）
- `extract_chapter_links(html, base_url)`：从目录页提取所有章节标题和链接
- `extract_content(html)`：提取第三个 `<p>` 标签或最长的段落
- `build_matthew_study_markdown()`：汇总所有内容并生成完整 Markdown

## 经验

1. **编码处理**：站点为 `GB18030`，若不手动指定，解析会乱码
2. **链接识别**：通过检查文本中是否存在"第"和"章"字符来识别有效章节链接
3. **内容定位**：优先取第三个 `<p>`，如不足则改用最长的段落，更鲁棒
4. **礼貌爬虫**：添加随机延迟（0.5-1.2秒）以避免对服务器的过度请求
5. **空格清理**：行首和行尾的各种空白符需要正则清理

<details markdown="1">
<summary>爬虫脚本</summary>

```python
import requests
from bs4 import BeautifulSoup, Tag
from urllib.parse import urljoin
import time
import random

BASE_URL = "http://lightinnj.org/%E5%80%AA%E6%9F%9D%E8%81%B2%E6%96%87%E9%9B%86/%E5%80%AA%E6%9F%9D%E8%81%B2%E6%96%87%E9%9B%86%E7%AC%AC%E4%B8%80%E8%BE%91/15%E9%A9%AC%E5%A4%AA%E7%A6%8F%E9%9F%B3%E6%9F%A5%E7%BB%8F%E8%AE%B0%E5%BD%95/%E9%A9%AC%E5%A4%AA%E7%A6%8F%E9%9F%B3%E6%9F%A5%E7%BB%8F%E8%AE%B0%E5%BD%95%E7%9B%AE%E5%BD%95.htm"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

def fetch_html(url: str, encoding: str = "gb18030") -> str:
    """Fetch HTML with proper encoding and polite delay"""
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.encoding = encoding
        time.sleep(random.uniform(0.5, 1.2))  # Polite crawling
        return r.text
    except Exception as e:
        print(f"❌ Failed to fetch {url}: {e}")
        return ""

def extract_chapter_links(html: str, base_url: str) -> list[tuple[str, str]]:
    """Extract chapter links from the table of contents
    Returns: list of (chapter_title, chapter_url)
    """
    soup = BeautifulSoup(html, "html.parser")
    chapters = []

    # Find all links that contain "第" and "章"
    for a in soup.find_all("a", href=True):
        text = a.get_text(strip=True)
        if "第" in text and "章" in text:
            href = a.get("href")
            if href and not href.startswith("#"):
                full_url = urljoin(base_url, href)
                chapters.append((text, full_url))

    return chapters

def extract_content(html: str) -> str:
    """Extract the third <p> tag content from the nested table structure
    Structure: html>body>div>table>tbody>tr>td>table>tbody>tr>td>p[2] (third p)
    """
    if not html:
        return ""

    soup = BeautifulSoup(html, "html.parser")

    # Navigate through the nested structure
    body = soup.find("body")
    if not body:
        return ""

    # Find the main content div/table structure
    # Try to find all <p> tags and get the third one with substantial content
    all_ps = soup.find_all("p")

    # Filter out empty paragraphs and get the ones with actual content
    content_ps = [p for p in all_ps if p.get_text(strip=True)]

    if len(content_ps) < 3:
        # If less than 3 paragraphs, try to get the longest one
        if content_ps:
            target_p = max(content_ps, key=lambda p: len(p.get_text(strip=True)))
        else:
            return ""
    else:
        # Get the third paragraph with content
        target_p = content_ps[2]

    # Convert to markdown-friendly text
    # Replace <br> with newlines
    for br in target_p.find_all("br"):
        br.replace_with("\n")

    text = target_p.get_text(separator="\n", strip=False)

    # Clean up excessive whitespace and newlines
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]  # Remove empty lines

    return "\n\n".join(lines)

def build_matthew_study_markdown() -> str:
    """Main function to scrape all chapters and build markdown"""
    print("📖 开始抓取马太福音查经记录...")

    # Fetch table of contents
    print(f"📥 Fetching table of contents: {BASE_URL}")
    toc_html = fetch_html(BASE_URL)
    if not toc_html:
        print("❌ Failed to fetch table of contents")
        return ""

    # Extract chapter links
    chapters = extract_chapter_links(toc_html, BASE_URL)
    print(f"✅ Found {len(chapters)} chapters")

    # Build markdown
    md_lines = ["# 马太福音查经记录\n"]

    for idx, (title, url) in enumerate(chapters, start=1):
        print(f"📥 Fetching chapter {idx}/{len(chapters)}: {title}")

        # Fetch chapter content
        chapter_html = fetch_html(url)
        content = extract_content(chapter_html)

        # Add to markdown
        md_lines.append(f"## {title}\n")
        if content:
            md_lines.append(content + "\n")
        else:
            md_lines.append("_（本章未检测到内容）_\n")
            print("❗️ empty content detected")

        md_lines.append("")  # Blank line between chapters

    return "\n".join(md_lines).strip() + "\n"

if __name__ == "__main__":
    markdown_content = build_matthew_study_markdown()

    if markdown_content:
        output_file = "马太福音查经记录.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        print(f"✅ 已生成：{output_file}")
    else:
        print("❌ 生成失败")
```
</details>
