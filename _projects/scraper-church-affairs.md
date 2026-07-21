---
content_id: scraper-church-affairs
language: zh
original_language: zh
reader_paths:
- projects-creation
- faith-spirituality
title_zh: "《教会的事务》抓取器"
title_en: Scraper for Church Affairs
summary_zh: 抓取并整理《教会的事务》在线内容的爬虫项目。
summary_en: A scraper for collecting and organizing the online text of Church Affairs.
title: "Scraper for 教会的事务 (Church Affairs)"
collection: projects
category: scraper
excerpt: "a scraper to extract church affairs content from [ezoe.work](https://ezoe.work/books/3/3007.html)"
permalink: "/games/scraper-church-affairs"
date: 2025-08-25
---

# 教会的事务在线抓取

[原文链接](https://ezoe.work/books/3/3007.html)

[代码链接](https://github.com/XavierOwen/Practicing-simple-spider/blob/main/scraper-church-affairs.py)

## 步骤

1. 从索引页提取所有章节的标题和链接（通过正则匹配 `3007-\d+.html` 模式）
2. 遍历每个章节页面
3. 提取章节标题（通过 `feature-title` 类）并去除"第X篇"前缀
4. 解析页面内容：H3 级别标题（`cn1` 类）、H4 级别副标题（`cn2` 类）和段落文本
5. 清理多余空格和换行符
6. 最终生成一个 **教会的事务.md** 文件

## 脚本演进

### 初始步骤

- 使用 `requests` 请求网页，站点采用 `UTF-8` 编码
- 添加随机延迟（0.5-1.5秒）进行礼貌爬虫
- 用 `BeautifulSoup` 解析 `HTML`

### 索引页解析

- 从 `3007.html` 页面遍历所有 `<a>` 标签
- 通过正则表达式 `3007-\d+\.html` 匹配有效章节链接
- 使用 `urljoin` 统一为绝对 `URL`

### 章节标题提取

- 从每个章节页查找 `feature-title` 类的 `div`
- 使用正则 `第[^篇]+篇\s*` 去除"第X篇"前缀，保留实际标题
- 若无 `feature-title` 则做文本搜索作为备选

### 正文内容解析

页面使用多个 CSS 类来组织内容：

- `cn1` 类 `div`：H3 级别主要标题（如"壹"、"贰"等）
- `cn2` 类 `div`：H4 级别副标题
- `cont` 类 `div`：段落文本容器
- 普通 `div` 或其他容器：段落文本内容

处理逻辑（三层结构）：

1. **顶层节点分类**
   - 遍历 `main` div 的所有子节点
   - 根据类名判断节点类型（标题 vs 段落）

2. **内容容器智能处理**
   - 对于有 `cont` 类子节点的 `div`：逐个处理每个 `cont` 子节点
   - 对于普通 `div`：直接遍历其所有后代节点

3. **文本收集与清理**
   - 对每个段落进行 `NavigableString` 遍历以收集文本
   - 将 `<br>` 标签替换为空格（合并文本流）
   - 使用 `\s+` 正则压缩多个空格为一个
   - 移除行首和行尾空格
   - 在段落间添加双空行以改善 Markdown 格式

### 文本清理

- 使用 `\s+` 正则压缩多个空格为一个
- 移除行首和行尾空格
- 压缩多余的空行（超过2个的换行压缩为2个）

### 汇总与输出

- 在 Markdown 中组织为：
- 一级标题：# 教会的事务
- 二级标题：每个章节的标题（不含"第X篇"前缀）
- 三级标题：章节内的 H3 级标题
- 四级标题：章节内的 H4 级副标题
- 正文 Markdown 内容
- 原文链接（便于回源）

## 脚本函数主要功能

- `fetch_html(url)`：抓取网页，设置 UTF-8 编码和礼貌延迟
- `extract_chapters_from_index(html)`：从索引页提取所有章节标题和链接
- `extract_section_heading_from_start(html)`：提取并清理章节标题（去除"第X篇"前缀）
- `extract_page_content(html)`：解析页面内容，转换 HTML 结构为 Markdown
- `build_book_markdown()`：汇总所有内容并生成完整 Markdown

## 经验

1. **编码处理**：站点为 `UTF-8`，注意与其他站点的编码差异
2. **CSS 类识别**：页面使用 CSS 类来标记不同的内容块，需要准确识别 `cn1`、`cn2`、`cont`、`main` 等类
3. **标题清理**：通过正则表达式去除"第X篇"这样的前缀，保留实际内容
4. **链接匹配**：使用正则表达式 `3007-\d+\.html` 准确匹配有效章节链接
5. **嵌套容器处理**：实际的页面结构可能比较复杂，需要智能检测是否存在 `cont` 类子节点，分别处理
6. **空格处理**：HTML 中的 `<br>` 和多个空格需要统一清理，避免 Markdown 格式混乱；段落间添加双空行以改善可读性
7. **礼貌爬虫**：添加随机延迟防止对服务器的过度请求

<details markdown="1">
<summary>爬虫脚本</summary>

```python
import requests
from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString
from urllib.parse import urljoin
import re
import time
import random

BASE_URL = "https://ezoe.work/books/3/3007.html"
BASE_PATH = "https://ezoe.work/books/3/3007"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

def fetch_html(url: str) -> str:
    """Fetch HTML and handle encoding"""
    r = requests.get(url, headers=HEADERS, timeout=25)
    r.encoding = "utf-8"
    # Polite delay to avoid overwhelming the server
    time.sleep(random.uniform(0.5, 1.5))
    return r.text

def extract_chapters_from_index(html: str) -> list[tuple[str, str]]:
    """Extract chapter titles and links from the index page (3007.html)

    Returns list of tuples: (chapter_title, chapter_url)
    """
    soup = BeautifulSoup(html, "html.parser")
    chapters = []

    # Find all links in the page
    anchors = soup.find_all("a", href=True)

    for a in anchors:
        if not isinstance(a, Tag):
            continue
        href_raw = a.get("href", "")
        if isinstance(href_raw, list):
            # If BeautifulSoup returns a list (e.g., AttributeValueList), join its parts
            href = ' '.join(href_raw)
        else:
            # Otherwise, it's a string, so just use it
            href = href_raw
        href = href.strip()
        text = a.get_text(strip=True)

        # Look for links matching pattern 3007-1.html, 3007-2.html, etc.
        if href and re.match(r"3007-\d+\.html", href):
            full_url = urljoin(BASE_URL, href)
            chapters.append((text, full_url))

    return chapters

def extract_section_heading_from_start(html: str) -> str:
    """Extract the main chapter title (e.g., '教会里的职分') without the 第X篇 prefix

    This appears at the beginning of each chapter page after navigation.
    """
    soup = BeautifulSoup(html, 'html.parser')

    # Find the feature-title div which contains the chapter heading
    feature_title = soup.find('div', class_='feature-title')
    if feature_title:
        text = feature_title.get_text(strip=True)
        # Remove "第X篇" prefix if present (with or without space)
        text = re.sub(r'^第[^篇]+篇\s*', '', text)
        return text.strip()

    # Fallback: search in all text
    text_content = soup.get_text()
    match = re.search(r'第[^篇]+篇\s*([^\n]+)', text_content)
    if match:
        return match.group(1).strip()

    return ""

def extract_page_content(html: str) -> str:
    """Extract main content from a chapter page.

    Converts HTML structure to Markdown:
    - <div class='main'> contains the actual content
    - <div class='cn1'> contains H3 section markers (壹, 贰, etc.)
    - <div class='cn2'> contains H4 subsection markers
    - <div class='cont'> or nested divs contain the content paragraphs
    - Preserves paragraph breaks with proper spacing
    """
    soup = BeautifulSoup(html, "html.parser")

    # Find the main content container
    main_div = soup.find('div', class_='main')
    if not main_div:
        # Fallback to id='c' if main div not found
        main_div = soup.find(id='c')

    if not main_div:
        return ""

    lines: list[str] = []

    # Process all children of the main content div
    for child in main_div.children:
        if not isinstance(child, Tag):
            continue

        # Skip <br> tags
        if child.name == 'br':
            continue

        # Check if this is a cn1 section heading (H3)
        if 'cn1' in child.get('class', []):
            heading_text = child.get_text(strip=True)
            lines.append(f"\n### {heading_text}\n")

        # Check if this is a cn2 subsection heading (H4)
        elif 'cn2' in child.get('class', []):
            heading_text = child.get_text(strip=True)
            lines.append(f"\n#### {heading_text}\n")

        # Check if this is a content paragraph with 'cont' class
        elif 'cont' in child.get('class', []):
            # Extract text content from cont div
            content_parts = []

            for elem in child.descendants:
                if isinstance(elem, NavigableString):
                    text = str(elem).strip()
                    if text:
                        content_parts.append(text)
                elif isinstance(elem, Tag) and elem.name == 'br':
                    content_parts.append(' ')

            # Join and clean the content
            content = ' '.join(content_parts)
            content = re.sub(r'\s+', ' ', content).strip()

            if content:
                lines.append(content)
                lines.append("")  # Add paragraph break
                lines.append("")  # Add extra line break for better separation

        # Otherwise, this is a plain content paragraph
        elif child.name == 'div':
            # This could be a container div (like id='c') with nested cont divs
            # Check if it has cont children
            cont_children = [c for c in child.children if isinstance(c, Tag) and 'cont' in c.get('class', [])]

            if cont_children:
                # Process each cont child separately
                for cont_child in cont_children:
                    content_parts = []
                    for elem in cont_child.descendants:
                        if isinstance(elem, NavigableString):
                            text = str(elem).strip()
                            if text:
                                content_parts.append(text)
                        elif isinstance(elem, Tag) and elem.name == 'br':
                            content_parts.append(' ')

                    # Join and clean the content
                    content = ' '.join(content_parts)
                    content = re.sub(r'\s+', ' ', content).strip()

                    if content:
                        lines.append(content)
                        lines.append("")  # Add paragraph break
                        lines.append("")  # Add extra line break for better separation

            else:
                # Regular div without cont children - process normally
                content_parts = []

                for elem in child.descendants:
                    if isinstance(elem, NavigableString):
                        text = str(elem).strip()
                        if text:
                            content_parts.append(text)
                    elif isinstance(elem, Tag) and elem.name == 'br':
                        content_parts.append(' ')

                # Join and clean the content
                content = ' '.join(content_parts)
                content = re.sub(r'\s+', ' ', content).strip()

                if content:
                    lines.append(content)
                    lines.append("")  # Add paragraph break
                    lines.append("")  # Add extra line break for better separation

    # Join lines and clean up excessive spacing
    md = "\n".join(lines)
    md = re.sub(r"\n{3,}", "\n\n", md).strip()

    return md

def build_book_markdown() -> str:
    """Build the complete markdown book"""
    print("📖 Fetching index page...")
    html_index = fetch_html(BASE_URL)
    chapters = extract_chapters_from_index(html_index)

    print(f"✅ Found {len(chapters)} chapters")

    out = []
    out.append("# 教会的事务\n")

    for idx, (chapter_title, chapter_url) in enumerate(chapters, start=1):
        print(f"📄 Processing chapter {idx}/{len(chapters)}: {chapter_title}")

        # Fetch chapter page
        chapter_html = fetch_html(chapter_url)

        # Extract section heading
        section_heading = extract_section_heading_from_start(chapter_html)
        if section_heading:
            out.append(f"## {section_heading}\n")

        # Extract main content
        content = extract_page_content(chapter_html)

        if content:
            out.append(content + "\n")

        # Add link to original
        out.append(f"[原文链接]({chapter_url})\n")
        out.append("")

    return "\n".join(out).strip() + "\n"

if __name__ == "__main__":
    print("🚀 Starting scraper for '教会的事务'...\n")
    md = build_book_markdown()

    output_file = "教会的事务.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"\n✅ Successfully generated: {output_file}")
```
</details>
