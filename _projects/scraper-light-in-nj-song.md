---
content_id: scraper-song-of-songs
language: zh
original_language: zh
reader_paths:
- projects-creation
- faith-spirituality
title_zh: "《歌中之歌》抓取器"
title_en: Scraper for The Song of Songs
summary_zh: 从倪柝声文集网站抓取《歌中之歌》内容的爬虫项目。
summary_en: A scraper that collects Watchman Nee's The Song of Songs from an online
  archive.
title: "Scraper for 歌中之歌 (Song in the Song)"
collection: projects
category: scraper
excerpt: "a scraper to extract Song in the Song (歌中之歌) content from [倪柝声文集](http://lightinnj.org)"
permalink: "/games/scraper-light-in-nj-song"
date: 2025-08-25
---

# 歌中之歌在线抓取

[原文链接](http://lightinnj.org/%E5%B1%9E%E7%81%B5%E4%B9%A6%E6%8A%A5/004%E8%AF%BB%E7%BB%8F%E7%B1%BB%20%E7%9B%AE%E5%BD%95/4004%E6%AD%8C%E4%B8%AD%E7%9A%84%E6%AD%8C/%E6%AD%8C%E4%B8%AD%E7%9A%84%E6%AD%8C%20%20%E7%9B%AE%E5%BD%95.htm)

[代码链接](https://github.com/XavierOwen/Practicing-simple-spider/blob/main/scraper-light-in-nj-song.py)

## 步骤

1. 从索引页中提取所有小节的标题和链接
2. 遍历每个小节页面
3. 解析内容：`<b>` 标签作为标题（H3或H4）、`<br>` 转换为段落分隔
4. 识别 H4 标题的特殊模式（以简体数字+全角空格开头，如"一　"）
5. 清理重复的开头标题（如果段落首行与小节标题相同则删除）
6. 规范化全角空格和整理格式
7. 最终生成一个 **歌中之歌.md** 文件

## 脚本演进

### 初始步骤

- 使用 `requests` 请求网页，站点采用 `GB18030` 编码
- 添加随机延迟（0.5-1.5秒）进行礼貌爬虫
- 用 `BeautifulSoup` 解析 `HTML`

### 索引页链接提取

- 从索引页遍历所有 `<a>` 标签
- 过滤掉导航链接（如"回首页"）
- 排除相对路径为 `../` 的链接（避免回退到父目录）
- 使用 `urljoin` 统一为绝对 `URL`

### 内容解析的复杂性

该页面结构相对灵活，内容组织方式包括：

1. **简单的标题+段落**：单个 `<b>` 作为标题，后接段落内容
2. **多重标题**：单个 `<p>` 中可能包含多个 `<b>`（如"导言"中的"一..八"）
3. **标题层级混合**：有些 `<b>` 是主标题（H3），有些是副标题（H4）
4. **换行处理**：`<br>` 标签用于段落分隔，需转换为 Markdown 中的 `\n\n`

### 标题层级识别

- **默认 H3**（###）：大多数 `<b>` 标签都作为 H3 处理
- **H4**（####）：如果 `<b>` 的文本以简体数字（一二三四五六七八九十）开头，后跟全角空格（`\u3000`），则归类为 H4

使用正则表达式识别：`r"^[一二三四五六七八九十]+\u3000"`

### 正文抓取与处理

主逻辑：

1. 遍历所有 `<p>` 标签，跳过导航相关（如"回目录"、"书名："前缀）
2. 对每个 `<p>` 的子节点进行逐个处理：
   - `<b>` 标签：作为标题，冲刷缓冲区后输出标题
   - `<br>` 标签：转换为 `\n\n`（Markdown 段落分隔）
   - 其他标签或文本：加入缓冲区
3. 缓冲区文本的清理：
   - 按 `\n\n` 分割成多个段落
   - 每个段落内按 `\n` 分割成行
   - 对每行做正则替换 `[ \t]+` → 单个空格（压缩多余空格）
   - 移除行首行尾空格
   - 删除完全空行

### 重复标题去重

如果段落的首行与小节标题相同（在规范化后比较），则删除该首行。规范化过程包括：

- 替换全角空格 `\u3000` 为普通空格
- 使用正则 `\s+` 压缩多个空白为一个空格
- 去除行首行尾空格
- 去除行首的 Markdown 哈希符号（如果段落内容误被处理为标题语法）

### 汇总与输出

- 在 Markdown 中组织为：
- 一级标题：# 歌中之歌
- 二级标题：每个小节的标题
- 三或四级标题：段落内的标题（取决于是否匹配 H4 模式）
- 正文 Markdown 内容
- 原文链接（便于回源）

后处理：

- 全角空格 `\u3000` → 普通空格
- 中文句号后多余空格清理（`。 +` → `。`）
- 压缩多余空行（3+个连续换行 → 2个）

## 脚本函数主要功能

- `fetch_html(url)`：抓取网页，设置 GB18030 编码和礼貌延迟
- `extract_links_from_index(html, base_url)`：从索引页提取所有小节标题和链接
- `extract_page_content(html)`：解析页面内容，处理 `<b>` 标题和 `<br>` 换行，返回 Markdown
- `dedup_leading_title(content, title)`：去除重复的开头标题行
- `build_book_markdown()`：汇总所有内容并生成完整 Markdown

## 经验

1. **编码处理**：站点为 `GB18030`，注意与其他站点的编码差异
2. **灵活的页面结构**：单个 `<p>` 可能包含多个 `<b>` 标签，需要逐个子节点处理而不是一次性提取
3. **标题层级识别**：通过正则检测简体数字+全角空格的特殊模式来区分 H3 和 H4
4. **换行处理**：`<br>` 在原 HTML 中代表段落分隔，在 Markdown 中应转换为 `\n\n`
5. **重复清理**：段落首行可能重复小节标题，需要通过规范化字符串后比较来去重
6. **空格规范化**：全角空格与普通空格、多余空格的混在一起时容易破坏 Markdown 格式，需统一处理
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

BASE_URL = "http://www.lightinnj.org/%E5%B1%9E%E7%81%B5%E4%B9%A6%E6%8A%A5/004%E8%AF%BB%E7%BB%8F%E7%B1%BB%20%E7%9B%AE%E5%BD%95/4004%E6%AD%8C%E4%B8%AD%E7%9A%84%E6%AD%8C/%E6%AD%8C%E4%B8%AD%E7%9A%84%E6%AD%8C%20%20%E7%9B%AE%E5%BD%95.htm"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

def fetch_html(url: str) -> str:
    """Fetch HTML and handle encoding"""
    r = requests.get(url, headers=HEADERS, timeout=25)
    r.encoding = "gb18030"  # lightinnj uses GB18030 encoding
    # Polite delay
    time.sleep(random.uniform(0.5, 1.5))
    return r.text

def extract_links_from_index(html: str, base_url: str) -> list[tuple[str, str]]:
    """Extract section titles and links from the index page"""
    soup = BeautifulSoup(html, "html.parser")

    links = []
    anchors = soup.find_all("a", href=True)

    for a in anchors:
        text = a.get_text(strip=True)
        href = a.get("href")

        # Skip navigation links
        if text in ["回首页"]:
            continue

        # Only get actual section links (not too long)
        if href and not href.startswith("../") and text:
            full_url = urljoin(base_url, href)
            links.append((text, full_url))

    return links

def extract_page_content(html: str) -> str:
    """Extract main text and treat <b> blocks as headings.

    - Skips navigation like 书名/回目录
    - Works for 导言 where multiple <b> (一..八) appear in one <p>
    - Works for sections with headings like 壹 羡慕（一章二至三节）
    - Converts <br> tags to paragraph breaks (\n\n) for Markdown
    - Treats <b> that start with simplified numerals followed by fullwidth space (一二三…＋\u3000) as level-4 (####)
    """
    soup = BeautifulSoup(html, "html.parser")

    lines: list[str] = []
    current_heading: str | None = None
    current_heading_level: int = 3
    buffer: list[str] = []

    def flush_buffer():
        nonlocal buffer, current_heading, current_heading_level
        # Join buffer and preserve intentional paragraph breaks from <br>
        text = "".join(buffer).strip()
        buffer.clear()
        if not text:
            return
        if current_heading:
            lines.append(f"{'#' * current_heading_level} {current_heading}")

        # Split by double newlines (from <br><br>) first to preserve paragraph structure
        paragraphs = re.split(r"\n\n+", text)

        cleaned_paragraphs = []
        for para in paragraphs:
            # Within each paragraph, normalize whitespace but preserve single line breaks
            para_lines = para.split("\n")
            cleaned_lines = []
            for line in para_lines:
                # Normalize whitespace within each line
                cleaned = re.sub(r"[ \t]+", " ", line).strip()
                if cleaned:
                    cleaned_lines.append(cleaned)

            # Rejoin lines within paragraph
            para_text = "\n".join(cleaned_lines).strip()
            if para_text:
                cleaned_paragraphs.append(para_text)

        # Join paragraphs with double newlines (Markdown paragraph separator)
        text = "\n\n".join(cleaned_paragraphs)
        lines.append(text)

    # regex to identify H4 headings that begin with simplified Chinese numerals followed by a fullwidth space
    h4_simplified_re = re.compile(r"^[一二三四五六七八九十]+\u3000")

    for p in soup.find_all("p"):
        # quick navigation skip
        p_text = p.get_text(strip=True)
        if not p_text:
            continue
        if p_text.startswith("回目录") or p_text.startswith("书名："):
            continue

        for node in p.children:
            if isinstance(node, Tag):
                name = (node.name or "").lower()
                if name == "b":
                    # heading inside this paragraph
                    heading = node.get_text(strip=True)
                    if heading and not heading.startswith("书名") and not heading.startswith("回目录"):
                        flush_buffer()
                        # classify heading level: default H3, but if starts with simplified numerals + fullwidth space, use H4
                        if h4_simplified_re.match(heading):
                            current_heading_level = 4
                        else:
                            current_heading_level = 3
                        current_heading = heading
                    # do not add <b> text to buffer
                    continue
                elif name == "br":
                    # Convert <br> to paragraph break (\n\n) in Markdown
                    buffer.append("\n\n")
                    continue
                else:
                    # add tag's text with spaces preserved
                    t = node.get_text(separator=" ", strip=True)
                    if t:
                        buffer.append(t)
                        buffer.append(" ")
            elif isinstance(node, NavigableString):
                t = str(node).strip()
                if t:
                    buffer.append(t)
                    buffer.append(" ")

    # flush remaining
    flush_buffer()

    # Compose markdown
    md = "\n\n".join(lines).strip()
    return md

def build_book_markdown() -> str:
    """Build the complete markdown document"""
    index_html = fetch_html(BASE_URL)
    links = extract_links_from_index(index_html, BASE_URL)

    lines = []
    lines.append("# 歌中之歌\n")

    def dedup_leading_title(content: str, title: str) -> str:
        """Remove a leading line that duplicates the section title.

        Compares after normalizing fullwidth spaces to normal spaces and collapsing
        whitespace. Also ignores leading Markdown hashes on the first line.
        """
        def canon(s: str) -> str:
            s = s.replace("\u3000", " ")
            s = re.sub(r"\s+", " ", s).strip()
            return s

        content = content.lstrip()
        m = re.match(r"^(.*?)(\n|$)", content, re.S)
        if not m:
            return content
        first_line = m.group(1).strip()
        # strip heading hashes if present
        first_line_no_hash = re.sub(r"^#+\s+", "", first_line)
        if canon(first_line_no_hash) == canon(title):
            # drop the first line and following newline
            return content[len(m.group(0)):].lstrip("\n")
        return content

    for section_title, section_url in links:
        print(f"Scraping: {section_title}")

        try:
            page_html = fetch_html(section_url)
            content = extract_page_content(page_html)
            # Remove duplicated inline title inside the content if present
            content = dedup_leading_title(content, section_title)

            if content:
                lines.append(f"## {section_title}\n")
                lines.append(content)
                lines.append("")
        except Exception as e:
            print(f"  ⚠️ Error scraping {section_title}: {e}")

    # Join all lines and clean up
    md = "\n".join(lines).strip()
    # Normalize: replace fullwidth space with normal space and remove normal space after ideographic full stop
    md = md.replace("\u3000", " ")
    md = re.sub(r"。 +", "。", md)
    # Remove excessive blank lines
    md = re.sub(r'\n{3,}', '\n\n', md)
    return md + "\n"

if __name__ == "__main__":
    md = build_book_markdown()
    filename = "歌中之歌.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"✅ 已生成：{filename}")
```
</details>
