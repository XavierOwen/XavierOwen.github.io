---
title: "Simple scraper for High wenli union Bible 深文理和合本圣经"
collection: projects
category: scraper
excerpt: "a simple scraper to obtain High wenli union Bible 深文理和合本圣经 from [维基文库](https://zh.wikisource.org/zh-hans/%E8%81%96%E7%B6%93_(%E6%96%87%E7%90%86%E5%92%8C%E5%90%88))"
permalink: "/games/scrapy-high-wenli-union-Bible"
date: 2025-08-25
---

# 在线抓取深文理和合本内容

[原文链接](https://zh.wikisource.org/zh-hans/%E8%81%96%E7%B6%93_(%E6%96%87%E7%90%86%E5%92%8C%E5%90%88))

[代码链接](https://github.com/XavierOwen/Practicing-simple-spider/blob/main/scrapy-high-wenli-union-Bible.py)

## 步骤

1. join得到正确书卷链接，提取所有书卷
2. 找到 h2 并转为 ##
3. 处理经节号码
4. 处理句末标点
5. 处理额外角标

<details markdown="1">
<summary>Show Python code</summary>

```python
import requests
from bs4 import BeautifulSoup, Tag
import time
import random
import re

# Use a Session with a real User-Agent and polite headers to avoid 403
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 BibleScraper/0.1 (+https://github.com/XavierOwen)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": "https://zh.wikisource.org/",
    "Connection": "keep-alive",
})

def fetch(url: str, min_delay: float = 0.8, max_delay: float = 1.6):
    """GET with session, basic retry, and polite delay to reduce 403 risk."""
    tries = 3
    last_exc = None
    for attempt in range(tries):
        try:
            resp = session.get(url, timeout=20)
            if resp.status_code == 200:
                # Wikisource is UTF-8
                resp.encoding = "utf-8"
                # polite delay between requests
                time.sleep(random.uniform(min_delay, max_delay))
                return resp
            elif resp.status_code in (403, 429):
                # Backoff and retry
                time.sleep(1.5 * (attempt + 1))
            else:
                # Small delay then retry for transient errors
                time.sleep(0.8)
        except Exception as e:
            last_exc = e
            time.sleep(0.8)
    # If all retries failed, raise for visibility
    if last_exc:
        raise last_exc
    raise RuntimeError(f"Failed to fetch {url} after {tries} attempts")

url = "https://zh.wikisource.org/zh-hans/%E8%81%96%E7%B6%93_(%E6%96%87%E7%90%86%E5%92%8C%E5%90%88)"
# Fetch the page
response = fetch(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Filtered extraction
book_titles_s = []
for li in soup.find_all("li"):
    a_tag = li.find("a")
    if a_tag and a_tag.has_attr("href"):
        if a_tag["href"].startswith("/wiki/%E8%81%96%E7%B6%93") or \
           a_tag["href"].startswith("https://zh.wikisource.org/wiki/%E8%81%96%E7%B6%93"):
            book_titles_s.append(a_tag.get_text(strip=True))

book_titles_s = book_titles_s[3:]

# Container for extracted content
results = []

for i, book_title in enumerate(book_titles):
    #chapter_title = "#" + book_titles_s[i]
    results.append(f"# {book_titles_s[i]}")
    url = "https://zh.wikisource.org/zh-hans/%E8%81%96%E7%B6%93_(%E6%96%87%E7%90%86%E5%92%8C%E5%90%88)/"+book_title
    response = fetch(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    # Walk through elements in document order
    for elem in soup.body.descendants:
        if isinstance(elem, Tag):
            # Handle headings
            if elem.name == 'h2':
                heading_text = elem.get_text(strip=True)
                if heading_text:
                    #results.append("\n## " + heading_text+"\n")
                    results.append(f"\n## {heading_text}\n")
            elif elem.name == "p":
                # 保留第一个数字型 <sup> 作为节编号；清理其余字母型脚注上标
                first_sup = elem.find("sup")
                verse_sup_text = None
                if first_sup:
                    t0 = first_sup.get_text(strip=True)
                    if t0.isdigit():
                        verse_sup_text = t0  # 记录节编号文本
                # 删除除第一个数字节编号之外的字母型 <sup>
                for s in elem.find_all("sup"):
                    if verse_sup_text and s is first_sup:
                        # 这是首个且为数字的节编号，保留
                        continue
                    tt = s.get_text(strip=True)
                    if tt and not tt.isdigit():
                        # 形如 a/b/c 等字母脚注
                        if re.fullmatch(r"[A-Za-z]+", tt):
                            s.decompose()
                # 之后按原逻辑处理
                sup = first_sup if (first_sup and verse_sup_text) else None
                text = elem.get_text(strip=True)
                if text:
                    if sup:
                        number = sup.get_text(strip=True)
                        # Remove the number from text body
                        text = text.replace(number, "", 1).strip()
                        if text.endswith("、○"):
                            text = text[:-2] + "。"
                        results.append(f"{number} {text}")
                    else:
                        if text.endswith("、○"):
                            text = text[:-2] + "。"
                        results.append(text)
    results.append('\n')

text_output = "\n".join(results)
#print(text_output)

# Optionally write to file
with open("bible.md", "w", encoding="utf-8") as f:
    f.write(text_output)
```
</details>
