---
title: "Simple scraper for 十二篮"
collection: projects
category: scraper
excerpt: "a simple scraper to obtain passages of 12 brackets from [this website](https://pages.uoregon.edu/fyin/%E7%81%B5%E7%B2%AE/%E5%8D%81%E4%BA%8C%E7%AF%AE/%E5%8D%81%E4%BA%8C%E7%AF%AE%20%E7%9B%AE%E5%BD%95.htm)"
permalink: "/games/scraper-12-brackets"
date: 2025-08-25
---

# 在线抓取十二篮内容

[原文链接](https://pages.uoregon.edu/fyin/%E7%81%B5%E7%B2%AE/%E5%8D%81%E4%BA%8C%E7%AF%AE/%E5%8D%81%E4%BA%8C%E7%AF%AE%20%E7%9B%AE%E5%BD%95.htm)

[代码链接](https://github.com/XavierOwen/Practicing-simple-spider/blob/main/scraper-12-brackets.py)

## 步骤

1. 从主目录页中提取 12 个一级链接
2. 进入子目录页，提取每辑中的文章链接
3. 抓取文章正文
4. 处理为规范化 `Markdown`
5. 最终合并生成一个 **十二篮.md** 文件，结构清晰，带目录、标题和原文链接

## 脚本演进

### 初始步骤

- 使用 `requests` 请求网页，注意到站点采用 `GB18030` 编码，因此强制 `r.encoding = "gb18030"`
- 再用 `BeautifulSoup` 解析 `HTML`，获取主目录页中的 `<a>` 标签

### 一级目录提取

- 观察网页结构，定位到 `id="table3"` 的表格，里面包含 12 个一级目录链接。
- 提取 `<a>` 标签并用 `urljoin` 统一为绝对 `URL`。

### 子目录解析

- 在每个一级目录页，找到 `<td colspan="4">` 中的 12 个文章链接，后来发现有些页面采用的是 `<td colspan="5">`。
- 提取锚点文本，并实现一个 `anchor_title_after_dunhao` 函数，截取**顿号**后的实际标题。

### 正文抓取与处理

一开始通过观察，决定取 第六个 `<p>` 段落，但发现存在以下问题：

- 某些文章正文并不在第六个 `<p>`
- `<b>` 标签既会转换为四级标题，又会重复出现在正文里
- 部分 `<p>` 前面有大量空格或全角空格，生成的 `Markdown` 有不必要的缩进

解决措施：

- 避免重复：在处理 `NavigableString` 时，检测父节点是否为 `<b>`，若是则跳过，以防止标题内容重复。
- 正文定位改进：不再硬编码取第 6 段，而是**遍历**所有 `<p>`，选取**文本长度最大的一段**作为正文。
- 缩进清理：在写入 `Markdown` 前，对每一行做 `lstrip()` 扩展，去除*普通空格*、*制表符*、*全角空格 `\u3000`*、*不换行空格 `\u00A0`*。

### 汇总与输出

遍历全部 12 个一级目录和子目录，依次抓取正文。

- 在 Markdown 中组织为：
- 一级标题：# 十二篮
- 二级标题：每辑名称（第一辑、第二辑……）
- 三级标题：文章标题
- 原文链接（便于跳转）
- 正文 Markdown 内容

### 微调

- 文本内存在的特殊字符消除
- 转pdf使用`npm markdown-pdf`，配置文件在最后

## 脚本函数主要功能

- `fetch_html(url)`：抓取网页，设置编码
- `extract_main_links(html, base_url)`：获取一级目录链接
- `extract_sub_anchors(html)`：获取子目录中的文章链接
- `anchor_title_after_dunhao(text)`：处理标题文本
- `third_p_to_markdown(html)`：抽取正文并转 Markdown（核心改造）
- `build_book_markdown()`：汇总所有内容并生成完整 Markdown

## 经验

1.	网页编码处理：站点为 `GB18030`，若不手动指定，解析会乱码。
2.	HTML 结构多样性：子目录 `<td>` 的 `colspan` 属性不同，需要兼容。
3.	内容定位：单纯用第三个 `<p>` 不够稳，改为**最长文本段落**更鲁棒。
4.	重复问题：`<b>` 标签既是标题又是正文，通过跳过 `NavigableString` 修复。
5.	空格清理：行首的各种空白符需要正则清理，否则生成的 `Markdown` 会有异常缩进。


<details markdown="1">
<summary>爬虫脚本</summary>

```python
import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from urllib.parse import urljoin
import re

BASE_URL = "https://pages.uoregon.edu/fyin/%E7%81%B5%E7%B2%AE/%E5%8D%81%E4%BA%8C%E7%AF%AE/%E5%8D%81%E4%BA%8C%E7%AF%AE%20%E7%9B%AE%E5%BD%95.htm"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}

CN_NUM = ["第一辑","第二辑","第三辑","第四辑","第五辑","第六辑","第七辑","第八辑","第九辑","第十辑","第十一辑","第十二辑"]

def fetch_html(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=25)
    r.encoding = "gb18030"  # 该站点国标编码
    return r.text

def extract_main_links(html: str, base_url: str) -> list[str]:
    """目录页：提取 12 个一级链接（注意你此前的 1:14 切片修正）"""
    soup = BeautifulSoup(html, "html.parser")
    table3 = soup.find(id="table3")
    if not table3:
        table3 = soup.select_one('#table3, a[name="table3"], [name="table3"]')
        if table3 and table3.name == "a":
            table3 = table3.parent
    anchors = table3.find_all("a", href=True)
    anchors = [a for a in anchors if not a["href"].startswith("#")]
    anchors = anchors[1:14]  # 你修过的范围
    return [urljoin(base_url, a["href"]) for a in anchors]

def extract_sub_anchors(html: str):
    """子页：返回 12 个 <a> 标签（对象），用于拿标题文本和链接"""
    soup = BeautifulSoup(html, "html.parser")
    td = soup.find("td", attrs={"colspan": "5"})
    if not td:
        td = soup.find("td", attrs={"colspan": "4"})
    if not td:
        return []
    anchors = td.find_all("a", href=True)
    return anchors[:12]

def anchor_title_after_dunhao(text: str) -> str:
    """从锚文本中取“顿号”后的标题；若无顿号，返回原文本"""
    t = (text or "").strip()
    if "、" in t:
        parts = t.split("、", 1)
        # 若顿号在最前或分割后为空，退回原文本
        cand = parts[1].strip()
        return cand or t
    return t

def third_p_to_markdown(html: str) -> str:
    """取第三个 <p>，把里面的 <b> 转成 #### 标题；其他按纯文本处理，<br> 转换为换行。"""
    soup = BeautifulSoup(html, "html.parser")
    p_list = soup.find_all("p")

    # 选择“可见文本长度”最大的 <p> 作为正文段落；把 <br> 当作换行
    candidates = [p for p in p_list if p.get_text(strip=True)]
    if not candidates:
        return ""

    def _text_len(p: Tag) -> int:
        return len(p.get_text(separator="\n", strip=True))

    target = max(candidates, key=_text_len)

    lines = []
    buf = []

    def flush_buf_as_text():
        text = "".join(buf)
        if not text:
            buf.clear()
            return
        # 统一换行
        text = re.sub(r"\r\n?", "\n", text)
        # 去除每行行首多余缩进（普通空格 / 制表符 / 不换行空格 / 全角空格）
        cleaned_lines = [re.sub(r'^[\u3000\u00A0 \t]+', '', ln) for ln in text.split('\n')]
        text_clean = "\n".join(cleaned_lines).strip()
        if text_clean:
            lines.append(text_clean)
        buf.clear()

    # 遍历 target 的直接/嵌套子节点，处理 <b>、<br> 等
    for node in target.descendants:
        if isinstance(node, NavigableString):
            # 如果该文本节点属于 <b> 内部，则跳过，避免将 <b> 内容既作为标题又重复为正文
            parent = getattr(node, 'parent', None)
            if isinstance(parent, Tag) and parent.name and parent.name.lower() == 'b':
                continue
            buf.append(str(node))
        elif isinstance(node, Tag):
            if node.name.lower() == "br":
                buf.append("\n")
            elif node.name.lower() == "b":
                # 输出之前的缓冲文本为段落
                flush_buf_as_text()
                title = node.get_text(strip=True)
                if title:
                    lines.append(f"#### {title}")
                # <b> 内文本不再重复追加
            else:
                # 其他标签按其纯文本加入缓冲（避免重复抓取其孩子）
                # 但 descendants 会再到其子节点，这里跳过以免重复；
                # 让 NavigableString 分支处理子文本即可
                pass

    # 收尾
    flush_buf_as_text()

    # 规范化：把多余空行压缩为两行
    md = "\n\n".join([s.strip() for s in lines if s is not None])
    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    return md

def build_book_markdown() -> str:
    html_main = fetch_html(BASE_URL)
    main_links = extract_main_links(html_main, BASE_URL)
    if len(main_links) != 12:
        print(f"⚠️ 一级链接数量={len(main_links)}（预期 12），将按实际处理。")

    out = []
    out.append("# 十二篮\n")

    for vol_idx, vol_url in enumerate(main_links, start=1):
        vol_name = CN_NUM[vol_idx - 1] if vol_idx - 1 < len(CN_NUM) else f"第{vol_idx}辑"
        out.append(f"## {vol_name}\n")

        sub_html = fetch_html(vol_url)
        anchors = extract_sub_anchors(sub_html)

        # 若不足 12 个锚点，按实际数量写
        for a in anchors:
            title_full = a.get_text(" ", strip=True)
            title = anchor_title_after_dunhao(title_full) or title_full
            link = urljoin(vol_url, a["href"])

            # 抓内容页的第三个 <p>
            page_html = fetch_html(link)
            body_md = third_p_to_markdown(page_html)

            # 写入一个条目
            out.append(f"### {title}\n")
            # 可在标题下放原文链接（可选）
            out.append(f"[原文链接]({link})\n")
            if body_md:
                out.append(body_md + "\n")
            else:
                out.append("_（本条未检测到第三个段落或内容为空）_\n")

        # 分卷之间加一行
        out.append("")

    return "\n".join(out).strip() + "\n"

if __name__ == "__main__":
    md = build_book_markdown()
    with open("十二篮.md", "w", encoding="utf-8") as f:
        f.write(md)
    print("✅ 已生成：十二篮.md")
```
</details>



<details markdown="1">
<summary>字符修正</summary>

```python
#!/usr/bin/env python3
import argparse, pathlib, re, shutil, sys

# 1) 只替换 U+2500 为 “——”（两枚 EM DASH）
REPLACE_MAP = {
    "\u2500": "——",  # BOX DRAWINGS LIGHT HORIZONTAL → EM DASH × 2
    "——声": ""
}

# 2) 需要“删除”的集合
# - C0 控制字符（保留 \n \r \t）与 DEL
CTRL_PATTERN = r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]"
# - 私用区 PUA（含 U+E216 等）
PUA_PATTERN  = r"[\uE000-\uF8FF]"
# - 几何图形块（含 U+25A1 等各类方块符号）
GEOM_PATTERN = r"[\u25A0-\u25FF]"

DELETE_RE = re.compile(f"(?:{CTRL_PATTERN}|{PUA_PATTERN}|{GEOM_PATTERN})")

def clean_text(s: str) -> str:
    # 先做精确替换（避免把 \u2500 当作几何图形误删）
    for src, dst in REPLACE_MAP.items():
        s = s.replace(src, dst)
    # 再删除“方块类 / 私用区 / 控制符”
    s = DELETE_RE.sub("", s)
    return s

def main():
    ap = argparse.ArgumentParser(description="Clean Markdown: replace U+2500 with '——' and remove squares/PUA/control chars")
    ap.add_argument("input", help="input .md file")
    ap.add_argument("-o", "--output", help="output file (default: stdout unless --inplace)")
    ap.add_argument("--inplace", action="store_true", help="overwrite input (creates .bak)")
    args = ap.parse_args()

    src = pathlib.Path(args.input)
    if not src.exists():
        print(f"File not found: {src}", file=sys.stderr); sys.exit(1)

    text = src.read_text(encoding="utf-8", errors="strict")
    cleaned = clean_text(text)

    if args.inplace:
        bak = src.with_suffix(src.suffix + ".bak")
        shutil.copyfile(src, bak)
        src.write_text(cleaned, encoding="utf-8")
        print(f"Done. In-place cleaned: {src.name} (backup: {bak.name})")
    elif args.output:
        out = pathlib.Path(args.output)
        out.write_text(cleaned, encoding="utf-8")
        print(f"Done. Wrote: {out}")
    else:
        sys.stdout.write(cleaned)

if __name__ == "__main__":
    main()
```
</details>



<details markdown="1">
<summary>markdown-pdf settings</summary>

```json
{
  "pdf_options": {
    "outline": true,
    "format": "Letter",
    "margin": {
      "top": "0.8in",
      "bottom": "0.8in",
      "left": "0.8in",
      "right": "0.8in"
    },
    "displayHeaderFooter": true,
    "headerTemplate": "<div></div>",
    "footerTemplate": "<div style='font-size:9px; width:100%; text-align:center;'><span class='pageNumber'></span> / <span class='totalPages'></span></div>"
  }
}
```
</details>
