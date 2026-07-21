# Yuanxing Cheng's website

这是 Yuanxing Cheng 的个人网站，发布在 GitHub Pages：
<https://xavierowen.github.io>。它以 Jekyll 静态站点为基础，但不只是学术
主页：研究与教学、数学笔记、项目、创作、翻译与灵性学习共同构成一个持续生长
的个人知识花园。

项目的约定、结构和长期决策见 [CONTEXT.md](CONTEXT.md)、
[架构说明](docs/architecture.md) 和 [ADR](docs/adr/)。

## 本地运行

需要 Ruby、Bundler 和 Node.js。首次安装依赖：

```sh
bundle config set --local path 'vendor/bundle'
bundle install
npm install
```

启动带实时刷新的本地预览：

```sh
bundle exec jekyll serve --livereload
```

打开 <http://localhost:4000>。修改 `_config.yml` 后需要重启服务。

## 验证

在提交任何布局、Liquid、Sass、配置或 JavaScript 变更前运行：

```sh
bundle exec jekyll build
node --check assets/js/toc-scrollspy.js
```

如果修改了 `assets/js/_main.js` 或 `assets/js/plugins/`，还要重新生成已提交的
浏览器 bundle：

```sh
npm run build:js
```

## 写内容

| 想发布的内容 | 源目录 | 公开索引 |
| --- | --- | --- |
| 数学、学习与个人笔记 | `_notes/` | `/notes/` |
| 灵性学习、译文与相关写作 | `_spirits/` | `/spirits/` |
| 软件、游戏、爬虫与艺术项目 | `_projects/` | `/projects/` |
| 论文与学术成果 | `_publications/` | `/publications/` |
| 教学经历与材料 | `_teaching/` | `/teaching/` |
| 独立页面 | `_pages/` | 由 front matter 的 `permalink` 决定 |

新文章优先使用 Markdown 与 YAML front matter。例如：

```md
---
title: "文章标题"
collection: notes
category: math
date: 2026-07-20
toc: true
tags: [Statistics]
---

正文……
```

`category` 必须先在 `_config.yml` 对应集合的分类中声明，才会在集合索引中显示。
`toc: true` 为长文生成二、三级标题目录。

## 站内链接

自定义集合支持两种 wiki 链接语法：

```text
[[另一篇文章的标题]]
[[显示文字::https://example.com]]
```

第一种在浏览器中解析为站内链接并显示预览；引用某篇文章的页面会在构建时出现在
该文章的反链列表中。标题是这套语法的键，因此改标题前应搜索已有引用：

```sh
rg -n -F '[[旧标题]]' _notes _spirits _projects _posts _pages
```

## 结构原则

- 内容与 URL 优先于上游主题实现；不要为了跟随模板而破坏已有 Markdown 或链接。
- 面向所有文章的行为放进一个 focused include，再由 `_layouts/single.html` 组合；
  不要把新逻辑散落进内容文件或重复写进布局。
- 需要新增内容类型时，先判断它是现有集合的分类、新的 Jekyll collection，还是
  独立页面；这个选择会影响 URL、导航、归档和后续维护。
- 有外部依赖或构建期行为的新增功能，必须带本地验证命令与必要的配置说明。

## 本地研究工具

`scripts/tavily-search.mjs` 是本地使用的检索辅助工具，不会进入网站构建。需要时
复制 `.env.example` 为 `.env`，填入自己的 `TAVILY_API_KEY`，然后运行：

```sh
npm run tavily:search -- "your query"
```

`.env` 已被忽略，绝不要提交密钥。

## 许可与来源

本站从 [Academic Pages](https://academicpages.github.io/) 派生；其底层主题为
[Minimal Mistakes](https://mmistakes.github.io/minimal-mistakes/)。各自的原始许可
文件保留在仓库中。站点内容的版权与使用条件以各页面另行注明的为准。
