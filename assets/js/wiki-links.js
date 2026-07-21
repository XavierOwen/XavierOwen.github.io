(function (globalObject, factory) {
  const api = factory();

  if (typeof module === "object" && module.exports) {
    module.exports = api;
  }

  if (globalObject && globalObject.document) {
    globalObject.XavierWikiLinks = api;
    api.start(globalObject);
  }
}(typeof window !== "undefined" ? window : null, function () {
  "use strict";

  function normalizeLanguage(value) {
    const language = String(value || "").toLowerCase();
    if (language.indexOf("zh") === 0) return "zh";
    if (language.indexOf("en") === 0) return "en";
    return "en";
  }

  function normalizeTitle(value) {
    return String(value || "").trim().toLowerCase();
  }

  function createLookup(items) {
    const byAlias = new Map();
    const byId = new Map();

    (items || []).forEach((item) => {
      byId.set(item.content_id, item);
      (item.aliases || []).forEach((alias) => byAlias.set(normalizeTitle(alias), item));
    });

    return { byAlias, byId };
  }

  function resolveItem(item, preferredLanguage) {
    if (!item) return null;

    const preferred = normalizeLanguage(preferredLanguage);
    const preferredVersion = item.versions[preferred];
    const version = preferredVersion || item.versions[item.original_language];
    if (!version) return null;

    return {
      contentId: item.content_id,
      fallback: !preferredVersion,
      language: version.language,
      originalLanguage: item.original_language,
      title: version.title,
      summary: version.summary,
      collection: version.collection,
      url: version.url,
    };
  }

  function parseWikiToken(token) {
    const delimiter = token.indexOf("::");
    if (delimiter === -1) return { type: "internal", title: token.trim() };

    return {
      type: "external",
      title: token.slice(0, delimiter).trim(),
      url: token.slice(delimiter + 2).trim(),
    };
  }

  function siteUrl(url, baseurl) {
    if (!url || /^(?:[a-z]+:)?\/\//i.test(url)) return url;
    const base = String(baseurl || "").replace(/\/$/, "");
    const path = url.indexOf("/") === 0 ? url : `/${url}`;
    if (base && path.indexOf(`${base}/`) === 0) return path;
    return `${base}${path}`;
  }

  function fallbackLabel(uiLanguage, originalLanguage) {
    const original = normalizeLanguage(originalLanguage);
    if (normalizeLanguage(uiLanguage) === "zh") {
      return `原文：${original === "zh" ? "中文" : "英文"}`;
    }
    return `Original: ${original === "zh" ? "Chinese" : "English"}`;
  }

  function excludedTextNode(node) {
    const parent = node.parentElement;
    return !parent || Boolean(parent.closest(
      "a, code, pre, script, style, textarea, .highlighter-rouge, .highlight, [class*='language-']",
    ));
  }

  function collectTextNodes(root) {
    const nodes = [];

    function visit(node) {
      if (node.nodeType === 3) {
        if (!excludedTextNode(node) && node.textContent.indexOf("[[") !== -1) nodes.push(node);
        return;
      }
      if (node.nodeType !== 1) return;
      Array.from(node.childNodes).forEach(visit);
    }

    visit(root);
    return nodes;
  }

  function hidePreview(documentObject) {
    const preview = documentObject.querySelector(".wiki-link-preview");
    if (preview) preview.remove();
  }

  function showPreview(documentObject, anchor, item, uiLanguage) {
    hidePreview(documentObject);
    const resolved = resolveItem(item, uiLanguage);
    if (!resolved) return;

    const preview = documentObject.createElement("div");
    preview.className = "wiki-link-preview";
    preview.setAttribute("role", "tooltip");

    const header = documentObject.createElement("div");
    header.className = "preview-header";
    const title = documentObject.createElement("strong");
    title.textContent = resolved.title;
    const collection = documentObject.createElement("span");
    collection.className = "preview-collection";
    collection.textContent = resolved.collection;
    header.append(title, collection);
    preview.appendChild(header);

    if (resolved.fallback) {
      const fallback = documentObject.createElement("span");
      fallback.className = "preview-fallback";
      fallback.textContent = fallbackLabel(uiLanguage, resolved.originalLanguage);
      preview.appendChild(fallback);
    }

    const summary = documentObject.createElement("p");
    summary.className = "preview-excerpt";
    summary.textContent = resolved.summary;
    preview.appendChild(summary);
    documentObject.body.appendChild(preview);

    const rect = anchor.getBoundingClientRect();
    preview.style.left = `${rect.left}px`;
    preview.style.top = `${rect.bottom + 5}px`;
    const previewRect = preview.getBoundingClientRect();
    if (previewRect.right > documentObject.defaultView.innerWidth) {
      preview.style.left = `${documentObject.defaultView.innerWidth - previewRect.width - 10}px`;
    }
    if (previewRect.bottom > documentObject.defaultView.innerHeight) {
      preview.style.top = `${rect.top - previewRect.height - 5}px`;
    }
  }

  function updateWikiAnchor(anchor, item, language, baseurl) {
    const resolved = resolveItem(item, language);
    if (!resolved) return;

    anchor.href = siteUrl(resolved.url, baseurl);
    anchor.lang = resolved.language;
    anchor.dataset.languageFallback = String(resolved.fallback);
    anchor.title = resolved.fallback ? fallbackLabel(language, resolved.originalLanguage) : "";
  }

  function createInternalLink(documentObject, token, item, language, baseurl) {
    if (!item) {
      const stale = documentObject.createElement("span");
      stale.textContent = token.title;
      stale.className = "wiki-link-stale";
      stale.title = `Content not found: ${token.title}`;
      return stale;
    }

    const link = documentObject.createElement("a");
    link.textContent = token.title;
    link.className = "wiki-link";
    link.dataset.wikiContentId = item.content_id;
    updateWikiAnchor(link, item, language, baseurl);
    link.addEventListener("mouseenter", () => {
      const currentLanguage = documentObject.documentElement.dataset.uiLanguage || language;
      showPreview(documentObject, link, item, currentLanguage);
    });
    link.addEventListener("mouseleave", () => hidePreview(documentObject));
    link.addEventListener("focus", () => {
      const currentLanguage = documentObject.documentElement.dataset.uiLanguage || language;
      showPreview(documentObject, link, item, currentLanguage);
    });
    link.addEventListener("blur", () => hidePreview(documentObject));
    return link;
  }

  function createExternalLink(documentObject, token) {
    const link = documentObject.createElement("a");
    link.href = token.url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = token.title;
    link.className = "wiki-link-external";
    return link;
  }

  function replaceWikiSyntax(documentObject, root, lookup, language, baseurl) {
    const wikiPattern = /\[\[([^\]]+)\]\]/g;

    collectTextNodes(root).forEach((node) => {
      const text = node.textContent;
      const fragment = documentObject.createDocumentFragment();
      let cursor = 0;
      let match;

      while ((match = wikiPattern.exec(text)) !== null) {
        fragment.appendChild(documentObject.createTextNode(text.slice(cursor, match.index)));
        const token = parseWikiToken(match[1]);
        if (token.type === "external") {
          fragment.appendChild(createExternalLink(documentObject, token));
        } else {
          fragment.appendChild(createInternalLink(
            documentObject,
            token,
            lookup.byAlias.get(normalizeTitle(token.title)),
            language,
            baseurl,
          ));
        }
        cursor = match.index + match[0].length;
      }

      fragment.appendChild(documentObject.createTextNode(text.slice(cursor)));
      node.parentNode.replaceChild(fragment, node);
      wikiPattern.lastIndex = 0;
    });
  }

  function updateLanguageAwareContent(documentObject, lookup, language, baseurl) {
    hidePreview(documentObject);

    documentObject.querySelectorAll("[data-wiki-content-id]").forEach((anchor) => {
      updateWikiAnchor(anchor, lookup.byId.get(anchor.dataset.wikiContentId), language, baseurl);
    });

    documentObject.querySelectorAll("[data-language-aware-link]").forEach((container) => {
      const resolved = resolveItem(lookup.byId.get(container.dataset.contentId), language);
      if (!resolved) return;

      const anchor = container.querySelector("[data-language-aware-anchor]");
      const summary = container.querySelector("[data-language-aware-summary]");
      const fallback = container.querySelector("[data-language-aware-fallback]");
      if (anchor) {
        anchor.href = siteUrl(resolved.url, baseurl);
        anchor.lang = resolved.language;
        anchor.textContent = resolved.title;
      }
      if (summary) summary.textContent = resolved.summary;
      if (fallback) {
        fallback.hidden = !resolved.fallback;
        fallback.textContent = fallbackLabel(language, resolved.originalLanguage);
      }
    });
  }

  function initialize(windowObject) {
    const documentObject = windowObject.document;
    const indexElement = documentObject.getElementById("wiki-link-index");
    if (!indexElement) return;

    const index = JSON.parse(indexElement.textContent);
    const lookup = createLookup(index.items);
    const baseurl = indexElement.dataset.baseurl || "";
    const language = normalizeLanguage(documentObject.documentElement.dataset.uiLanguage);
    const content = documentObject.querySelector(".page__content");

    if (content) replaceWikiSyntax(documentObject, content, lookup, language, baseurl);
    updateLanguageAwareContent(documentObject, lookup, language, baseurl);
    documentObject.addEventListener("site-language-change", (event) => {
      updateLanguageAwareContent(documentObject, lookup, normalizeLanguage(event.detail.language), baseurl);
    });
  }

  function start(windowObject) {
    if (windowObject.document.readyState === "loading") {
      windowObject.document.addEventListener("DOMContentLoaded", () => initialize(windowObject), { once: true });
    } else {
      initialize(windowObject);
    }
  }

  return {
    createLookup,
    fallbackLabel,
    normalizeTitle,
    parseWikiToken,
    resolveItem,
    siteUrl,
    start,
  };
}));
