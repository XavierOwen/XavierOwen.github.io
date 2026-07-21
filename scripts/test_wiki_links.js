"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const wiki = require("../assets/js/wiki-links.js");

const index = JSON.parse(fs.readFileSync(path.join(__dirname, "../_data/content-index.json"), "utf8"));
const lookup = wiki.createLookup(index.items);
const scriptureItem = lookup.byAlias.get(wiki.normalizeTitle("读书之路"));
assert.ok(scriptureItem, "existing authored titles must remain valid wiki-link aliases");

const fallback = wiki.resolveItem(scriptureItem, "en");
assert.equal(fallback.fallback, true);
assert.equal(fallback.language, "zh");
assert.ok(fallback.url, "a missing English translation must resolve to the Chinese original");

const paired = {
  content_id: "paired",
  original_language: "zh",
  aliases: ["配对", "Pair"],
  versions: {
    zh: { language: "zh", title: "配对", summary: "中文", collection: "notes", url: "/pair/" },
    en: { language: "en", title: "Pair", summary: "English", collection: "notes", url: "/en/pair/" },
  },
};
assert.equal(wiki.resolveItem(paired, "en").url, "/en/pair/");
assert.equal(wiki.resolveItem(paired, "en").fallback, false);

assert.deepEqual(wiki.parseWikiToken("Label::https://example.com/a::b"), {
  type: "external",
  title: "Label",
  url: "https://example.com/a::b",
});
assert.deepEqual(wiki.parseWikiToken(" 读书之路 "), { type: "internal", title: "读书之路" });
assert.equal(wiki.siteUrl("/notes/example/", "/garden"), "/garden/notes/example/");
assert.equal(wiki.fallbackLabel("en", "zh"), "Original: Chinese");

console.log("Wiki-link language resolution tests: passed.");
