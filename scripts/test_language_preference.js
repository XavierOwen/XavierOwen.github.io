"use strict";

const assert = require("node:assert/strict");
const language = require("../assets/js/language-preference.js");

assert.equal(language.normalizeLanguage("zh-CN"), "zh");
assert.equal(language.normalizeLanguage("en-US"), "en");
assert.equal(language.normalizeLanguage("fr"), null);

assert.deepEqual(
  language.chooseLanguage("en", ["zh-CN"]),
  { language: "en", source: "stored" },
  "an explicit selection must override browser detection",
);
assert.deepEqual(
  language.chooseLanguage(null, ["zh-TW", "en-US"]),
  { language: "zh", source: "browser" },
  "the first supported browser language should win on a first visit",
);
assert.deepEqual(
  language.chooseLanguage(null, ["fr-FR"]),
  { language: "en", source: "browser" },
  "unsupported browser languages should use the English interface",
);

assert.deepEqual(
  language.languageTarget({ zhAvailable: "true", zhUrl: "/work/" }, "zh"),
  { available: true, url: "/work/" },
);
assert.deepEqual(
  language.languageTarget({ enAvailable: "false", enUrl: "/work/" }, "en"),
  { available: false, url: "/work/" },
);

assert.equal(language.needsFallback("article", "zh", "en", false), true);
assert.equal(language.needsFallback("article", "zh", "en", true), false);
assert.equal(language.needsFallback("route", "zh", "en", false), false);
assert.equal(language.needsRedirect("zh", "en", true, "/en/work/"), true);
assert.equal(language.needsRedirect("zh", "en", false, "/work/"), false);

console.log("Language-preference tests: passed.");
