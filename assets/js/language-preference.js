(function (globalObject, factory) {
  const api = factory();

  if (typeof module === "object" && module.exports) {
    module.exports = api;
  }

  if (globalObject && globalObject.document) {
    globalObject.XavierSiteLanguage = api;
    api.start(globalObject);
  }
}(typeof window !== "undefined" ? window : null, function () {
  "use strict";

  const STORAGE_KEY = "xavier-site-language";
  const SUPPORTED_LANGUAGES = ["zh", "en"];

  function normalizeLanguage(value) {
    const normalized = String(value || "").toLowerCase();
    if (normalized.indexOf("zh") === 0) return "zh";
    if (normalized.indexOf("en") === 0) return "en";
    return null;
  }

  function chooseLanguage(storedLanguage, browserLanguages) {
    const stored = normalizeLanguage(storedLanguage);
    if (stored) return { language: stored, source: "stored" };

    const detected = Array.from(browserLanguages || [])
      .map(normalizeLanguage)
      .find(Boolean);
    return { language: detected || "en", source: "browser" };
  }

  function languageTarget(data, language) {
    const normalized = normalizeLanguage(language);
    if (!normalized) return { available: false, url: null };

    return {
      available: data[`${normalized}Available`] !== "false",
      url: data[`${normalized}Url`] || null,
    };
  }

  function needsFallback(contextType, currentLanguage, preferredLanguage, targetAvailable) {
    return contextType === "article"
      && normalizeLanguage(currentLanguage) !== normalizeLanguage(preferredLanguage)
      && !targetAvailable;
  }

  function needsRedirect(currentLanguage, preferredLanguage, targetAvailable, targetUrl) {
    return normalizeLanguage(currentLanguage) !== normalizeLanguage(preferredLanguage)
      && targetAvailable
      && Boolean(targetUrl);
  }

  function readStoredLanguage(storage) {
    try {
      return storage.getItem(STORAGE_KEY);
    } catch (error) {
      return null;
    }
  }

  function storeLanguage(storage, language) {
    try {
      storage.setItem(STORAGE_KEY, language);
    } catch (error) {
      // A blocked storage API must not make language navigation unusable.
    }
  }

  function browserLanguages(navigatorObject) {
    if (navigatorObject.languages && navigatorObject.languages.length) {
      return navigatorObject.languages;
    }
    return [navigatorObject.language || "en"];
  }

  function withCurrentFragment(targetUrl, locationObject) {
    const target = new URL(targetUrl, locationObject.href);
    if (!target.search) target.search = locationObject.search;
    if (!target.hash) target.hash = locationObject.hash;
    return target;
  }

  function sameLocation(target, locationObject) {
    return target.pathname === locationObject.pathname
      && target.search === locationObject.search
      && target.hash === locationObject.hash;
  }

  function updateContext(context, language, windowObject, allowRedirect) {
    const target = languageTarget(context.dataset, language);
    const currentLanguage = context.dataset.languageCurrent;
    const contextType = context.dataset.languageContext;

    context.dataset.selectedLanguage = language;
    context.querySelectorAll("[data-language-choice]").forEach((choice) => {
      if (normalizeLanguage(choice.dataset.languageChoice) === language) {
        choice.setAttribute("aria-current", "true");
      } else {
        choice.removeAttribute("aria-current");
      }
    });

    const fallback = context.querySelector("[data-language-fallback]");
    if (fallback) {
      fallback.hidden = !needsFallback(contextType, currentLanguage, language, target.available);
    }

    if (!allowRedirect || !needsRedirect(currentLanguage, language, target.available, target.url)) {
      return false;
    }

    const targetLocation = withCurrentFragment(target.url, windowObject.location);
    if (sameLocation(targetLocation, windowObject.location)) return false;

    windowObject.location.replace(targetLocation.href);
    return true;
  }

  function announceLanguage(documentObject, language, source) {
    documentObject.dispatchEvent(new documentObject.defaultView.CustomEvent("site-language-change", {
      detail: { language, source },
    }));
  }

  function updateInterfaceLinks(documentObject, language) {
    documentObject.querySelectorAll("[data-ui-language-link]").forEach((link) => {
      const target = languageTarget(link.dataset, language);
      if (target.url) link.href = target.url;
    });
  }

  function updateGlobalChoices(documentObject, context, language) {
    documentObject.querySelectorAll("[data-language-choice]").forEach((choice) => {
      if (choice.closest("[data-language-context]")) return;

      const choiceLanguage = normalizeLanguage(choice.dataset.languageChoice);
      if (choiceLanguage === language) {
        choice.setAttribute("aria-current", "true");
      } else {
        choice.removeAttribute("aria-current");
      }
      if (!context || !choiceLanguage) return;

      const target = languageTarget(context.dataset, choiceLanguage);
      if (target.url) choice.href = target.url;
    });
  }

  function initialize(windowObject) {
    const documentObject = windowObject.document;
    const preference = chooseLanguage(
      readStoredLanguage(windowObject.localStorage),
      browserLanguages(windowObject.navigator),
    );

    documentObject.documentElement.dataset.uiLanguage = preference.language;
    documentObject.documentElement.dataset.languagePreferenceSource = preference.source;

    const contexts = Array.from(documentObject.querySelectorAll("[data-language-context]"));
    const primaryContext = contexts[0] || null;
    updateInterfaceLinks(documentObject, preference.language);
    updateGlobalChoices(documentObject, primaryContext, preference.language);
    const redirecting = contexts.some((context) => updateContext(context, preference.language, windowObject, true));
    if (!redirecting) announceLanguage(documentObject, preference.language, preference.source);

    documentObject.querySelectorAll("[data-language-choice]").forEach((choice) => {
      choice.addEventListener("click", (event) => {
        const language = normalizeLanguage(choice.dataset.languageChoice);
        if (!language || !SUPPORTED_LANGUAGES.includes(language)) return;

        storeLanguage(windowObject.localStorage, language);
        documentObject.documentElement.dataset.uiLanguage = language;
        documentObject.documentElement.dataset.languagePreferenceSource = "stored";
        updateInterfaceLinks(documentObject, language);
        updateGlobalChoices(documentObject, primaryContext, language);

        const context = choice.closest("[data-language-context]") || primaryContext;
        if (!context) {
          announceLanguage(documentObject, language, "stored");
          return;
        }

        const target = languageTarget(context.dataset, language);
        const currentLanguage = normalizeLanguage(context.dataset.languageCurrent);
        if (!target.available || currentLanguage === language) {
          event.preventDefault();
          updateContext(context, language, windowObject, false);
        }
        announceLanguage(documentObject, language, "stored");
      });
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
    STORAGE_KEY,
    chooseLanguage,
    languageTarget,
    needsFallback,
    needsRedirect,
    normalizeLanguage,
    start,
  };
}));
