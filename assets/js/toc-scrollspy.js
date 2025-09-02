(function(){
    // Add no-spy class to h4-h6 elements
    document.querySelectorAll('h4, h5, h6').forEach(h => {
        h.classList.add('no-spy');
    });

    // 1) 收集 TOC 链接和正文标题
    const toc = document.querySelector('.toc__left');
    if (!toc) return;

    const links = Array.from(toc.querySelectorAll('a[href^="#"]'));
    if (!links.length) return;

    // 允许的标题层级（与 TOC 设置匹配）
    const selectors = ['h2','h3'];
    const headings = Array.from(document.querySelectorAll(selectors.join(',')))
        .filter(h => h.id && !h.closest('.no-spy'));

    // 建立 #id -> <a> 映射
    const map = new Map();
    links.forEach(a => {
        const id = decodeURIComponent(a.getAttribute('href').slice(1));
        map.set(id, a);
    });

    // 2) 平滑滚动（点击 TOC 时） - "Bait and Switch" to prevent jQuery error
    links.forEach(a => {
        // On mousedown, before the click is processed by other scripts,
        // store the real href and replace it with something harmless.
        a.addEventListener('mousedown', e => {
            const href = a.getAttribute('href');
            if (href && href.startsWith('#')) {
                a.setAttribute('data-original-href', href);
                a.setAttribute('href', 'javascript:void(0)');
            }
        });

        // On click, do our custom scrolling.
        a.addEventListener('click', e => {
            // Restore the original href from the data attribute.
            const href = a.getAttribute('data-original-href');

            // It's good practice to restore the href attribute immediately.
            if (href) {
                a.setAttribute('href', href);
                a.removeAttribute('data-original-href');
            }

            if (!href || !href.startsWith('#')) {
                return; // Not a TOC link we should handle.
            }

            const id = decodeURIComponent(href.slice(1));
            const target = document.getElementById(id);

            if (!target) {
                return;
            }

            e.preventDefault();
            e.stopImmediatePropagation(); // Prevent any other click handlers from running.

            // Pause the scrollspy to avoid conflicts.
            pauseSpy(200);

            // Calculate the scroll position, accounting for the masthead and an extra offset.
            const masthead = document.querySelector('.masthead');
            const mastheadHeight = masthead ? masthead.offsetHeight : 0;
            const fontSize = parseFloat(getComputedStyle(document.documentElement).fontSize);
            const extraOffset = 0.5 * fontSize; // 0.5em in pixels
            const endY = target.getBoundingClientRect().top + window.scrollY - mastheadHeight + extraOffset;

            // Scroll to the target instantly.
            window.scrollTo(0, endY);

            // Update the URL hash.
            history.replaceState(null, '', href);

            // Set the active link in the TOC.
            setActive(a);
        });
    });

    // 3) 观察正文标题进入视口
    let spyPausedUntil = 0;
    const pauseSpy = (ms) => { spyPausedUntil = Date.now() + ms; };

    const setActive = (a) => {
        links.forEach(x => {
            x.classList.remove('is-active');
            x.style.color = '#666';
        });
        if (!a) return;
        a.classList.add('is-active');
        a.style.color = '#000';
        // 让当前项在 TOC 容器内可见
        if (toc && typeof toc.scrollTo === 'function') {
            a.scrollIntoView({ block: 'nearest' });
        }

        // Add progress indicator
        const progress = Math.min(100, (window.scrollY / (document.documentElement.scrollHeight - window.innerHeight)) * 100);
        a.style.background = `linear-gradient(to right, var(--global-link-color-hover) ${progress}%, transparent ${progress}%)`;
    };

    // rootMargin 调整触发点（顶部下移 20% 处触发）
    const io = new IntersectionObserver((entries) => {
        if (Date.now() < spyPausedUntil) return; // 点击滚动期间跳过
        // 取当前最靠上的正在可见的 heading
        const visible = entries
            .filter(en => en.isIntersecting)
            .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
        if (!visible) return;

        const a = map.get(visible.target.id);
        if (a) setActive(a);
    }, {
        root: null,
        threshold: [0, 0.1],         // 进入一点点就算可见
        rootMargin: '-20% 0px -70% 0px'
    });

    headings.forEach(h => io.observe(h));

    // 4) 初始状态：根据 URL hash 或第一个 heading
    const init = () => {
        const hash = decodeURIComponent(location.hash.replace('#',''));
        const a = map.get(hash) || map.get(headings[0]?.id);
        if (a) setActive(a);
    };
    init();

    // Debounced resize handler
    let resizeTimeout;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
            pauseSpy(200);
        }, 100);
    });
})();