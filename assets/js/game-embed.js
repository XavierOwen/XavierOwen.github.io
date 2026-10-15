document.querySelectorAll('[data-game-embed]').forEach((embed) => {
  const frame = embed.querySelector('[data-game-frame]');
  const stage = embed.querySelector('[data-game-stage]');
  const status = embed.querySelector('[data-game-status]');
  const fullscreen = embed.querySelector('[data-game-fullscreen]');
  const startButtons = embed.querySelectorAll('[data-game-start]');
  let started = false;
  let gameResult = null;
  let expanded = false;
  let switching = false;
  let scrollPosition = null;

  const updateDisplay = () => {
    const nativeFullscreen = document.fullscreenElement === embed;
    const enlarged = expanded || nativeFullscreen;
    embed.classList.toggle('is-expanded', enlarged);
    fullscreen.textContent = nativeFullscreen ? '退出全屏' : (expanded ? '收起游戏' : '展开游玩');
    fullscreen.setAttribute('aria-pressed', String(enlarged));
    if (!started) return;
    const mode = nativeFullscreen ? '已进入原生全屏' : (expanded ? '已在浏览器内展开' : '');
    if (gameResult === 'error') {
      status.textContent = '游戏载入失败，请检查网络后刷新页面。';
    } else if (mode) {
      status.textContent = mode + (gameResult === 'ready' ? ' · 收起后继续当前游戏。' : ' · 正在加载游戏…');
    } else {
      status.textContent = gameResult === 'ready' ? '游戏已准备就绪。点击画面开始操作；展开后可继续当前游戏。' : '游戏页面已打开，正在加载资源…';
    }
  };

  const collapse = () => {
    expanded = false;
    document.documentElement.classList.remove('has-expanded-game');
    document.documentElement.style.removeProperty('--game-embed-scroll-top');
    document.documentElement.style.removeProperty('--game-embed-scroll-left');
    updateDisplay();
    fullscreen.focus({ preventScroll: true });
    window.scrollTo(scrollPosition.x, scrollPosition.y);
    scrollPosition = null;
  };

  const expand = () => {
    scrollPosition = { x: window.scrollX, y: window.scrollY };
    document.documentElement.style.setProperty('--game-embed-scroll-top', `${-scrollPosition.y}px`);
    document.documentElement.style.setProperty('--game-embed-scroll-left', `${-scrollPosition.x}px`);
    document.documentElement.classList.add('has-expanded-game');
    expanded = true;
    updateDisplay();
    frame.focus({ preventScroll: true });
  };

  startButtons.forEach((button) => {
    button.addEventListener('click', () => {
      if (started) return;
      started = true;
      startButtons.forEach((startButton) => {
        startButton.disabled = true;
        startButton.textContent = '试玩已打开';
      });
      frame.src = frame.dataset.src;
      stage.classList.add('is-started');
      fullscreen.disabled = false;
      status.textContent = '正在打开游戏页面…';
    });
  });

  frame.addEventListener('load', () => {
    if (started) updateDisplay();
  });

  window.addEventListener('message', (event) => {
    if (!started || event.origin !== location.origin || event.source !== frame.contentWindow) return;
    if (event.data?.type === 'ash-promise-ready') {
      gameResult = 'ready';
      updateDisplay();
    } else if (event.data?.type === 'ash-promise-error') {
      gameResult = 'error';
      updateDisplay();
    }
  });

  fullscreen.addEventListener('click', async () => {
    if (!started || switching) return;
    if (expanded) {
      collapse();
      return;
    }
    switching = true;
    fullscreen.disabled = true;
    try {
      if (document.fullscreenElement === embed) {
        await document.exitFullscreen();
      } else if (typeof embed.requestFullscreen === 'function' && document.fullscreenEnabled !== false) {
        try {
          await embed.requestFullscreen();
        } catch {
          expand();
        }
      } else {
        expand();
      }
    } catch {
      status.textContent = '暂时无法退出全屏，请再点一次退出全屏。';
    } finally {
      switching = false;
      fullscreen.disabled = false;
    }
  });

  document.addEventListener('fullscreenchange', () => {
    updateDisplay();
    if (document.fullscreenElement === embed) frame.focus({ preventScroll: true });
  });

  document.addEventListener('keydown', (event) => {
    if (expanded && event.key === 'Escape') collapse();
  });

  updateDisplay();
});
