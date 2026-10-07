document.querySelectorAll('[data-game-embed]').forEach((embed) => {
  const frame = embed.querySelector('[data-game-frame]');
  const stage = embed.querySelector('[data-game-stage]');
  const status = embed.querySelector('[data-game-status]');
  const fullscreen = embed.querySelector('[data-game-fullscreen]');
  let started = false;
  let gameResult = null;

  const startButtons = embed.querySelectorAll('[data-game-start]');
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
    if (!started) return;
    if (!gameResult) status.textContent = '游戏页面已打开，正在加载资源…';
  });

  window.addEventListener('message', (event) => {
    if (!started || event.origin !== location.origin || event.source !== frame.contentWindow) return;
    if (event.data?.type === 'ash-promise-ready') {
      gameResult = 'ready';
      status.textContent = '游戏已准备就绪。点击游戏画面开始操作；全屏游玩时文字更清楚。';
    } else if (event.data?.type === 'ash-promise-error') {
      gameResult = 'error';
      status.textContent = '游戏载入失败。请检查网络后刷新页面，或在独立窗口中重试。';
    }
  });

  fullscreen.addEventListener('click', async () => {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else await embed.requestFullscreen();
    } catch {
      status.textContent = '无法切换全屏。请在独立窗口中打开，或尝试浏览器的全屏模式（通常为 F11）。';
    }
  });

  document.addEventListener('fullscreenchange', () => {
    const isFullscreen = document.fullscreenElement === embed;
    fullscreen.textContent = isFullscreen ? '退出全屏' : '全屏游玩';
    fullscreen.setAttribute('aria-pressed', String(isFullscreen));
    if (gameResult === 'ready') status.textContent = isFullscreen ? '已进入全屏。退出后可继续当前游戏。' : '游戏已准备就绪。点击游戏画面开始操作；全屏游玩时文字更清楚。';
    if (started) frame.focus();
  });
});
