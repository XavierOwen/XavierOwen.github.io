(() => {
  const dialog = document.querySelector('.gospel-lightbox');
  if (!dialog) return;
  document.body.append(dialog);
  const background = [...document.body.children].filter(element => element !== dialog);
  const originalInert = background.map(element => element.inert);

  const image = dialog.querySelector('.gospel-lightbox__stage img');
  const caption = dialog.querySelector('figcaption');
  const thumbs = dialog.querySelector('.gospel-lightbox__thumbs');
  let items = [];
  let index = 0;
  let opener;

  function close() {
    dialog.hidden = true;
    document.body.classList.remove('gospel-lightbox-open');
    background.forEach((element, position) => { element.inert = originalInert[position]; });
    opener?.focus();
  }

  function show(next) {
    index = (next + items.length) % items.length;
    const selected = items[index];
    image.src = selected.href;
    image.alt = selected.querySelector('img').alt;
    caption.textContent = `${selected.closest('[data-gospel-gallery]').querySelector('h2').textContent} · ${index + 1} / ${items.length}`;
    [...thumbs.children].forEach((button, position) => button.setAttribute('aria-current', position === index));
    thumbs.children[index].scrollIntoView({ block: 'nearest', inline: 'nearest' });
  }

  document.querySelectorAll('[data-gallery-image]').forEach(link => {
    link.addEventListener('click', event => {
      event.preventDefault();
      opener = link;
      items = [...link.closest('[data-gospel-gallery]').querySelectorAll('[data-gallery-image]')];
      thumbs.replaceChildren(...items.map((item, position) => {
        const button = document.createElement('button');
        const preview = item.querySelector('img');
        button.type = 'button';
        button.setAttribute('aria-label', preview.alt);
        const thumb = document.createElement('img');
        thumb.src = preview.src;
        thumb.alt = '';
        button.append(thumb);
        button.addEventListener('click', () => show(position));
        return button;
      }));
      dialog.hidden = false;
      document.body.classList.add('gospel-lightbox-open');
      background.forEach(element => { element.inert = true; });
      show(items.indexOf(link));
      dialog.querySelector('.gospel-lightbox__close').focus();
    });
  });

  dialog.querySelector('.gospel-lightbox__close').addEventListener('click', close);
  dialog.querySelector('[data-gallery-prev]').addEventListener('click', () => show(index - 1));
  dialog.querySelector('[data-gallery-next]').addEventListener('click', () => show(index + 1));
  dialog.addEventListener('keydown', event => {
    if (event.key === 'Escape') {
      event.preventDefault();
      close();
    }
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
      event.preventDefault();
      show(index + (event.key === 'ArrowRight' ? 1 : -1));
    }
    if (event.key === 'Tab') {
      const buttons = [...dialog.querySelectorAll('button')];
      const edge = event.shiftKey ? buttons[0] : buttons[buttons.length - 1];
      if (document.activeElement === edge) {
        event.preventDefault();
        buttons[event.shiftKey ? buttons.length - 1 : 0].focus();
      }
    }
  });
  dialog.addEventListener('click', event => {
    if (event.target === dialog) close();
  });

  let touchStart;
  image.addEventListener('pointerdown', event => {
    if (event.pointerType === 'touch') touchStart = event.clientX;
  });
  image.addEventListener('pointerup', event => {
    if (touchStart === undefined) return;
    const distance = event.clientX - touchStart;
    touchStart = undefined;
    if (Math.abs(distance) > 50) show(index + (distance < 0 ? 1 : -1));
  });
})();
