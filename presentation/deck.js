/* lensing deck engine — dependency-free.
   Visual-first: short on-slide text, detail in speaker notes. Each slide is a
   fixed 16:9 canvas; its elements reveal staggered on enter; one keypress
   advances. No-JS / print / reduced-motion / ?all show everything at once. */
(() => {
  'use strict';
  const html = document.documentElement;
  html.classList.add('js');

  const NS = 'http://www.w3.org/2000/svg';
  const qs = new URLSearchParams(location.search);
  const REVEAL_ALL = qs.has('all');
  const PRINT = qs.has('print');
  const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const ANIM = !(REVEAL_ALL || PRINT || reduceMotion);
  if (REVEAL_ALL) html.classList.add('reveal-all');
  if (PRINT) html.classList.add('print-mode');

  const slides = [...document.querySelectorAll('.slide')];
  const total = slides.length;
  const pad2 = (n) => String(n).padStart(2, '0');

  slides.forEach((s) => [...s.querySelectorAll('.frag')].forEach((f) => {
    const st = parseInt(f.dataset.step || '0', 10);
    f.style.setProperty('--i', isNaN(st) ? 0 : st);
  }));

  const bz = {
    section: document.getElementById('bz-section'),
    cur: document.getElementById('bz-cur'),
    total: document.getElementById('bz-total'),
    bar: document.getElementById('bz-bar'),
  };
  if (bz.total) bz.total.textContent = String(total);

  let i = -1;
  function show(idx) {
    idx = Math.max(0, Math.min(total - 1, idx));
    if (idx === i) return;
    i = idx;
    slides.forEach((s, k) => {
      s.classList.toggle('is-active', k === idx);
      s.classList.toggle('is-prev', k < idx);
      if (k !== idx) s.querySelectorAll('.frag.is-in').forEach((f) => f.classList.remove('is-in'));
    });
    slides[idx].querySelectorAll('.frag').forEach((f) => f.classList.add('is-in'));
    updateChrome();
    if (location.hash !== '#' + (idx + 1)) history.replaceState(null, '', '#' + (idx + 1));
    onEnter(slides[idx]);
  }
  function updateChrome() {
    const s = slides[i];
    if (bz.section) bz.section.textContent = s.dataset.section || '';
    if (bz.cur) bz.cur.textContent = pad2(i + 1);
    if (bz.bar) bz.bar.style.width = ((i + 1) / total) * 100 + '%';
    if (notesPanel.classList.contains('show')) fillNotes();
    syncOverview();
  }
  const next = () => show(i + 1);
  const prev = () => show(i - 1);

  /* ---- enter hooks ------------------------------------------------------ */
  const replays = {};
  function onEnter(s) {
    s.querySelectorAll('.draw').forEach((el) => { if (ANIM) drawLine(el); });
    s.querySelectorAll('.live-head').forEach((el) => pulse(el));
    s.querySelectorAll('[data-fx]').forEach((el) => { const r = replays[el.id]; if (r) r(); });
  }
  function drawLine(el) {
    try {
      el.style.strokeDasharray = 100;
      el.animate([{ strokeDashoffset: 100 }, { strokeDashoffset: 0 }],
        { duration: 1100, easing: 'cubic-bezier(0.22,1,0.36,1)', fill: 'forwards' });
    } catch (e) {}
  }
  function pulse(el) {
    if (!ANIM || el._p) return; el._p = true;
    const r = el.getAttribute('r');
    el.animate([{ opacity: 1, r }, { opacity: 0.35, r: parseFloat(r) * 2.1 }, { opacity: 1, r }],
      { duration: 1600, iterations: Infinity, easing: 'ease-in-out' });
  }

  /* ---- animated Einstein ring (the brand motif) ------------------------- */
  function mk(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function arcPath(cx, cy, r, a0, a1) {
    const p = (a) => [cx + r * Math.cos(a * Math.PI / 180), cy + r * Math.sin(a * Math.PI / 180)];
    const [x0, y0] = p(a0), [x1, y1] = p(a1);
    const large = (a1 - a0) % 360 > 180 ? 1 : 0;
    return `M${x0.toFixed(2)} ${y0.toFixed(2)} A${r} ${r} 0 ${large} 1 ${x1.toFixed(2)} ${y1.toFixed(2)}`;
  }
  function buildRing(svg) {
    const cx = 100, cy = 100, R = 72;
    const defs = mk('defs', {}, svg);
    const grad = mk('linearGradient', { id: svg.id + '-g', x1: '0', y1: '0', x2: '1', y2: '1' }, defs);
    mk('stop', { offset: '0', 'stop-color': 'oklch(0.7 0.13 245)' }, grad);   // blue
    mk('stop', { offset: '0.5', 'stop-color': 'oklch(0.78 0.16 55)' }, grad);  // orange
    mk('stop', { offset: '1', 'stop-color': 'oklch(0.58 0.2 26)' }, grad);     // red
    // halo + mass
    mk('circle', { cx, cy, r: 30, fill: 'oklch(0.1 0.02 255)', opacity: '0.5' }, svg);
    // faint locus
    mk('circle', { cx, cy, r: R, fill: 'none', stroke: 'oklch(0.5 0.02 250)', 'stroke-width': '1', 'stroke-dasharray': '1.5 3', opacity: '0.5' }, svg);
    // main ring (draws in)
    const C = 2 * Math.PI * R;
    const ring = mk('circle', { cx, cy, r: R, fill: 'none', stroke: 'oklch(0.62 0.02 250)', 'stroke-width': '2.4', 'stroke-linecap': 'round' }, svg);
    // mass
    const mass = mk('circle', { cx, cy, r: 17, fill: 'oklch(0.09 0.015 255)' }, svg);
    mk('circle', { cx, cy, r: 17, fill: 'none', stroke: 'oklch(0.5 0.03 250)', 'stroke-width': '1', opacity: '0.7' }, svg);
    // a couple of faint source points on the ring
    [40, 200].forEach((a) => mk('circle', { cx: cx + R * Math.cos(a * Math.PI / 180), cy: cy + R * Math.sin(a * Math.PI / 180), r: '2', fill: 'oklch(0.78 0.16 55)', opacity: '0.6' }, svg));
    // rotating Doppler sweep
    const spin = mk('g', {}, svg);
    mk('path', { d: arcPath(cx, cy, R, -34, 34), fill: 'none', stroke: `url(#${svg.id}-g)`, 'stroke-width': '4', 'stroke-linecap': 'round' }, spin);
    svg.style.setProperty('transform-box', 'view-box');
    spin.style.transformOrigin = `${cx}px ${cy}px`;
    mass.style.transformOrigin = `${cx}px ${cy}px`;
    let spun = false;
    return () => {
      if (!ANIM) return;
      ring.style.strokeDasharray = C;
      ring.animate([{ strokeDashoffset: C }, { strokeDashoffset: 0 }], { duration: 1100, easing: 'cubic-bezier(0.22,1,0.36,1)', fill: 'forwards' });
      mass.animate([{ transform: 'scale(0.2)', opacity: 0 }, { transform: 'scale(1)', opacity: 1 }], { duration: 700, easing: 'cubic-bezier(0.22,1,0.36,1)', fill: 'forwards' });
      if (!spun) { spun = true; spin.animate([{ transform: 'rotate(0deg)' }, { transform: 'rotate(360deg)' }], { duration: 16000, iterations: Infinity, easing: 'linear' }); }
    };
  }

  /* ---- comps map (representation) --------------------------------------- */
  function rng(seed) { return () => { seed |= 0; seed = (seed + 0x6D2B79F5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  function buildCompsMap(svg) {
    const W = 320, H = 200, cx = 165, cy = 96, r = rng(7);
    // faint streets
    const g = mk('g', { stroke: 'oklch(0.34 0.02 250)', 'stroke-width': '0.7', opacity: '0.5' }, svg);
    for (let k = 0; k < 7; k++) { const y = 18 + k * 26 + (r() - 0.5) * 8; mk('line', { x1: 0, y1: y, x2: W, y2: y + (r() - 0.5) * 14 }, g); }
    for (let k = 0; k < 9; k++) { const x = 14 + k * 36 + (r() - 0.5) * 8; mk('line', { x1: x, y1: 0, x2: x + (r() - 0.5) * 14, y2: H }, g); }
    // dots — neighbors scored model vs asking
    const dots = mk('g', {}, svg);
    for (let k = 0; k < 52; k++) {
      // gaussian-ish cluster
      const a = r() * 6.283, rad = (r() + r()) * 46;
      const x = cx + Math.cos(a) * rad * 1.25, y = cy + Math.sin(a) * rad * 0.8;
      if (x < 6 || x > W - 6 || y < 6 || y > H - 6) continue;
      const under = r() > 0.46;
      const c = mk('circle', { cx: x.toFixed(1), cy: y.toFixed(1), r: (1.6 + r() * 2.4).toFixed(1), fill: under ? 'var(--series-b)' : 'var(--bad)', opacity: '0' }, dots);
      if (ANIM) c.animate([{ opacity: 0, transform: 'scale(0.3)' }, { opacity: 0.85, transform: 'scale(1)' }], { duration: 500, delay: 200 + k * 14, easing: 'cubic-bezier(0.22,1,0.36,1)', fill: 'forwards' });
      else c.setAttribute('opacity', '0.85');
    }
    // the subject listing — gold star
    const star = mk('path', { d: starPath(cx, cy, 8.5), fill: 'var(--accent)', stroke: 'oklch(0.2 0.04 47)', 'stroke-width': '0.6' }, svg);
    star.style.transformOrigin = `${cx}px ${cy}px`;
    return () => { if (ANIM) star.animate([{ transform: 'scale(0)', opacity: 0 }, { transform: 'scale(1)', opacity: 1 }], { duration: 700, delay: 950, easing: 'cubic-bezier(0.22,1,0.36,1)', fill: 'both' }); };
  }
  function starPath(cx, cy, R) {
    let d = ''; for (let k = 0; k < 10; k++) { const rad = k % 2 ? R * 0.42 : R; const a = -Math.PI / 2 + k * Math.PI / 5; d += (k ? 'L' : 'M') + (cx + rad * Math.cos(a)).toFixed(1) + ' ' + (cy + rad * Math.sin(a)).toFixed(1) + ' '; } return d + 'Z';
  }

  function initFx() {
    // the Einstein-ring mark is the real mark-dark.svg, animated via CSS on .is-active
    const cm = document.getElementById('comps-map'); if (cm) { cm.dataset.fx = '1'; replays['comps-map'] = buildCompsMap(cm); }
  }

  /* ---- overview / notes / help / fullscreen ----------------------------- */
  const overview = document.getElementById('overview');
  function buildOverview() {
    slides.forEach((s, k) => {
      const t = document.createElement('div');
      t.className = 'ov-thumb';
      t.style.background = s.dataset.bg === 'slate' ? 'var(--chrome)' : 'var(--bg)';
      const h = s.querySelector('h1, .title-lg, .kicker');
      t.innerHTML = '<span class="n">' + pad2(k + 1) + '</span><span class="t">' + (h ? h.textContent.trim().slice(0, 70) : (s.dataset.section || '')) + '</span>';
      t.addEventListener('click', () => { hideOverview(); show(k); });
      overview.appendChild(t);
    });
  }
  const syncOverview = () => [...overview.children].forEach((c, k) => c.classList.toggle('cur', k === i));
  const showOverview = () => { overview.classList.add('show'); syncOverview(); };
  const hideOverview = () => overview.classList.remove('show');
  const toggleOverview = () => overview.classList.contains('show') ? hideOverview() : showOverview();

  const notesPanel = document.getElementById('notes');
  const notesBody = document.getElementById('notes-body');
  function fillNotes() { const a = slides[i].querySelector('aside.notes'); notesBody.innerHTML = a ? a.innerHTML : '<p class="muted">—</p>'; }
  function toggleNotes() { notesPanel.classList.toggle('show'); if (notesPanel.classList.contains('show')) fillNotes(); }

  const help = document.getElementById('help');
  const toggleHelp = () => help.classList.toggle('show');
  function toggleFs() { if (!document.fullscreenElement) document.documentElement.requestFullscreen?.(); else document.exitFullscreen?.(); }

  /* ---- input ------------------------------------------------------------ */
  document.addEventListener('keydown', (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const k = e.key;
    if (k === 'Escape') { help.classList.remove('show'); hideOverview(); return; }
    switch (k) {
      case 'ArrowRight': case ' ': case 'PageDown': case 'Enter': e.preventDefault(); next(); break;
      case 'ArrowLeft': case 'PageUp': case 'Backspace': e.preventDefault(); prev(); break;
      case 'ArrowDown': e.preventDefault(); next(); break;
      case 'ArrowUp': e.preventDefault(); prev(); break;
      case 'Home': e.preventDefault(); show(0); break;
      case 'End': e.preventDefault(); show(total - 1); break;
      case 'o': case 'O': e.preventDefault(); toggleOverview(); break;
      case 's': case 'S': e.preventDefault(); toggleNotes(); break;
      case 'f': case 'F': e.preventDefault(); toggleFs(); break;
      case '?': e.preventDefault(); toggleHelp(); break;
      default: if (k >= '1' && k <= '9') { const n = +k - 1; if (n < total) show(n); }
    }
  });
  document.querySelector('.frames').addEventListener('click', (e) => {
    if (e.target.closest('a, button')) return;
    (e.clientX < innerWidth * 0.18) ? prev() : next();
  });
  let tx = 0, ty = 0;
  addEventListener('touchstart', (e) => { tx = e.changedTouches[0].clientX; ty = e.changedTouches[0].clientY; }, { passive: true });
  addEventListener('touchend', (e) => { const dx = e.changedTouches[0].clientX - tx, dy = e.changedTouches[0].clientY - ty; if (Math.abs(dx) > 50 && Math.abs(dx) > Math.abs(dy)) dx < 0 ? next() : prev(); }, { passive: true });

  const hint = document.getElementById('hint');
  setTimeout(() => hint && hint.classList.add('fade'), 4200);

  /* ---- boot ------------------------------------------------------------- */
  initFx();
  buildOverview();
  show(Math.max(0, (parseInt((location.hash || '').slice(1), 10) || 1) - 1));
})();
