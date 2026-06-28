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

  /* ---- slide 16: space-time well + precessing keplerian orbit ----------- */
  function buildWellOrbit(svg) {
    const grid = svg.querySelector('.g-grid'), trail = svg.querySelector('.g-trail'),
          back = svg.querySelector('.g-back'), mass = svg.querySelector('.g-mass'), front = svg.querySelector('.g-front');
    const clr = (g) => { while (g.firstChild) g.removeChild(g.firstChild); };
    const cx = 110, cy = 54, halfW = 100, halfD = 48, sig = 0.40;
    const project = (u, v, amp) => {
      const sag = amp * Math.exp(-(u * u + v * v) / (2 * sig * sig));
      const persp = 0.5 + 0.5 * ((v + 1) / 2);
      return [cx + u * halfW * persp, cy + v * halfD + sag];
    };
    function drawGrid(amp) {
      clr(grid); const N = 24;
      const ln = (d) => mk('path', { d, fill: 'none', stroke: 'oklch(0.74 0.02 250)', 'stroke-width': 0.45, opacity: 0.5 }, grid);
      for (let r = 0; r <= 12; r++) { const v = -1 + r * (2 / 12); let d = '';
        for (let j = 0; j <= N; j++) { const [x, y] = project(-1 + j * (2 / N), v, amp); d += (j ? 'L' : 'M') + x.toFixed(1) + ' ' + y.toFixed(1) + ' '; } ln(d); }
      for (let c = 0; c <= 12; c++) { const u = -1 + c * (2 / 12); let d = '';
        for (let j = 0; j <= N; j++) { const [x, y] = project(u, -1 + j * (2 / N), amp); d += (j ? 'L' : 'M') + x.toFixed(1) + ' ' + y.toFixed(1) + ' '; } ln(d); }
    }
    function drawMass(amp) {
      clr(mass); const [mx, my] = project(0, 0, amp);
      mk('circle', { cx: mx, cy: my, r: 21, fill: 'url(#wo-mglow)' }, mass);
      mk('circle', { cx: mx, cy: my, r: 11, fill: 'none', stroke: 'oklch(0.9 0.08 88)', 'stroke-width': 0.5, opacity: 0.28, filter: 'url(#wo-soft)' }, mass);
      mk('circle', { cx: mx, cy: my, r: 6.2, fill: 'oklch(0.14 0.02 262)' }, mass);
      mk('circle', { cx: mx, cy: my, r: 6.2, fill: 'none', stroke: 'oklch(0.88 0.12 85)', 'stroke-width': 0.8, opacity: 0.8, filter: 'url(#wo-soft)' }, mass);
    }
    const A = 0.62, E = 0.5, T = 6.0, WDOT = 0.22;
    function kepler(t) {
      let M = 2 * Math.PI * (t / T), Ec = M;
      for (let k = 0; k < 6; k++) Ec = Ec - (Ec - E * Math.sin(Ec) - M) / (1 - E * Math.cos(Ec));
      const nu = 2 * Math.atan2(Math.sqrt(1 + E) * Math.sin(Ec / 2), Math.sqrt(1 - E) * Math.cos(Ec / 2));
      const r = A * (1 - E * Math.cos(Ec)), ang = nu + WDOT * t;
      return [r * Math.cos(ang), r * Math.sin(ang)];
    }
    function drawOrbit(t, amp) {
      clr(trail); clr(back); clr(front);
      const NT = 130, step = 0.06; let px = null, py = null;
      for (let k = 0; k <= NT; k++) {
        const [u, v] = kepler(t - k * step), [x, y] = project(u, v, amp);
        if (px !== null) { const f = 1 - k / NT;
          mk('line', { x1: px, y1: py, x2: x, y2: y, stroke: 'oklch(0.86 0.12 232)', 'stroke-width': (0.35 + 0.7 * f).toFixed(2), 'stroke-linecap': 'round', opacity: (0.55 * f * f).toFixed(3) }, trail); }
        px = x; py = y;
      }
      const [u, v] = kepler(t), [x, y] = project(u, v, amp), near = (v + 1) / 2, rad = 1.5 + 1.8 * near, g = v >= 0 ? front : back;
      mk('circle', { cx: x, cy: y, r: rad * 2.8, fill: 'oklch(0.84 0.12 232)', opacity: 0.18, filter: 'url(#wo-soft)' }, g);
      mk('circle', { cx: x, cy: y, r: rad, fill: 'oklch(0.93 0.07 228)' }, g);
    }
    const render = (t) => { const amp = 37 + 3 * Math.sin(t * 0.55); drawGrid(amp); drawOrbit(t, amp); drawMass(amp); };
    render(0.8);                              // initial static frame
    if (ANIM) {
      const slide = svg.closest('.slide'); let t0 = null;
      (function loop(ts) { if (t0 === null) t0 = ts;
        if (!slide || slide.classList.contains('is-active')) render((ts - t0) / 1000);
        requestAnimationFrame(loop); })(performance.now());
    }
  }

  /* ---- task-flow graph: nodes = tasks, edges = artifacts (produced→consumed) -- */
  function buildTaskGraph(svg) {
    const gE = svg.querySelector('.tg-edges'), gN = svg.querySelector('.tg-nodes'), gL = svg.querySelector('.tg-labels');
    const NH = 94;
    const N = {
      corpus: { t: 'corpus', who: 'Qdrant', x: 88, y: 250, w: 124, ill: 'db', src: true },
      probe:  { t: 'sondea el corpus', who: 'bootstrap', x: 158, y: 95, w: 190, ill: 'magnifier' },
      data:   { t: 'construye el dataset', who: 'dataset-architect', x: 372, y: 188, w: 216, ill: 'matrix' },
      design: { t: 'propone el scan', who: 'experiment-designer', x: 372, y: 366, w: 208, ill: 'scangrid' },
      run:    { t: 'corre los runs', who: 'experiment-runner', x: 582, y: 250, w: 190, ill: 'heatmap' },
      pdf:    { t: 'consolida el informe', who: 'report-curator', x: 866, y: 98, w: 210, ill: 'report' },
      prom:   { t: 'cura los mejores', who: 'best-model-selector', x: 868, y: 306, w: 212, ill: 'podium' },
      onnx:   { t: 'exporta a ONNX', who: '/model-export', x: 1108, y: 170, w: 170, ill: 'export' },
      show:   { t: 'publica el showcase', who: 'showcase-builder', x: 1108, y: 310, w: 208, ill: 'browser' },
      list:   { t: 'puntúa items nuevos', who: 'listing-generator', x: 1108, y: 450, w: 198, ill: 'tag' },
    };
    const E = [
      ['corpus', 'probe', '', 't', 'b'],
      ['corpus', 'data', 'embeddings', 'r', 'l'],
      ['probe', 'data', 'domain.toml', 'r', 't'],
      ['data', 'run', 'dataset', 'r', 't'],
      ['design', 'run', 'diseño', 't', 'l'],
      ['run', 'design', 'PROJECT-FACTS', 'b', 'r', 'loop'],
      ['run', 'pdf', 'reportes', 'r', 'l'],
      ['run', 'prom', 'métricas', 'r', 'l'],
      ['prom', 'onnx', 'modelo', 'r', 'l'],
      ['prom', 'show', '', 'r', 'l'],
      ['prom', 'list', 'modelo', 'r', 'l'],
      ['onnx', 'show', 'bundle', 'b', 't'],
    ];
    const anchor = (n, s) => s === 'l' ? [n.x - n.w / 2, n.y] : s === 'r' ? [n.x + n.w / 2, n.y] : s === 't' ? [n.x, n.y - NH / 2] : [n.x, n.y + NH / 2];
    const push = (p, s, k) => s === 'l' ? [p[0] - k, p[1]] : s === 'r' ? [p[0] + k, p[1]] : s === 't' ? [p[0], p[1] - k] : [p[0], p[1] + k];
    const bez = (a, c1, c2, b, t) => { const u = 1 - t; return [u*u*u*a[0]+3*u*u*t*c1[0]+3*u*t*t*c2[0]+t*t*t*b[0], u*u*u*a[1]+3*u*u*t*c1[1]+3*u*t*t*c2[1]+t*t*t*b[1]]; };
    function glyph(kind, gx, gy) {
      const G = mk('g', {}, gN), S = (t, a) => mk(t, a, G);
      if (kind === 'db') {                                   // corpus: vector DB
        S('ellipse', { cx: gx, cy: gy-9, rx: 13, ry: 4.5, class: 'gi-stroke' });
        S('path', { d: `M${gx-13} ${gy-9} V${gy+9} A13 4.5 0 0 0 ${gx+13} ${gy+9} V${gy-9}`, class: 'gi-stroke' });
        S('path', { d: `M${gx-13} ${gy} A13 4.5 0 0 0 ${gx+13} ${gy}`, class: 'gi-stroke' });
        [[-5,-2],[3,-3],[-2,5],[5,4]].forEach(o => S('circle', { cx: gx+o[0], cy: gy+o[1], r: 1.3, class: 'gi-accent', opacity: 0.75 }));
      } else if (kind === 'magnifier') {                     // sondea: inspect the corpus
        [[-9,4],[-3,7],[3,5]].forEach(o => S('circle', { cx: gx+o[0], cy: gy+o[1], r: 1.7, class: 'gi-fill', opacity: 0.4 }));
        S('circle', { cx: gx-2, cy: gy-2, r: 8.5, class: 'gi-stroke' });
        S('line', { x1: gx+4, y1: gy+4, x2: gx+12, y2: gy+12, class: 'gi-stroke', 'stroke-width': 2.6 });
      } else if (kind === 'matrix') {                        // dataset: feature matrix (target col highlighted)
        S('rect', { x: gx-17, y: gy-13, width: 34, height: 26, rx: 2.5, class: 'gi-stroke' });
        S('rect', { x: gx-17, y: gy-13, width: 34, height: 7.5, class: 'gi-accent', opacity: 0.22 });
        S('rect', { x: gx+8.5, y: gy-5.5, width: 8.5, height: 18.5, class: 'gi-accent', opacity: 0.16 });
        [-5.7,2.5].forEach(dx => S('line', { x1: gx+dx, y1: gy-13, x2: gx+dx, y2: gy+13, class: 'gi-stroke', 'stroke-width': 0.8, opacity: 0.5 }));
        [-1,5.5].forEach(dy => S('line', { x1: gx-17, y1: gy+dy, x2: gx+17, y2: gy+dy, class: 'gi-stroke', 'stroke-width': 0.8, opacity: 0.5 }));
      } else if (kind === 'scangrid') {                      // scan: a grid of parameter combos
        S('path', { d: `M${gx-13} ${gy-13} V${gy+13} H${gx+15}`, class: 'gi-stroke', 'stroke-width': 1.4 });
        for (let r = 0; r < 3; r++) for (let k = 0; k < 3; k++) S('circle', { cx: gx-6+k*9, cy: gy+8-r*9, r: 1.8, class: (r===1&&k===1) ? 'gi-accent' : 'gi-fill', opacity: (r===1&&k===1) ? 1 : 0.5 });
      } else if (kind === 'heatmap') {                       // runs: parameter-scan results
        const v = [[0.85,0.3,0.55,0.15],[0.4,0.7,0.95,0.5],[0.2,0.45,0.6,0.8]], c = 7.2, g = 1.4, x0 = gx-(4*c+3*g)/2, y0 = gy-(3*c+2*g)/2;
        for (let r = 0; r < 3; r++) for (let k = 0; k < 4; k++) S('rect', { x: x0+k*(c+g), y: y0+r*(c+g), width: c, height: c, rx: 1, class: 'gi-accent', opacity: (0.15+0.8*v[r][k]).toFixed(2) });
      } else if (kind === 'report') {                        // report: a page with a chart
        S('path', { d: `M${gx-13} ${gy-14} h18 l6 6 v22 h-24 z`, class: 'gi-card' });
        S('path', { d: `M${gx+5} ${gy-14} v6 h6`, class: 'gi-stroke' });
        S('line', { x1: gx-9, y1: gy-7, x2: gx+4, y2: gy-7, class: 'gi-stroke', 'stroke-width': 1, opacity: 0.5 });
        [[-9.5,7],[-4,11],[1.5,8],[7,13]].forEach(b => S('rect', { x: gx+b[0], y: gy+10-b[1], width: 3.4, height: b[1], class: 'gi-accent', opacity: 0.8 }));
      } else if (kind === 'podium') {                        // best models: ranked, star on the winner
        [[-13,11],[-1,19],[11,9]].forEach((b, i) => S('rect', { x: gx+b[0], y: gy+13-b[1], width: 9, height: b[1], rx: 1, class: i===1 ? 'gi-accent' : 'gi-fill', opacity: i===1 ? 0.85 : 0.4 }));
        S('path', { d: 'M0 -4 L1.3 -1 L4.5 -0.8 L2.1 1.3 L2.9 4.5 L0 2.8 L-2.9 4.5 L-2.1 1.3 L-4.5 -0.8 L-1.3 -1 Z', transform: `translate(${gx+3.5} ${gy-10})`, class: 'gi-accent' });
      } else if (kind === 'export') {                        // export: model leaves as a bundle
        S('rect', { x: gx-15, y: gy-4, width: 17, height: 17, rx: 2.5, class: 'gi-card' });
        S('line', { x1: gx, y1: gy-2, x2: gx+13, y2: gy-13, class: 'gi-stroke', 'stroke-width': 1.9 });
        S('path', { d: `M${gx+7} ${gy-13} H${gx+13} V${gy-7}`, class: 'gi-stroke', 'stroke-width': 1.9 });
      } else if (kind === 'browser') {                       // showcase: a live demo page
        S('rect', { x: gx-17, y: gy-13, width: 34, height: 26, rx: 2.5, class: 'gi-stroke' });
        S('line', { x1: gx-17, y1: gy-5, x2: gx+17, y2: gy-5, class: 'gi-stroke', 'stroke-width': 1.2 });
        [-13,-9.5,-6].forEach(dx => S('circle', { cx: gx+dx, cy: gy-9, r: 1.1, class: 'gi-fill' }));
        S('polyline', { points: `${gx-12},${gy+8} ${gx-5},${gy+1} ${gx+1},${gy+4} ${gx+12},${gy-3}`, class: 'gi-astroke', 'stroke-width': 1.6 });
      } else if (kind === 'tag') {                           // listing: a scored item (price tag)
        S('path', { d: `M${gx-15} ${gy} L${gx-5} ${gy-12} H${gx+15} V${gy+12} H${gx-5} Z`, class: 'gi-card' });
        S('circle', { cx: gx-7, cy: gy, r: 2.1, class: 'gi-stroke' });
        S('line', { x1: gx, y1: gy-3, x2: gx+11, y2: gy-3, class: 'gi-stroke', 'stroke-width': 1.4, opacity: 0.6 });
        S('line', { x1: gx, y1: gy+3.5, x2: gx+8, y2: gy+3.5, class: 'gi-stroke', 'stroke-width': 1.4, opacity: 0.6 });
      }
    }
    E.forEach(([fk, tk, label, fs, ts, type, K]) => {
      K = K || 58;
      const a = anchor(N[fk], fs), b = anchor(N[tk], ts), c1 = push(a, fs, K), c2 = push(b, ts, K);
      const cls = 'tg-edge' + (type === 'loop' ? ' loop' : '');
      mk('path', { d: `M${a[0]} ${a[1]} C ${c1[0]} ${c1[1]} ${c2[0]} ${c2[1]} ${b[0]} ${b[1]}`, class: cls, 'marker-end': `url(#${type === 'loop' ? 'tg-ahl' : 'tg-ah'})` }, gE);
      if (label) {
        const m = bez(a, c1, c2, b, 0.5), w = label.length * 7.4 + 10;
        mk('rect', { x: m[0]-w/2, y: m[1]-9, width: w, height: 18, rx: 3, class: 'tg-elabel-bg' }, gL);
        mk('text', { x: m[0], y: m[1]+4.5, class: 'tg-elabel' + (type === 'loop' ? ' loop' : '') }, gL).textContent = label;
      }
    });
    Object.values(N).forEach((n) => {
      mk('rect', { x: n.x-n.w/2, y: n.y-NH/2, width: n.w, height: NH, rx: 11, class: 'tg-node' + (n.src ? ' src' : '') }, gN);
      glyph(n.ill, n.x, n.y-19);
      mk('text', { x: n.x, y: n.y+20, class: 'tg-title' }, gN).textContent = n.t;
      mk('text', { x: n.x, y: n.y+37, class: 'tg-who' }, gN).textContent = n.who;
    });
  }

  function initFx() {
    // the Einstein-ring mark is the real mark-dark.svg, animated via CSS on .is-active
    const cm = document.getElementById('comps-map'); if (cm) { cm.dataset.fx = '1'; replays['comps-map'] = buildCompsMap(cm); }
    const wo = document.getElementById('well-orbit'); if (wo) buildWellOrbit(wo);
    const tg = document.getElementById('task-graph'); if (tg) buildTaskGraph(tg);
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
