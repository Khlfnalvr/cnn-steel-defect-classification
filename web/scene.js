// ===== Tampilan 3D konvolusi (three.js) =====
// Tata letak, fase animasi, dan kamera dibangun dari rencana lapisan (M.plan), sehingga
// LeNet-5, AlexNet, dan Mini-VGG memakai kode yang sama. setModel() membangun ulang adegan.
function makeScene(root, classes) {
  const q = s => root.querySelector(s);
  const stage = q('.s3-stage'), labelsEl = q('.s3-labels');
  if (typeof THREE === 'undefined') { stage.innerHTML = '<p class="s3-fallback">Tampilan 3D tidak dapat dimuat. Periksa koneksi lalu muat ulang halaman.</p>'; return null; }
  let renderer;
  try { renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false }); }
  catch (e) { stage.innerHTML = '<p class="s3-fallback">Browser ini tidak mendukung WebGL, jadi tampilan 3D tidak tersedia.</p>'; return null; }
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.setClearColor(0xf4f4f4);
  stage.prepend(renderer.domElement);
  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0xf4f4f4, 55, 130);
  const camera = new THREE.PerspectiveCamera(38, 2, 0.1, 400);
  const ANISO = renderer.capabilities.getMaxAnisotropy();

  // ---------- warna ----------
  let TH = { bg: '#f4f4f4', grid: '#e8e8e8', empty: '#ffffff', m0: [236, 236, 236], m1: [13, 13, 13], kernel: '#ff4f1f', out: '#0a84ff', frame: '#e8e8e8', ghost: '#8c8c8c', text: '#0d0d0d', muted: '#666666' };
  function cmap(t) {
    t = Math.max(0, Math.min(1, t));
    if (TH.ramp) { const R = TH.ramp; for (let i = 1; i < R.length; i++) if (t <= R[i][0]) { const [t0, a] = R[i - 1], [t1, c] = R[i], f = (t - t0) / (t1 - t0); return `rgb(${a[0] + (c[0] - a[0]) * f | 0},${a[1] + (c[1] - a[1]) * f | 0},${a[2] + (c[2] - a[2]) * f | 0})`; } }
    t = Math.pow(t, 0.7); const a = TH.m0, c = TH.m1;
    return `rgb(${a[0] + (c[0] - a[0]) * t | 0},${a[1] + (c[1] - a[1]) * t | 0},${a[2] + (c[2] - a[2]) * t | 0})`;
  }
  let GRID = TH.grid, EMPTY = TH.empty;
  const FRAMES = [], GHOSTS = [];

  // ---------- bidang lapisan ----------
  // bidang tegak lurus sumbu x, menghadap kamera di sisi -x. Kolom gambar ke arah +z.
  function makePlane(o) {
    const p = Object.assign({ drawn: 0, ghosts: 0 }, o);
    const big = Math.max(p.n, p.m);
    p.cs = o.cs || Math.max(8, Math.floor(512 / big));
    p.canvas = document.createElement('canvas');
    p.canvas.width = p.m * p.cs + 1; p.canvas.height = p.n * p.cs + 1;
    p.ctx = p.canvas.getContext('2d');
    p.tex = new THREE.CanvasTexture(p.canvas);
    p.tex.anisotropy = ANISO; p.tex.minFilter = THREE.LinearMipmapLinearFilter;
    const mat = new THREE.MeshBasicMaterial({ map: p.tex, side: THREE.DoubleSide });
    p.mesh = new THREE.Mesh(new THREE.PlaneGeometry(p.w, p.h), mat);
    p.mesh.rotation.y = -Math.PI / 2; p.mesh.position.x = p.x;
    scene.add(p.mesh);
    const edge = rectPts(p, 0, 0, p.n - 1, p.m - 1, 0);
    p.frame = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(edge), new THREE.LineBasicMaterial({ color: TH.frame }));
    scene.add(p.frame); FRAMES.push(p.frame.material);
    p.objs = [p.mesh, p.frame]; p.shown = true;
    // irisan bayangan, tanda ada banyak kanal di belakangnya
    for (let k = 1; k <= p.ghosts; k++) {
      const pts = rectPts(p, 0, 0, p.n - 1, p.m - 1, -k * 0.22).map(v => v.add(new THREE.Vector3(0, k * 0.1, k * 0.1)));
      const g = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: TH.ghost, transparent: true, opacity: 0.5 / k }));
      scene.add(g); GHOSTS.push(g.material); p.objs.push(g);
    }
    p.full = () => { p.ctx.fillStyle = GRID; p.ctx.fillRect(0, 0, p.canvas.width, p.canvas.height); for (let i = 0; i < p.n * p.m; i++) cell(p, i, false); p.drawn = 0; };
    return p;
  }
  function cell(p, i, on) {
    const r = (i / p.m) | 0, c = i % p.m, cs = p.cs;
    if (p.drawRow) return p.drawRow(p, i, on);
    if (p.N != null && i >= p.N) { p.ctx.fillStyle = GRID; p.ctx.fillRect(c * cs + 1, r * cs + 1, cs - 1, cs - 1); return; }
    p.ctx.fillStyle = on && p.data ? cmap(p.data[i] / p.max) : EMPTY;
    p.ctx.fillRect(c * cs + 1, r * cs + 1, cs - 1, cs - 1);
  }
  function reveal(p, count) {
    if (count < p.drawn) p.full();
    for (let i = p.drawn; i < count; i++) cell(p, i, true);
    if (count !== p.drawn) { p.drawn = count; p.tex.needsUpdate = true; }
  }
  // sudut persegi sel r0..r1, c0..c1 dalam koordinat dunia, digeser dx ke arah kamera
  function rectPts(p, r0, c0, r1, c1, dx = -0.03) {
    const zL = -p.w / 2 + c0 / p.m * p.w, zR = -p.w / 2 + (c1 + 1) / p.m * p.w;
    const yT = p.h / 2 - r0 / p.n * p.h, yB = p.h / 2 - (r1 + 1) / p.n * p.h, x = p.x + dx;
    return [new THREE.Vector3(x, yT, zL), new THREE.Vector3(x, yT, zR), new THREE.Vector3(x, yB, zR), new THREE.Vector3(x, yB, zL)];
  }
  // softmax digambar sebagai baris berlabel dengan batang probabilitas
  function makeSoftmax(o) {
    const NC = classes.length, rh = 40, W = 300;
    const p = makePlane(Object.assign({ n: NC, m: 1, cs: 0 }, o));
    p.canvas.width = W; p.canvas.height = NC * rh; p.rh = rh;
    p.drawRow = (p, i, on) => {
      const y = i * rh, x = p.ctx, best = i === p.best;
      x.fillStyle = EMPTY; x.fillRect(1, y + 1, W - 2, rh - 1);
      x.fillStyle = on ? (best ? TH.kernel : TH.muted) : TH.grid;
      x.font = '600 17px "Geist Mono", ui-monospace, monospace'; x.textBaseline = 'middle';
      x.fillText(classes[i], 10, y + rh / 2 + 1);
      if (on && p.data) { const v = p.data[i]; x.fillStyle = best ? TH.kernel : cmap(0.3 + v * 0.7); x.fillRect(10, y + rh - 11, Math.max(2, v * (W - 20)), 6); }
    };
    p.full = () => { p.ctx.fillStyle = GRID; p.ctx.fillRect(0, 0, W, NC * rh); for (let i = 0; i < NC; i++) p.drawRow(p, i, false); p.drawn = 0; };
    p.tex.dispose(); p.tex = new THREE.CanvasTexture(p.canvas); p.tex.anisotropy = ANISO; p.mesh.material.map = p.tex;
    return p;
  }

  // ---------- kernel, sorotan, dan garis medan reseptif ----------
  const mkLoop = (color, op) => { const l = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(Array(4).fill(new THREE.Vector3())), new THREE.LineBasicMaterial({ color, transparent: true, opacity: op })); scene.add(l); return l; };
  const srcBox = mkLoop(TH.kernel, 1), dstBox = mkLoop(TH.out, 1);
  const fill = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ color: TH.kernel, transparent: true, opacity: 0.14, side: THREE.DoubleSide, depthWrite: false }));
  fill.rotation.y = -Math.PI / 2; scene.add(fill);
  const rays = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(Array(8).fill(new THREE.Vector3())), new THREE.LineBasicMaterial({ color: TH.kernel, transparent: true, opacity: 0.6 }));
  scene.add(rays);
  [srcBox, dstBox, rays, fill].forEach(o => o.frustumCulled = false);
  const setLoop = (l, pts) => { l.geometry.setFromPoints(pts); l.geometry.attributes.position.needsUpdate = true; };
  function setCursor(src, dst) {
    const show = !!src;
    [srcBox, dstBox, rays, fill].forEach(o => o.visible = show);
    if (!show) return;
    setLoop(srcBox, src); setLoop(dstBox, dst);
    const ray = []; for (let k = 0; k < 4; k++) ray.push(src[k], dst[k]);
    rays.geometry.setFromPoints(ray); rays.geometry.attributes.position.needsUpdate = true;
    const cy = (src[0].y + src[2].y) / 2, cz = (src[0].z + src[1].z) / 2;
    fill.position.set(src[0].x - 0.005, cy, cz); fill.scale.set(Math.abs(src[1].z - src[0].z), Math.abs(src[0].y - src[2].y), 1);
  }

  // ---------- keadaan model ----------
  let M = null, P = {}, LBL = {}, PH = [], CK = [], OWN = {}, CH = {}, TOTAL = 0;
  let ch = null, t = 0, playing = false, speed = 4, onState = () => {};
  let MID = 0, XSTART = 0, OVER = 1.6, NPL = 0;
  const view = { az: -1.0, el: 0.16, dist: 1.3, tx: 0, ty: 0, tdist: 1.3, ttx: 0 };
  const AZ0 = -1.0, EL0 = 0.16, BASE = 34, MAXSTEPS = 1024, MAXVEC = 256;   // batas langkah animasi per lapisan peta / lapisan dense

  const log2 = Math.log2;
  const sizeOf = n => Math.min(6.6, Math.max(1.6, 1.5 * log2(Math.max(1, n)) - 1.6));
  const ghostsOf = C => Math.min(5, Math.max(1, Math.round(log2(C)) - 1));

  function clearModel() {
    for (const k in P) { const p = P[k]; for (const o of p.objs) { scene.remove(o); if (o.geometry) o.geometry.dispose(); if (o.material) o.material.dispose(); } p.tex.dispose(); }
    for (const k in LBL) LBL[k].remove();
    FRAMES.length = 0; GHOSTS.length = 0; P = {}; LBL = {}; PH = []; CK = []; OWN = {}; CH = {};
  }

  function setModel(m) {
    clearModel(); M = m; ch = null; playing = false;
    const size = M.size, specs = [];
    specs.push({ key: 'inp', kind: 'map', n: size, m: size, w: sizeOf(size), h: sizeOf(size), ghosts: 0 });
    for (const l of M.plan) {
      if (l.type === 'dense' && l.act === 'softmax') specs.push({ key: l.id, kind: 'sm', w: 3.2, h: classes.length * 0.43 });
      else if (l.kind === 'map') specs.push({ key: l.id, kind: 'map', n: l.out[1], m: l.out[2], w: sizeOf(l.out[1]), h: sizeOf(l.out[1]), ghosts: ghostsOf(l.out[0]) });
      else { const N = l.size, m = N >= 1024 ? 32 : N >= 256 ? 16 : 8, n = Math.ceil(N / m); specs.push({ key: l.id, kind: 'vec', N, n, m, w: m * 0.14, h: n * 0.14, cs: 16 }); }
    }
    let x = 0;
    specs.forEach((s, i) => {
      if (i) x += 2.3 + 0.9 * (specs[i - 1].w + s.w) / 2;
      s.x = x;
      P[s.key] = s.kind === 'sm' ? makeSoftmax(s) : makePlane(s);
      P[s.key].kind = s.kind;
    });
    XSTART = 0; MID = x / 2; OVER = Math.max(1.0, x / 39); NPL = specs.length;
    scene.fog.near = BASE * OVER * 0.95; scene.fog.far = BASE * OVER * 2.4;

    // fase animasi: satu fase per lapisan. Peta besar dilewati dengan langkah > 1 piksel agar tetap ringkas.
    let prev = 'inp', acc = 0;
    for (const l of M.plan) {
      const D = P[l.id], full = l.kind === 'map' ? l.out[1] * l.out[2] : l.size;
      const cap = l.kind === 'map' ? MAXSTEPS : MAXVEC, stride = full > cap ? Math.ceil(full / cap) : 1;
      const kind = l.kind === 'map' ? (l.type === 'conv' ? 'conv' : 'pool') : 'vec';
      PH.push({ k: l.id, from: prev, kind, l, Nfull: full, stride, N: Math.ceil(full / stride), name: l.title, start: acc });
      acc += Math.ceil(full / stride); prev = l.id;
      if (l.type === 'conv' && l.kind === 'map') { CK.push(l.id); CH[l.id] = 0; }
    }
    TOTAL = acc; t = TOTAL;
    // pemilik kanal: konvolusi memilih kanalnya sendiri, pooling mengikuti konvolusi sebelumnya
    let own = null;
    for (const l of M.plan) { if (l.kind !== 'map') { own = null; continue; } if (l.type === 'conv') own = l.id; OWN[l.id] = own; }
    OWN.inp = null;
    // label HTML
    for (const k in P) { const d = document.createElement('div'); d.className = 's3-lbl'; labelsEl.append(d); LBL[k] = d; }
    for (const k in P) P[k].full();
    reveal(P.inp, size * size);
    labelAll();
    view.tx = view.ttx = MID; view.tdist = view.dist = OVER; view.az = AZ0; view.el = EL0;
    setCursor(null);
    showLayers();
    onState({ t, total: TOTAL, phase: null });
  }

  const argmaxCh = (a, C, S) => { let b = 0, bv = -1; for (let c = 0; c < C; c++) { let s = 0, s2 = 0; for (let i = 0; i < S; i++) { const v = a[c * S + i]; s += v; s2 += v * v; } const sd = s2 / S - (s / S) ** 2; if (sd > bv) { bv = sd; b = c; } } return b; };
  const slice = (a, c, S) => a.subarray(c * S, (c + 1) * S);
  const mx = a => { let m = 1e-6; for (const v of a) if (v > m) m = v; return m; };
  const planOf = id => M.plan.find(l => l.id === id);

  function labelAll() {
    LBL.inp.innerHTML = `<b>Masukan</b><span>${M.size} × ${M.size}</span>`;
    for (const l of M.plan) {
      const d = LBL[l.id];
      if (l.type === 'dense' && l.act === 'softmax') d.innerHTML = `<b>${l.short}</b><span>${classes.length} kelas${ch ? ` · <em>${classes[ch.best]}</em>` : ''}</span>`;
      else if (l.kind === 'map') { const o = OWN[l.id]; d.innerHTML = `<b>${l.short}</b><span>${l.out[1]} × ${l.out[2]}${o != null ? ` · kanal ${CH[o] + 1}/${planOf(o).out[0]}` : ''}</span>`; }
      else d.innerHTML = `<b>${l.short}</b><span>${l.size}</span>`;
    }
  }

  function loadData(keepCh) {
    if (!ch) return;
    if (!keepCh) CK.forEach(k => { const l = planOf(k); CH[k] = argmaxCh(ch.acts[k], l.out[0], l.out[1] * l.out[2]); });
    const set = (p, d, m) => { p.data = d; p.max = m || mx(d); };
    set(P.inp, ch.x, 1);
    for (const l of M.plan) {
      const p = P[l.id], a = ch.acts[l.id];
      if (l.type === 'dense' && l.act === 'softmax') { set(p, a, 1); p.best = ch.best; }
      else if (l.kind === 'map') {
        const S = l.out[1] * l.out[2], o = OWN[l.id], d = slice(a, CH[o], S);
        if (l.type === 'conv') set(p, d); else set(p, d, P[o].max);
      } else set(p, a);
    }
    for (const k in P) P[k].full();
    reveal(P.inp, M.size * M.size);
    labelAll();
    apply();
  }

  // posisi kursor untuk langkah global t
  function phaseAt(tt) { for (const ph of PH) if (tt < ph.start + ph.N) return ph; return null; }
  // lapisan di belakang lapisan yang sedang dioperasikan disembunyikan dulu; muncul saat fasenya mulai
  function showLayers() {
    for (const k in P) {
      const ph = PH.find(x => x.k === k), p = P[k], vis = !ph || t >= ph.start;
      p.shown = vis; for (const o of p.objs) o.visible = vis;
    }
  }
  function apply() {
    if (!M) return;
    showLayers();
    for (const ph of PH) reveal(P[ph.k], Math.max(0, Math.min(ph.Nfull, (t - ph.start) * ph.stride)));
    const ph = phaseAt(t);
    if (!ph) { setCursor(null); onState({ t, total: TOTAL, phase: null }); return; }
    const i = t - ph.start, j = Math.min(ph.Nfull - 1, i * ph.stride), S = P[ph.from], D = P[ph.k], l = ph.l;
    let src, dst, info = { t, total: TOTAL, phase: ph.name, key: ph.k, i, j, kind: ph.kind, l };
    if (ph.kind === 'conv') {
      const r = (j / D.m) | 0, c = j % D.m, p0 = l.pad, st = l.s || 1, r0 = r * st - p0, c0 = c * st - p0;
      src = rectPts(S, r0, c0, r0 + l.k - 1, c0 + l.k - 1); dst = rectPts(D, r, c, r, c); info.r = r; info.c = c;
    } else if (ph.kind === 'pool') {
      const r = (j / D.m) | 0, c = j % D.m;
      src = rectPts(S, l.s * r, l.s * c, l.s * r + l.k - 1, l.s * c + l.k - 1); dst = rectPts(D, r, c, r, c); info.r = r; info.c = c;
    } else {
      src = rectPts(S, 0, 0, S.n - 1, S.m - 1); dst = rectPts(D, (j / D.m) | 0, j % D.m, (j / D.m) | 0, j % D.m);
    }
    setCursor(src, dst);
    onState(info);
  }

  // ---------- kamera ----------
  function focus() {
    const ph = playing || t < TOTAL ? phaseAt(t) : null;
    if (!ph) { const nar = stage.clientWidth < 560; view.ttx = nar ? 6 : MID; view.tdist = nar ? 0.62 : OVER; return; }
    view.ttx = (P[ph.from].x + P[ph.k].x) / 2; view.tdist = 0.42;
  }
  function placeCamera() {
    const d = BASE * view.dist;
    camera.position.set(view.tx - d * Math.cos(view.el) * Math.cos(view.az), view.ty + d * Math.sin(view.el), -d * Math.cos(view.el) * Math.sin(view.az));
    camera.lookAt(view.tx, view.ty, 0);
  }
  let drag = null;
  const cv = renderer.domElement;
  cv.addEventListener('pointerdown', e => { drag = { x: e.clientX, y: e.clientY, az: view.az, el: view.el, id: e.pointerId }; });
  cv.addEventListener('pointermove', e => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (Math.abs(dx) > 4 && !drag.cap) { drag.cap = true; try { cv.setPointerCapture(drag.id); } catch (_) {} }
    view.az = drag.az - dx * 0.006; view.el = Math.max(-0.4, Math.min(1.1, drag.el + dy * 0.004));
  });
  const up = () => { drag = null; };
  cv.addEventListener('pointerup', up); cv.addEventListener('pointercancel', up);

  // ---------- ukuran dan loop ----------
  function resize() {
    const w = stage.clientWidth, h = stage.clientHeight;
    renderer.setSize(w, h, false); camera.aspect = w / h;
    camera.fov = w < 560 ? 62 : 36; camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(stage); resize();
  const v3 = new THREE.Vector3();
  function placeLabels() {
    labelsEl.classList.toggle('rest', !phaseAt(t));
    const w = stage.clientWidth, h = stage.clientHeight, ph = phaseAt(t);
    let li = 0;
    for (const k in P) {
      const p = P[k]; v3.set(p.x, p.h / 2 + 0.25, 0).project(camera); const lift = NPL > 9 && (li++ % 2) ? 20 : 0;
      const d = LBL[k], vis = v3.z < 1 && v3.x > -1.2 && v3.x < 1.2;
      d.style.transform = `translate(${(v3.x * 0.5 + 0.5) * w}px, ${(-v3.y * 0.5 + 0.5) * h}px) translate(-50%, -100%) translateY(${-lift}px)`;
      d.style.opacity = vis && p.shown ? 1 : 0;
      const on = !!ph && (ph.k === k || ph.from === k); d.classList.toggle('on', on); d.classList.toggle('dim', !!ph && !on);
    }
  }
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  let last = 0;
  function frame() {
    requestAnimationFrame(frame);
    if (!M) return;
    if (playing) {
      const ph = phaseAt(t);
      // fase pendek dibuat lebih lambat agar tetap terlihat
      const sp = ph && ph.N < 10 ? Math.min(speed, 0.08) : ph && ph.N <= 64 ? Math.max(0.5, speed / 4) : speed;
      last += sp; const n = Math.floor(last); last -= n;
      t = Math.min(TOTAL, t + n);
      if (t >= TOTAL) { playing = false; }
      apply();
    }
    focus();
    const k = reduce ? 1 : 0.06;
    view.tx += (view.ttx - view.tx) * k; view.dist += (view.tdist - view.dist) * k;
    placeCamera(); renderer.render(scene, camera); placeLabels();
  }
  requestAnimationFrame(frame);

  return {
    setModel,
    setResult(r) { ch = r; t = TOTAL; playing = false; loadData(false); },
    play() { if (t >= TOTAL) t = 0; playing = true; last = 0; },
    pause() { playing = false; },
    toggle() { playing ? this.pause() : this.play(); return playing; },
    restart() { t = 0; playing = true; apply(); },
    seek(v) { playing = false; t = Math.max(0, Math.min(TOTAL, v | 0)); apply(); },
    setSpeed(s) { speed = s; },
    get playing() { return playing; },
    get t() { return t; },
    get TOTAL() { return TOTAL; },
    get convIds() { return CK.slice(); },
    chOf(id) { return CH[OWN[id]] || 0; },         // kanal yang sedang ditampilkan pada lapisan peta id
    channel(id, d) {
      const C = planOf(id).out[0];
      CH[id] = (CH[id] + d + C) % C;
      loadData(true);
      return CH[id];
    },
    setCh(id, v) { CH[id] = v; },
    resetView() { view.az = AZ0; view.el = EL0; },
    onState(f) { onState = f; },
    setTheme(th) {
      TH = th; GRID = th.grid; EMPTY = th.empty;
      renderer.setClearColor(th.bg); scene.fog.color.set(th.bg);
      FRAMES.forEach(m => m.color.set(th.frame)); GHOSTS.forEach(m => m.color.set(th.ghost));
      srcBox.material.color.set(th.kernel); fill.material.color.set(th.kernel); rays.material.color.set(th.kernel); dstBox.material.color.set(th.out);
      if (ch) loadData(true); else for (const k in P) { P[k].full(); P[k].tex.needsUpdate = true; }
    },
  };
}
