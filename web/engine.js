// ===== Mesin CNN untuk cacat permukaan baja NEU (LeNet-5, AlexNet, dan Mini-VGG) =====
// Arsitektur dibaca dari spesifikasi di weights.json, jadi satu mesin menjalankan ketiga model.
// Dipakai di browser dan bisa diuji di Node.
const NEU = (() => {
  // ---- bobot ----
  function b64bytes(s) {
    if (typeof atob === 'function') { const r = atob(s), u = new Int8Array(r.length); for (let i = 0; i < r.length; i++) u[i] = (r.charCodeAt(i) << 24) >> 24; return u; }
    return new Int8Array(Buffer.from(s, 'base64'));
  }
  const f32 = a => Float32Array.from(a);

  // bentuk keluaran tiap lapisan. kind: 'map' (C x H x W) atau 'vec' (N)
  function plan(arch) {
    let C = 1, H = arch.input, W = arch.input, prev = [1, H, W];
    const P = [];
    for (const l of arch.layers) {
      const o = Object.assign({}, l, { in: prev });
      if (l.type === 'conv') { const s = l.s || 1; H = ((H + 2 * l.pad - l.k) / s | 0) + 1; W = ((W + 2 * l.pad - l.k) / s | 0) + 1; C = l.cout; o.out = [C, H, W]; o.s = s; }
      else if (l.type === 'maxpool' || l.type === 'avgpool') { H = ((H - l.k) / l.s | 0) + 1; W = ((W - l.k) / l.s | 0) + 1; o.out = [C, H, W]; }
      else if (l.type === 'gap') { o.out = [C]; }
      else if (l.type === 'dense') { o.out = [l.out]; }
      o.kind = o.out.length === 3 && o.out[1] * o.out[2] > 1 ? 'map' : 'vec';
      o.size = o.out.reduce((a, b) => a * b, 1);          // jumlah nilai keluaran
      P.push(o); prev = o.out;
    }
    return P;
  }

  function load(json) {
    const out = {};
    for (const key in json) {
      const m = json[key], L = {};
      for (const id in m.layers) {
        const l = m.layers[id], q = b64bytes(l.q), w = new Float32Array(q.length), per = q.length / l.shape[0];
        for (let i = 0; i < q.length; i++) w[i] = q[i] * l.s[(i / per) | 0];
        const o = { w, b: l.b ? f32(l.b) : new Float32Array(l.shape[0]), shape: l.shape };
        if (l.bn) {
          const bn = { g: f32(l.bn.g), be: f32(l.bn.be), m: f32(l.bn.m), v: f32(l.bn.v), eps: l.bn.eps };
          bn.sc = bn.g.map((g, i) => g / Math.sqrt(bn.v[i] + bn.eps));            // y = sc * x + sh
          bn.sh = bn.be.map((b, i) => b - bn.m[i] * bn.sc[i]);
          o.bn = bn;
        }
        L[id] = o;
      }
      out[key] = { key, arch: m.arch, L, plan: plan(m.arch), name: m.arch.name, size: m.arch.input };
    }
    return out;
  }

  // ---- lapisan ----
  // konvolusi K x K, padding P, langkah 1: keluaran (H + 2P - K + 1)
  function conv3(inp, C, H, W, l, P) {              // jalur cepat 3 x 3: masukan diberi padding nol, 9 suku sekaligus
    const O = l.shape[0], Hp = H + 2 * P, Wp = W + 2 * P, Ho = Hp - 2, Wo = Wp - 2, out = new Float32Array(O * Ho * Wo), w = l.w;
    let p = inp;
    if (P) { p = new Float32Array(C * Hp * Wp); for (let c = 0; c < C; c++) for (let y = 0; y < H; y++) p.set(inp.subarray(c * H * W + y * W, c * H * W + (y + 1) * W), c * Hp * Wp + (y + P) * Wp + P); }
    for (let o = 0; o < O; o++) {
      const ob = o * Ho * Wo; out.fill(l.b[o], ob, ob + Ho * Wo);
      for (let c = 0; c < C; c++) {
        const wb = (o * C + c) * 9, w0 = w[wb], w1 = w[wb + 1], w2 = w[wb + 2], w3 = w[wb + 3], w4 = w[wb + 4], w5 = w[wb + 5], w6 = w[wb + 6], w7 = w[wb + 7], w8 = w[wb + 8];
        if (!(w0 || w1 || w2 || w3 || w4 || w5 || w6 || w7 || w8)) continue;
        const cb = c * Hp * Wp;
        for (let y = 0; y < Ho; y++) {
          const r0 = cb + y * Wp, r1 = r0 + Wp, r2 = r1 + Wp, orow = ob + y * Wo;
          for (let x = 0; x < Wo; x++)
            out[orow + x] += w0 * p[r0 + x] + w1 * p[r0 + x + 1] + w2 * p[r0 + x + 2] + w3 * p[r1 + x] + w4 * p[r1 + x + 1] + w5 * p[r1 + x + 2] + w6 * p[r2 + x] + w7 * p[r2 + x + 1] + w8 * p[r2 + x + 2];
        }
      }
    }
    return out;
  }
  function convS(inp, C, H, W, l, K, P, S) {       // langkah > 1 (AlexNet conv1: 11 x 11, langkah 4)
    const O = l.shape[0], Ho = ((H + 2 * P - K) / S | 0) + 1, Wo = ((W + 2 * P - K) / S | 0) + 1, KK = K * K, out = new Float32Array(O * Ho * Wo), w = l.w;
    for (let o = 0; o < O; o++) {
      const ob = o * Ho * Wo;
      for (let y = 0; y < Ho; y++) for (let x = 0; x < Wo; x++) {
        let sum = l.b[o];
        for (let c = 0; c < C; c++) {
          const ib = c * H * W, wb = (o * C + c) * KK;
          for (let ky = 0; ky < K; ky++) {
            const yy = y * S - P + ky; if (yy < 0 || yy >= H) continue;
            for (let kx = 0; kx < K; kx++) { const xx = x * S - P + kx; if (xx >= 0 && xx < W) sum += w[wb + ky * K + kx] * inp[ib + yy * W + xx]; }
          }
        }
        out[ob + y * Wo + x] = sum;
      }
    }
    return out;
  }
  function conv(inp, C, H, W, l, K, P, S) {
    if (S > 1) return convS(inp, C, H, W, l, K, P, S);
    if (K === 3) return conv3(inp, C, H, W, l, P);
    const O = l.shape[0], Ho = H + 2 * P - K + 1, Wo = W + 2 * P - K + 1, KK = K * K, out = new Float32Array(O * Ho * Wo), w = l.w;
    for (let o = 0; o < O; o++) {
      const ob = o * Ho * Wo;
      out.fill(l.b[o], ob, ob + Ho * Wo);
      for (let c = 0; c < C; c++) {
        const ib = c * H * W, wb = (o * C + c) * KK;
        for (let ky = 0; ky < K; ky++) for (let kx = 0; kx < K; kx++) {
          const wv = w[wb + ky * K + kx]; if (wv === 0) continue;
          const dy = ky - P, dx = kx - P, y0 = Math.max(0, -dy), y1 = Math.min(Ho, H - dy), x0 = Math.max(0, -dx), x1 = Math.min(Wo, W - dx);
          for (let y = y0; y < y1; y++) {
            const orow = ob + y * Wo, irow = ib + (y + dy) * W + dx;
            for (let x = x0; x < x1; x++) out[orow + x] += wv * inp[irow + x];
          }
        }
      }
    }
    return out;
  }
  function batchnorm(a, bn, C, S) {
    for (let c = 0; c < C; c++) { const s = bn.sc[c], h = bn.sh[c], b = c * S; for (let i = 0; i < S; i++) a[b + i] = a[b + i] * s + h; }
    return a;
  }
  // Local Response Normalization antarkanal (AlexNet): a / (bias + alpha * jumlah kuadrat 2r+1 kanal sekitar)^beta
  function lrn(a, C, S, p) {
    const o = new Float32Array(a.length), r = p.radius;
    for (let i = 0; i < S; i++) for (let c = 0; c < C; c++) {
      let sq = 0; for (let d = Math.max(0, c - r); d <= Math.min(C - 1, c + r); d++) { const v = a[d * S + i]; sq += v * v; }
      o[c * S + i] = a[c * S + i] / Math.pow(p.bias + p.alpha * sq, p.beta);
    }
    return o;
  }
  function relu(a) { for (let i = 0; i < a.length; i++) if (a[i] < 0) a[i] = 0; return a; }
  function pool(a, C, H, W, k, s, max) {
    const h = ((H - k) / s | 0) + 1, w = ((W - k) / s | 0) + 1, o = new Float32Array(C * h * w), inv = 1 / (k * k);
    for (let c = 0; c < C; c++) for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      let m = max ? -Infinity : 0; const b = c * H * W + s * y * W + s * x;
      for (let dy = 0; dy < k; dy++) for (let dx = 0; dx < k; dx++) { const v = a[b + dy * W + dx]; if (max) { if (v > m) m = v; } else m += v; }
      o[c * h * w + y * w + x] = max ? m : m * inv;
    }
    return o;
  }
  function gap(a, C, S) { const o = new Float32Array(C); for (let c = 0; c < C; c++) { let s = 0; for (let i = 0; i < S; i++) s += a[c * S + i]; o[c] = s / S; } return o; }
  function dense(v, l) {
    const [O, I] = l.shape, o = new Float32Array(O);
    for (let j = 0; j < O; j++) { let s = l.b[j]; const b = j * I; for (let i = 0; i < I; i++) s += l.w[b + i] * v[i]; o[j] = s; }
    return o;
  }
  function softmax(z) { const m = Math.max(...z); const e = Array.from(z, v => Math.exp(v - m)); const s = e.reduce((a, b) => a + b, 0); return e.map(v => v / s); }

  // x: Float32Array size*size (0..1). acts[id] = keluaran lapisan setelah BN dan ReLU (untuk dense softmax: probabilitas)
  function forward(M, x) {
    let cur = x; const acts = {}; let logits = null;
    for (const l of M.plan) {
      const L = M.L[l.id];
      if (l.type === 'conv') {
        const [C, H, W] = l.in, [O, Ho, Wo] = l.out;
        cur = conv(cur, C, H, W, L, l.k, l.pad, l.s || 1);
        if (L.bn) batchnorm(cur, L.bn, O, Ho * Wo);
        relu(cur);
        if (l.lrn) cur = lrn(cur, O, Ho * Wo, M.arch.lrn);
      } else if (l.type === 'maxpool' || l.type === 'avgpool') {
        const [C, H, W] = l.in; cur = pool(cur, C, H, W, l.k, l.s, l.type === 'maxpool');
      } else if (l.type === 'gap') {
        const [C, H, W] = l.in; cur = gap(cur, C, H * W);
      } else if (l.type === 'dense') {
        cur = dense(cur, L);
        if (l.act === 'relu') relu(cur);
        else if (l.act === 'softmax') { logits = cur; cur = Float32Array.from(softmax(cur)); }
      }
      acts[l.id] = cur;
    }
    const probs = Array.from(cur); let best = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
    return { x, acts, logits, probs, best };
  }

  // ---- praproses citra, mengikuti training: convert('L') lalu resize LANCZOS ke size x size ----
  // abu-abu PIL: (19595 R + 38470 G + 7471 B + 32768) >> 16
  function toGray(rgba, n) {
    const g = new Uint8Array(n);
    for (let i = 0; i < n; i++) g[i] = (19595 * rgba[i * 4] + 38470 * rgba[i * 4 + 1] + 7471 * rgba[i * 4 + 2] + 32768) >> 16;
    return g;
  }
  const sinc = x => x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x);
  const lanczos = x => Math.abs(x) < 3 ? sinc(x) * sinc(x / 3) : 0;
  // koefisien resample sama dengan Pillow (dukungan filter diperlebar saat memperkecil)
  function coeffs(inS, outS) {
    const scale = inS / outS, fs = Math.max(scale, 1), support = 3 * fs, list = [];
    for (let o = 0; o < outS; o++) {
      const center = (o + 0.5) * scale;
      const xmin = Math.max(Math.trunc(center - support + 0.5), 0), xmax = Math.min(Math.trunc(center + support + 0.5), inS);
      const ws = []; let tot = 0;
      for (let x = xmin; x < xmax; x++) { const v = lanczos((x - center + 0.5) / fs); ws.push(v); tot += v; }
      list.push({ xmin, ws: ws.map(v => tot ? v / tot : 0) });
    }
    return list;
  }
  const clamp8 = v => v < 0 ? 0 : v > 255 ? 255 : Math.round(v);   // Pillow membulatkan tiap lintasan ke 8 bit
  function resize(src, w, h, nw, nh) {
    const cx = coeffs(w, nw), cy = coeffs(h, nh), tmp = new Float32Array(nw * h), out = new Float32Array(nw * nh);
    for (let y = 0; y < h; y++) for (let x = 0; x < nw; x++) { const c = cx[x]; let s = 0; for (let k = 0; k < c.ws.length; k++) s += src[y * w + c.xmin + k] * c.ws[k]; tmp[y * nw + x] = clamp8(s); }
    for (let x = 0; x < nw; x++) for (let y = 0; y < nh; y++) { const c = cy[y]; let s = 0; for (let k = 0; k < c.ws.length; k++) s += tmp[(c.xmin + k) * nw + x] * c.ws[k]; out[y * nw + x] = clamp8(s); }
    return out;
  }
  // gray: Uint8 W x H. Dipotong persegi di tengah (citra NEU sudah persegi), lalu diperkecil ke size x size.
  function prepare(gray, W, H, size) {
    const s = Math.min(W, H), ox = (W - s) >> 1, oy = (H - s) >> 1, sq = new Float32Array(s * s);
    for (let y = 0; y < s; y++) for (let x = 0; x < s; x++) sq[y * s + x] = gray[(y + oy) * W + x + ox];
    const r = s === size ? sq : resize(sq, s, s, size, size), out = new Float32Array(size * size);
    for (let i = 0; i < out.length; i++) out[i] = r[i] / 255;
    return { x: out, crop: { x0: ox, y0: oy, s } };
  }

  return { load, plan, forward, prepare, toGray, resize, softmax };
})();
if (typeof module !== 'undefined') module.exports = NEU;
