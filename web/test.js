// Uji kecocokan: inferensi JS (praproses + jaringan) terhadap logit acuan dari NumPy.
//   node web/test.js
const NEU = require('./engine.js'), fs = require('fs'), path = require('path'), zlib = require('zlib');
const D = path.join(__dirname, 'data');

// pembaca PNG 8-bit (skala abu-abu atau RGB), cukup untuk contoh citra NEU
function readPNG(buf) {
  let p = 8, w = 0, h = 0, ct = 0; const idat = [];
  while (p < buf.length) {
    const len = buf.readUInt32BE(p), type = buf.toString('latin1', p + 4, p + 8), d = buf.subarray(p + 8, p + 8 + len);
    if (type === 'IHDR') { w = d.readUInt32BE(0); h = d.readUInt32BE(4); ct = d[9]; if (d[8] !== 8 || d[12]) throw new Error('PNG tidak didukung'); }
    if (type === 'IDAT') idat.push(d);
    p += 12 + len;
  }
  const bpp = { 0: 1, 2: 3, 6: 4 }[ct], raw = zlib.inflateSync(Buffer.concat(idat)), st = w * bpp, out = Buffer.alloc(h * st);
  for (let y = 0; y < h; y++) {
    const f = raw[y * (st + 1)], line = raw.subarray(y * (st + 1) + 1, (y + 1) * (st + 1));
    for (let i = 0; i < st; i++) {
      const a = i >= bpp ? out[y * st + i - bpp] : 0, b = y ? out[(y - 1) * st + i] : 0, c = i >= bpp && y ? out[(y - 1) * st + i - bpp] : 0;
      let v = line[i];
      if (f === 1) v += a; else if (f === 2) v += b; else if (f === 3) v += (a + b) >> 1;
      else if (f === 4) { const pp = a + b - c, pa = Math.abs(pp - a), pb = Math.abs(pp - b), pc = Math.abs(pp - c); v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c; }
      out[y * st + i] = v & 255;
    }
  }
  const rgba = new Uint8Array(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    if (bpp === 1) rgba.set([out[i], out[i], out[i], 255], i * 4); else rgba.set([out[i * bpp], out[i * bpp + 1], out[i * bpp + 2], 255], i * 4);
  }
  return { w, h, rgba };
}

const models = NEU.load(JSON.parse(fs.readFileSync(path.join(D, 'weights.json'))));
const samples = JSON.parse(fs.readFileSync(path.join(D, 'samples.json')));
let fail = 0;
for (const key in models) {
  const M = models[key];
  let md = 0, ok = 0, t = 0;
  for (const s of samples) {
    const im = readPNG(Buffer.from(s.png, 'base64')), g = NEU.toGray(im.rgba, im.w * im.h);
    const t0 = Date.now(), { x } = NEU.prepare(g, im.w, im.h, M.size), r = NEU.forward(M, x); t += Date.now() - t0;
    r.logits.forEach((v, k) => { md = Math.max(md, Math.abs(v - s.expect[key][k])); });
    if (r.best === s.y) ok++;
  }
  const pass = md < 0.05;
  if (!pass) fail++;
  console.log(`${pass ? 'OK  ' : 'GAGAL'} ${M.name.padEnd(9)} selisih logit maks vs NumPy ${md.toExponential(2)} | benar ${ok}/${samples.length} | ${(t / samples.length).toFixed(0)} ms/citra`);
}
process.exit(fail ? 1 : 0);
