// ---------- tema ----------
// adegan 3D memakai palet gelap tetap supaya biru dan jingga kontras
function sceneTheme(){
  return { bg:'#0B0E14', grid:'#1A2230', empty:'#05070B', ramp:RAMP, kernel:'#FF8A3D', out:'#38C6F0', frame:'#2A3646', ghost:'#3A6FA8', text:'#E6EDF5', muted:'#7F93AB' };
}
function setIcons(){ const d=isDark(); $('ic-sun').toggleAttribute('hidden', d); $('ic-moon').toggleAttribute('hidden', !d); }
function retheme(){
  readTheme(); setIcons();
  if (S3) S3.setTheme(sceneTheme());
  if (state.res) redraw(true);
}
$('theme').onclick = () => {
  const n = isDark() ? 'light' : 'dark';
  document.documentElement.dataset.theme = n;
  try { localStorage.setItem('lensa-baja-theme', n); } catch (e) {}
  retheme();
};

// ---------- rel bagian ----------
(() => {
  const rail = $('rail'), secs = [...document.querySelectorAll('[data-rail]')];
  rail.innerHTML = secs.map((s, i) => `<a href="#${s.id}"><span>${String(i + 1).padStart(2, '0')} · ${s.dataset.rail}</span><i></i></a>`).join('');
  const links = [...rail.children];
  const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) links.forEach((l, i) => l.classList.toggle('on', secs[i] === e.target)); }), { rootMargin: '-45% 0px -50% 0px' });
  secs.forEach(s => io.observe(s));
})();

setIcons();
if (S3) S3.setTheme(sceneTheme());
