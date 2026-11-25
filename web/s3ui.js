// ---------- kontrol tampilan 3D ----------
const PLAY_D='M4 2.8v10.4c0 .6.6.9 1.1.6l8.2-5.2c.5-.3.5-.9 0-1.2L5.1 2.2C4.6 1.9 4 2.2 4 2.8z', PAUSE_D='M4 2.5h2.6v11H4zM9.4 2.5H12v11H9.4z';
function s3btn(){ const p=S3&&S3.playing; $('s3icon').setAttribute('d', p?PAUSE_D:PLAY_D); $('s3playtxt').textContent=p?'Jeda':'Putar'; }
function divCol(v, m){ return diverge(v/(m||1)); }
function actCol(v, m){ return magma(Math.max(0,v)/(m||1)); }
const lum=c=>(0.299*c[0]+0.587*c[1]+0.114*c[2])/255;
const mat=(vals, n, col, fmt)=>`<div class="mat${n>5?' sm':''}" style="grid-template-columns:repeat(${n},auto)">${vals.map(v=>{ const c=col(v); return `<i style="background:rgb(${c.map(z=>z|0)});color:${lum(c)>.55?'#111':'#f4f4f4'}">${fmt(v)}</i>`; }).join('')}</div>`;
const f2=v=>v.toFixed(2);
const at2=(a,H,W,r,c,off=0)=> (r<0||c<0||r>=H||c>=W)?0:a[off+r*W+c];

// sumber (keluaran lapisan sebelumnya) dari lapisan l
function srcOf(l){
  const M=state.M, i=M.plan.findIndex(x=>x.id===l.id);
  if(i===0) return { id:'inp', data:state.res.x, shape:[1,M.size,M.size], short:'masukan' };
  const p=M.plan[i-1]; return { id:p.id, data:state.res.acts[p.id], shape:p.out, short:p.short, l:p };
}
function calc(info){
  const el=$('s3calc');
  $('s3seek').value=info.t; s3btn();
  const res=state.res;
  if(!res){ el.textContent=''; return; }
  const M=state.M, L=M.L, win=CN[res.best];
  if(!info.phase){
    $('s3hud').innerHTML=`Selesai · hasil <b>${win}</b> ${(res.probs[res.best]*100).toFixed(1)}%`;
    el.innerHTML=`<span>Tekan <b class="res">Putar</b> atau geser penggeser</span>`; return;
  }
  const pos = info.r!=null ? ` · baris ${info.r+1}, kolom ${info.c+1}` : ` · unit ${info.j+1}/${info.l.size}`;
  $('s3hud').innerHTML=`<b>${info.phase}</b>${pos} · langkah ${info.t+1}/${S3.TOTAL}`;
  const l=info.l, L1=L[l.id], src=srcOf(l), a=res.acts[l.id];
  if(info.kind==='conv'){
    const [Cin,H,W]=l.in, K=l.k, P=l.pad, ST=l.s||1, KK=K*K, S=H*W, SO=l.out[1]*l.out[2];
    const kin=src.id==='inp'?0:S3.chOf(src.id), kout=S3.chOf(l.id), off=kin*S;
    const patch=[]; for(let dy=0;dy<K;dy++) for(let dx=0;dx<K;dx++) patch.push(at2(src.data,H,W,info.r*ST-P+dy,info.c*ST-P+dx,off));
    const w=Array.from(L1.w.subarray((kout*Cin+kin)*KK,(kout*Cin+kin+1)*KK));
    const wm=Math.max(...w.map(Math.abs)), pm=Math.max(1e-6,...patch);
    const part=patch.reduce((s,v,i)=>s+v*w[i],0);
    let pre=L1.b[kout];
    for(let ci=0;ci<Cin;ci++) for(let dy=0;dy<K;dy++) for(let dx=0;dx<K;dx++) pre+=at2(src.data,H,W,info.r*ST-P+dy,info.c*ST-P+dx,ci*S)*L1.w[((kout*Cin+ci)*K+dy)*K+dx];
    const bn=L1.bn?pre*L1.bn.sc[kout]+L1.bn.sh[kout]:null, rl=Math.max(0,bn!==null?bn:pre);
    const srcName=src.id==='inp'?'Patch masukan':`Patch ${src.short}, kanal ${kin+1}`;
    el.innerHTML=`<div><span class="tag">${srcName}</span>${mat(patch,K,v=>actCol(v,pm),f2)}</div><span class="op">×</span>
      <div><span class="tag">Kernel ${K} × ${K} · ${kout+1}${Cin>1?'·'+(kin+1):''}</span>${mat(w,K,v=>divCol(v,wm),f2)}</div><span class="op">=</span>
      <div><span class="tag">${Cin>1?`Σ ${Cin} kanal + bias`:'Σ + bias'}</span><span class="res">${pre.toFixed(3)}</span>${Cin>1?`<br><span>dari kanal ${kin+1} saja ${part.toFixed(3)}</span>`:''}</div>
      ${bn!==null?`<span class="op">→</span><div><span class="tag">BatchNorm</span><span class="res">${bn.toFixed(3)}</span></div>`:''}
      <span class="op">→</span><div><span class="tag">ReLU</span><span class="res">${rl.toFixed(3)}</span></div>
      ${l.lrn?`<span class="op">→</span><div><span class="tag">LRN</span><span class="res">${a[kout*SO+info.j].toFixed(3)}</span></div>`:''}
      ${ST>1?`<span>Kernel digeser dengan langkah ${ST} piksel.</span>`:''}`;
  } else if(info.kind==='pool'){
    const [C,H,W]=l.in, k=l.k, kc=S3.chOf(l.id), off=kc*H*W, mx=l.type==='maxpool';
    const v=[]; for(let dy=0;dy<k;dy++) for(let dx=0;dx<k;dx++) v.push(at2(src.data,H,W,l.s*info.r+dy,l.s*info.c+dx,off));
    const m=mx?Math.max(...v):v.reduce((s,x)=>s+x,0)/v.length, top=Math.max(...v,1e-6);
    el.innerHTML=`<div><span class="tag">Blok ${k} × ${k}</span>${mat(v,k,x=>actCol(x,top),f2)}</div><span class="op">→</span><div><span class="tag">${mx?'Maks':'Rata-rata'}</span><span class="res">${m.toFixed(3)}</span></div><span>Jendela ${k} × ${k} digeser dengan langkah ${l.s}${k>l.s?`, jadi jendela yang bersebelahan saling tumpang tindih`:' sehingga tidak tumpang tindih'}. ${mx?'Hanya nilai terbesar yang diteruskan.':`${k*k} nilai dirata-ratakan.`}</span>`;
  } else if(l.type==='gap'){
    const [C,H,W]=l.in;
    el.innerHTML=`<div><span class="tag">Kanal ${info.j+1} dari ${C}</span><span>Σ ${H} × ${W} nilai / ${H*W}</span></div><span class="op">=</span><div><span class="tag">Rata-rata</span><span class="res">${a[info.j].toFixed(3)}</span></div><span>Setiap peta fitur diringkas menjadi satu angka.</span>`;
  } else if(l.type==='dense' && l.act==='softmax'){
    const z=res.logits[info.j];
    el.innerHTML=`<div><span class="tag">Kelas ${CN[info.j]}</span><span>skor z = ${z.toFixed(2)}</span></div><span class="op">→</span><div><span class="tag">e<sup>z</sup> / Σ e<sup>z</sup></span><span class="res">${(res.probs[info.j]*100).toFixed(2)}%</span></div>`;
  } else {
    // dense, atau konvolusi yang menutup seluruh peta (C5 pada LeNet-5), sama dengan dense atas nilai yang diratakan
    const I=L1.shape.slice(1).reduce((x,y)=>x*y,1), v=src.data;
    let pre=L1.b[info.j]; for(let i=0;i<I;i++) pre+=L1.w[info.j*I+i]*v[i];
    el.innerHTML=`<div><span class="tag">Neuron ${info.j+1} dari ${l.size}</span><span>Σ ${I} nilai × bobot + bias</span></div><span class="op">=</span><div><span class="tag">Hasil</span><span class="res">${pre.toFixed(3)}</span></div><span class="op">→</span><div><span class="tag">ReLU</span><span class="res">${a[info.j].toFixed(3)}</span></div>`;
  }
}
// pemilih kanal dibuat sesuai jumlah lapisan konvolusi pada model
function buildChannelRows(){
  const box=$('s3ch'); box.textContent='';
  S3.convIds.forEach(id=>{
    const l=state.M.plan.find(x=>x.id===id), row=document.createElement('div'); row.className='chsel';
    row.innerHTML=`<span>kanal ${l.short.toLowerCase()}</span><button class="pill" aria-label="Kanal sebelumnya">‹</button><b>1</b><button class="pill" aria-label="Kanal berikutnya">›</button>`;
    const [bp,bn]=row.querySelectorAll('button'), lab=row.querySelector('b'); lab.dataset.id=id;
    const go=d=>()=>{ const k=S3.channel(id,d); lab.textContent=k+1; syncChannelLabels(); calc({t:S3.t,total:S3.TOTAL,phase:null}); S3.seek(S3.t); };
    bp.onclick=go(-1); bn.onclick=go(1); box.append(row);
  });
}
function syncChannelLabels(){ $('s3ch').querySelectorAll('b').forEach(b=>b.textContent=S3.chOf(b.dataset.id)+1); }
if(S3){
  S3.onState(calc);
  $('s3play').onclick=()=>{ S3.toggle(); s3btn(); };
  $('s3restart').onclick=()=>{ S3.restart(); s3btn(); };
  $('s3seek').addEventListener('input', e=>{ S3.seek(+e.target.value); s3btn(); });
  [1,4,14].forEach(v=>$('sp'+v).onclick=()=>{ S3.setSpeed(v); [1,4,14].forEach(u=>$('sp'+u).setAttribute('aria-pressed',u===v)); });
} else {
  $('s3').querySelector('.s3-bar').hidden=true; $('s3').querySelector('.s3-foot').hidden=true;
}
