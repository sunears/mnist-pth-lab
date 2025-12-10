// vis.js: render grayscale base64 images into pseudo-color heatmap canvases
function b64ToImageData(b64, callback) {
  const img = new Image();
  img.onload = () => {
    const c = document.createElement('canvas');
    c.width = img.width; c.height = img.height;
    const ctx = c.getContext('2d');
    ctx.drawImage(img,0,0);
    const id = ctx.getImageData(0,0,c.width,c.height);
    callback(id);
  };
  img.src = 'data:image/png;base64,' + b64;
}

function applyJetColormap(imageData) {
  const w = imageData.width, h = imageData.height;
  const inData = imageData.data;
  const out = new Uint8ClampedArray(w*h*4);
  for(let i=0;i<w*h;i++){
    const r = inData[i*4]; // grayscale in R channel
    const t = r/255.0;
    // simple jet colormap
    const rr = Math.min(255, Math.max(0, Math.floor(255 * (1.5 - Math.abs(4*t-3)))));
    const gg = Math.min(255, Math.max(0, Math.floor(255 * (1.5 - Math.abs(4*t-2)))));
    const bb = Math.min(255, Math.max(0, Math.floor(255 * (1.5 - Math.abs(4*t-1)))));
    out[i*4+0]=rr; out[i*4+1]=gg; out[i*4+2]=bb; out[i*4+3]=255;
  }
  return new ImageData(out, w, h);
}

function renderHeatmapFromB64(container, b64, meta) {
  b64ToImageData(b64, (id)=>{
    // normalize brightness
    const arr = id.data; let min=255,max=0;
    for(let i=0;i<arr.length;i+=4){ const v=arr[i]; if(v<min)min=v; if(v>max)max=v; }
    if(max>min){
      for(let i=0;i<arr.length;i+=4){ arr[i]=Math.floor((arr[i]-min)/(max-min)*255); arr[i+1]=arr[i]; arr[i+2]=arr[i]; }
    }
    const cmap = applyJetColormap(id);
    const canvas = document.createElement('canvas'); canvas.width = cmap.width; canvas.height = cmap.height; canvas.className='heatmap-canvas';
    canvas.getContext('2d').putImageData(cmap,0,0);
    const col = document.createElement('div'); col.className='vis-col';
    const label = document.createElement('div'); label.className='meta'; label.innerText = meta||'';
    col.appendChild(canvas); col.appendChild(label); container.appendChild(col);
  });
}

function renderGroup(containerId, items, key){
  const container = document.getElementById(containerId);
  if(!container) return;
  const group = document.createElement('div'); group.className='vis-group';
  const row = document.createElement('div'); row.className='vis-row';
  items.forEach(it=>{
    const b64 = it[key]||it.image||it.kernel;
    const meta = `ch:${it.channel} ${it.score?('score:'+it.score.toFixed(3)) : ''}`;
    renderHeatmapFromB64(row, b64, meta);
  });
  group.appendChild(row); container.appendChild(group);
}

// export for index usage
window.renderGroup = renderGroup;
