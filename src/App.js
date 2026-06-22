import React, { useState, useRef, useCallback } from "react";

// ═══════════════════════════════════════════════════════════════════════════════
// USERS
// ═══════════════════════════════════════════════════════════════════════════════
const INITIAL_USERS = {
  admin:    { password: "admin123", role: "Administrateur", name: "Administrateur SegVision",       validated: true },
  Administrateur:  { password: "abcd",     role: "Administrateur", name: "Responsable Sécurité",      validated: true },
  Utilisateur1: { password: "pass1",    role: "client",  name: "Jean Martin",     validated: true },
  Utilisateur2: { password: "pass2",    role: "client",  name: "Sophie Durand",   validated: true },
};

// ═══════════════════════════════════════════════════════════════════════════════
// PALETTE
// ═══════════════════════════════════════════════════════════════════════════════
const C = {
  bg: "#F0F4FF", surface: "#FFFFFF", border: "#DDE3F0",
  primary: "#2D5BE3", primaryDark: "#1A3DBF", primaryLight: "#EEF3FF",
  accent: "#7C3AED", accentLight: "#F3EEFF",
  success: "#059669", successBg: "#ECFDF5",
  warning: "#D97706", warningBg: "#FFFBEB",
  red: "#DC2626", redBg: "#FEF2F2",
  text: "#0F172A", sub: "#4B5563", muted: "#9CA3AF",
  sidebar: "#0C1A3A", sidebarText: "#A0B0CC",
  clientSidebar: "#1A0A3A", clientActive: "#7C3AED",
};

// ═══════════════════════════════════════════════════════════════════════════════
// BIOMETRIC PROCESSING ENGINE — PIPELINE COMPLET
// ═══════════════════════════════════════════════════════════════════════════════

// ── Squelettisation (Zhang-Suen thinning) ────────────────────────────────────
function skeletonize(mask, W, H) {
  const skel = new Uint8Array(mask);
  let changed = true;
  while (changed) {
    changed = false;
    const toRemove = new Uint8Array(W * H);
    for (let pass = 0; pass < 2; pass++) {
      for (let y = 1; y < H - 1; y++) {
        for (let x = 1; x < W - 1; x++) {
          const i = y * W + x;
          if (!skel[i]) continue;
          const p2=skel[(y-1)*W+x], p3=skel[(y-1)*W+(x+1)], p4=skel[y*W+(x+1)];
          const p5=skel[(y+1)*W+(x+1)], p6=skel[(y+1)*W+x], p7=skel[(y+1)*W+(x-1)];
          const p8=skel[y*W+(x-1)], p9=skel[(y-1)*W+(x-1)];
          const nb = p2+p3+p4+p5+p6+p7+p8+p9;
          if (nb < 2 || nb > 6) continue;
          const seq = [p2,p3,p4,p5,p6,p7,p8,p9,p2];
          let transitions = 0;
          for (let k = 0; k < 8; k++) if (seq[k] === 0 && seq[k+1] === 1) transitions++;
          if (transitions !== 1) continue;
          if (pass === 0 && (p2*p4*p6 !== 0 || p4*p6*p8 !== 0)) continue;
          if (pass === 1 && (p2*p4*p8 !== 0 || p2*p6*p8 !== 0)) continue;
          toRemove[i] = 1; changed = true;
        }
      }
      for (let i = 0; i < W * H; i++) if (toRemove[i]) { skel[i] = 0; toRemove[i] = 0; }
    }
  }
  return skel;
}

// ── Extraction features RÉTINE ────────────────────────────────────────────────
function extractRetineFeatures(mask, skel, W, H) {
  // Labellisation des segments du squelette par BFS
  const labels = new Int32Array(W * H).fill(-1);
  const segments = [];
  let labelCount = 0;

  // Détecter les minuties sur le squelette (bifurcations + terminaisons)
  const bifs = [], terms = [];
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const i = y * W + x;
      if (!skel[i]) continue;
      const nb = skel[(y-1)*W+(x-1)]+skel[(y-1)*W+x]+skel[(y-1)*W+(x+1)]
               + skel[y*W+(x-1)]+skel[y*W+(x+1)]
               + skel[(y+1)*W+(x-1)]+skel[(y+1)*W+x]+skel[(y+1)*W+(x+1)];
      if (nb === 1) terms.push({y, x});
      else if (nb >= 3) bifs.push({y, x});
    }
  }

  // Labellisation des segments vasculaires par BFS
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      if (!skel[i] || labels[i] >= 0) continue;
      const seg = []; const queue = [{y, x}]; labels[i] = labelCount;
      while (queue.length) {
        const {y: cy, x: cx} = queue.shift();
        seg.push({y: cy, x: cx});
        for (const [dy, dx] of [[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1],[1,-1],[1,1]]) {
          const ny = cy+dy, nx = cx+dx;
          if (ny<0||ny>=H||nx<0||nx>=W) continue;
          const ni = ny*W+nx;
          if (skel[ni] && labels[ni] < 0) { labels[ni] = labelCount; queue.push({y:ny,x:nx}); }
        }
      }
      if (seg.length >= 5) segments.push(seg);
      labelCount++;
    }
  }

  // ── Features par segment ──────────────────────────────────────────────────
  const segLengths = [], tortuosities = [], diameters = [];

  for (const seg of segments) {
    // Longueur du segment (nombre de pixels squelette)
    const len = seg.length;
    // Distance euclidienne entre les deux extrémités
    const start = seg[0], end = seg[seg.length-1];
    const eucDist = Math.sqrt((end.x-start.x)**2 + (end.y-start.y)**2) || 1;
    // Tortuosité = longueur / distance euclidienne
    const tort = len / eucDist;
    segLengths.push(len);
    tortuosities.push(tort);

    // Diamètre : distance transform approximée (épaisseur du vaisseau)
    // On mesure la largeur locale au centre du segment
    const mid = seg[Math.floor(seg.length/2)];
    let diam = 1;
    for (let r = 1; r <= 10; r++) {
      let allIn = true;
      for (const [dy, dx] of [[-r,0],[r,0],[0,-r],[0,r]]) {
        const ny=mid.y+dy, nx=mid.x+dx;
        if (ny<0||ny>=H||nx<0||nx>=W||!mask[ny*W+nx]) { allIn=false; break; }
      }
      if (allIn) diam = r*2+1; else break;
    }
    diameters.push(diam);
  }

  // ── Calcul des métriques globales ─────────────────────────────────────────
  const totalFg = Array.from(mask).filter(v=>v>0).length;
  const density = totalFg / (W * H);

  // OvLen : longueur vasculaire totale (somme des longueurs de segments)
  const OvLen = segLengths.reduce((a,b)=>a+b,0);

  // TI : tortuosity index = moyenne des tortuosités
  const TI = tortuosities.length > 0 ? tortuosities.reduce((a,b)=>a+b,0)/tortuosities.length : 1;

  // MedTor : tortuosité médiane
  const sortedTort = [...tortuosities].sort((a,b)=>a-b);
  const MedTor = sortedTort.length > 0 ? sortedTort[Math.floor(sortedTort.length/2)] : 1;

  // D1 : diamètre moyen
  const D1 = diameters.length > 0 ? diameters.reduce((a,b)=>a+b,0)/diameters.length : 1;

  // D2 : écart-type des diamètres
  const meanD = D1;
  const D2 = diameters.length > 1
    ? Math.sqrt(diameters.reduce((a,b)=>a+(b-meanD)**2,0)/diameters.length)
    : 0;

  // ── Vecteur complet ───────────────────────────────────────────────────────
  // Directions (8 bins)
  const dirHist = new Array(8).fill(0);
  for (const seg of segments) {
    if (seg.length < 2) continue;
    for (let k = 0; k < seg.length-1; k++) {
      const dx = seg[k+1].x-seg[k].x, dy = seg[k+1].y-seg[k].y;
      const angle = Math.atan2(dy, dx);
      const idx = Math.floor(((angle+Math.PI)/(2*Math.PI))*8)%8;
      dirHist[idx]++;
    }
  }
  const totalDir = dirHist.reduce((a,b)=>a+b,0)||1;
  const normDir = dirHist.map(v=>parseFloat((v/totalDir).toFixed(4)));

  const fullVector = {
    // Morphologie vasculaire
    OvLen:  parseFloat(OvLen.toFixed(2)),
    TI:     parseFloat(TI.toFixed(4)),
    MedTor: parseFloat(MedTor.toFixed(4)),
    D1:     parseFloat(D1.toFixed(4)),
    D2:     parseFloat(D2.toFixed(4)),
    // Réseau
    nbSegments:    segments.length,
    nbBifurcations: bifs.length,
    nbTerminations: terms.length,
    density:       parseFloat((density*100).toFixed(2)),
    // Directions
    dir0: normDir[0], dir1: normDir[1], dir2: normDir[2], dir3: normDir[3],
    dir4: normDir[4], dir5: normDir[5], dir6: normDir[6], dir7: normDir[7],
  };

  // ── Vecteur optimisé (5 features rétine) ─────────────────────────────────
  const optimizedVector = {
    OvLen:  fullVector.OvLen,
    TI:     fullVector.TI,
    MedTor: fullVector.MedTor,
    D1:     fullVector.D1,
    D2:     fullVector.D2,
  };

  return {
    fullVector,
    optimizedVector,
    optimizedArray: [fullVector.OvLen, fullVector.TI, fullVector.MedTor, fullVector.D1, fullVector.D2],
    stats: {
      density: fullVector.density,
      bifurcations: bifs.length,
      terminations: terms.length,
      totalFg, OvLen, TI, MedTor, D1, D2,
      segments: segments.length,
    }
  };
}

// ── Extraction features EMPREINTE ─────────────────────────────────────────────
function extractFingerprintFeatures(mask, W, H) {
  let bifurcations = 0, terminations = 0;
  const minutiaeList = [];
  const orientations = [];

  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      if (!mask[y*W+x]) continue;
      const nb = mask[(y-1)*W+(x-1)]+mask[(y-1)*W+x]+mask[(y-1)*W+(x+1)]
               + mask[y*W+(x-1)]+mask[y*W+(x+1)]
               + mask[(y+1)*W+(x-1)]+mask[(y+1)*W+x]+mask[(y+1)*W+(x+1)];
      if (nb === 1) { terminations++; minutiaeList.push({y,x,type:'T'}); }
      else if (nb >= 3) { bifurcations++; minutiaeList.push({y,x,type:'B'}); }

      // Orientation locale des crêtes
      const gx = mask[y*W+(x+1)] - mask[y*W+(x-1)];
      const gy = mask[(y+1)*W+x] - mask[(y-1)*W+x];
      if (gx !== 0 || gy !== 0) orientations.push(Math.atan2(gy, gx));
    }
  }

  const totalFg = Array.from(mask).filter(v=>v>0).length;
  const density = totalFg / (W * H);
  const nbMinutiae = bifurcations + terminations;

  // Densité des minuties (par rapport à la surface)
  const minutiaeDensity = nbMinutiae / (totalFg || 1) * 1000;

  // Orientation moyenne des crêtes
  let sumSin = 0, sumCos = 0;
  for (const a of orientations) { sumSin += Math.sin(2*a); sumCos += Math.cos(2*a); }
  const meanOrientation = orientations.length > 0
    ? Math.atan2(sumSin/orientations.length, sumCos/orientations.length) / 2
    : 0;

  // Variation des orientations (écart-type)
  let varSum = 0;
  for (const a of orientations) varSum += (a - meanOrientation)**2;
  const orientationVariation = orientations.length > 1
    ? Math.sqrt(varSum/orientations.length)
    : 0;

  const fullVector = {
    nbMinutiae,
    nbBifurcations:      bifurcations,
    nbTerminations:      terminations,
    minutiaeDensity:     parseFloat(minutiaeDensity.toFixed(4)),
    meanOrientation:     parseFloat(meanOrientation.toFixed(4)),
    orientationVariation:parseFloat(orientationVariation.toFixed(4)),
    density:             parseFloat((density*100).toFixed(2)),
    totalFg,
  };

  const optimizedVector = {
    nbMinutiae:           fullVector.nbMinutiae,
    nbBifurcations:       fullVector.nbBifurcations,
    nbTerminations:       fullVector.nbTerminations,
    minutiaeDensity:      fullVector.minutiaeDensity,
    meanOrientation:      fullVector.meanOrientation,
    orientationVariation: fullVector.orientationVariation,
  };

  return {
    fullVector,
    optimizedVector,
    optimizedArray: [
      fullVector.nbMinutiae,
      fullVector.nbBifurcations,
      fullVector.nbTerminations,
      fullVector.minutiaeDensity,
      fullVector.meanOrientation,
      fullVector.orientationVariation,
    ],
    stats: fullVector,
  };
}

// ── Distance entre vecteurs optimisés ─────────────────────────────────────────
function euclideanDistance(a, b) {
  if (!a || !b || a.length !== b.length) return Infinity;
  return Math.sqrt(a.reduce((sum, v, i) => sum + (v - b[i])**2, 0));
}

function cosineSimilarity(a, b) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i]*b[i]; normA += a[i]*a[i]; normB += b[i]*b[i];
  }
  return dot / (Math.sqrt(normA)*Math.sqrt(normB) || 1);
}

// Normaliser un vecteur pour la comparaison
function normalizeVector(v) {
  const norm = Math.sqrt(v.reduce((s,x)=>s+x*x,0))||1;
  return v.map(x=>x/norm);
}

// Deux vecteurs biométriques sont identiques (à 4 décimales près)
function vectorsMatch(a, b) {
  if (!a || !b || a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (Number(a[i]).toFixed(4) !== Number(b[i]).toFixed(4)) return false;
  }
  return true;
}

// ─── Segmentation biométrique ────────────────────────────────────────────────
async function processBiometric(file, mode) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const url = URL.createObjectURL(file);
    img.onload = () => {
      const W = 512, H = 512;

      const origCanvas = document.createElement("canvas");
      origCanvas.width = W; origCanvas.height = H;
      const oCtx = origCanvas.getContext("2d");
      oCtx.fillStyle = "#000"; oCtx.fillRect(0,0,W,H);
      oCtx.drawImage(img, 0, 0, W, H);
      URL.revokeObjectURL(url);
      const raw = oCtx.getImageData(0,0,W,H).data;

      function gauss(src, sigma) {
        const k=Math.ceil(sigma*3)*2+1, half=Math.floor(k/2);
        const ker=new Float32Array(k); let ks=0;
        for(let i=0;i<k;i++){ker[i]=Math.exp(-0.5*((i-half)/sigma)**2);ks+=ker[i];}
        for(let i=0;i<k;i++) ker[i]/=ks;
        const tmp=new Float32Array(W*H), out=new Float32Array(W*H);
        for(let y=0;y<H;y++) for(let x=0;x<W;x++){let s=0;for(let ki=0;ki<k;ki++){const xi=Math.min(Math.max(x+ki-half,0),W-1);s+=src[y*W+xi]*ker[ki];}tmp[y*W+x]=s;}
        for(let y=0;y<H;y++) for(let x=0;x<W;x++){let s=0;for(let ki=0;ki<k;ki++){const yi=Math.min(Math.max(y+ki-half,0),H-1);s+=tmp[yi*W+x]*ker[ki];}out[y*W+x]=s;}
        return out;
      }

      // Canal vert inversé : vaisseaux rouges sombres → hauts
      const signal = new Float32Array(W*H);
      for(let i=0;i<W*H;i++) {
        if(mode==="retine") signal[i] = 1.0 - raw[i*4+1]/255.0;
        else { const g=(raw[i*4]*0.299+raw[i*4+1]*0.587+raw[i*4+2]*0.114)/255.0; signal[i]=1.0-g; }
      }

      // DoG multi-échelle
      const vessel = new Float32Array(W*H);
      const sigmas = mode==="retine" ? [1,1.5,2,3,4] : [0.5,1,1.5,2];
      for(const s of sigmas) {
        const b1=gauss(signal,s), b2=gauss(signal,s*1.6);
        for(let i=0;i<W*H;i++){const r=Math.max(0,b1[i]-b2[i]*0.75);if(r>vessel[i])vessel[i]=r;}
      }

      const sorted=Float32Array.from(vessel).sort();
      const pct = mode==="retine" ? 0.70 : 0.65;
      const thresh=sorted[Math.floor(W*H*pct)];

      const mask=new Uint8Array(W*H);
      for(let i=0;i<W*H;i++){
        const isBlack=(raw[i*4]+raw[i*4+1]+raw[i*4+2])<15;
        mask[i]=(vessel[i]>=thresh&&!isBlack)?1:0;
      }

      const clean=new Uint8Array(W*H);
      for(let y=1;y<H-1;y++) for(let x=1;x<W-1;x++){
        if(!mask[y*W+x]) continue;
        const nb=mask[(y-1)*W+x]+mask[(y+1)*W+x]+mask[y*W+(x-1)]+mask[y*W+(x+1)]+mask[(y-1)*W+(x-1)]+mask[(y-1)*W+(x+1)]+mask[(y+1)*W+(x-1)]+mask[(y+1)*W+(x+1)];
        if(nb>=2) clean[y*W+x]=1;
      }

      const maskCanvas=document.createElement("canvas");
      maskCanvas.width=W; maskCanvas.height=H;
      const mCtx=maskCanvas.getContext("2d");
      const mData=mCtx.createImageData(W,H);
      for(let i=0;i<W*H;i++){const v=clean[i]?255:0;mData.data[i*4]=v;mData.data[i*4+1]=v;mData.data[i*4+2]=v;mData.data[i*4+3]=255;}
      mCtx.putImageData(mData,0,0);

      const ovCanvas=document.createElement("canvas");
      ovCanvas.width=W; ovCanvas.height=H;
      const ovCtx=ovCanvas.getContext("2d");
      ovCtx.drawImage(origCanvas,0,0);
      const ovData=ovCtx.getImageData(0,0,W,H);
      const od=ovData.data;
      const [hr,hg,hb]=mode==="retine"?[255,80,80]:[60,160,255];
      for(let i=0;i<W*H;i++) if(clean[i]){od[i*4]=Math.min(255,od[i*4]*0.2+hr*0.8);od[i*4+1]=Math.min(255,od[i*4+1]*0.2+hg*0.8);od[i*4+2]=Math.min(255,od[i*4+2]*0.2+hb*0.8);}
      ovCtx.putImageData(ovData,0,0);

      const cleanF=new Uint8Array(W*H);
      for(let i=0;i<W*H;i++) cleanF[i]=clean[i];

      // Squelettisation
      const skel = skeletonize(cleanF, W, H);

      // Rendu squelette
      const skelCanvas=document.createElement("canvas");
      skelCanvas.width=W; skelCanvas.height=H;
      const sCtx=skelCanvas.getContext("2d");
      const sData=sCtx.createImageData(W,H);
      for(let i=0;i<W*H;i++){const v=skel[i]?255:0;sData.data[i*4]=v;sData.data[i*4+1]=v;sData.data[i*4+2]=v;sData.data[i*4+3]=255;}
      sCtx.putImageData(sData,0,0);

      // Extraction features selon mode
      let features;
      if(mode==="retine") {
        features = extractRetineFeatures(cleanF, skel, W, H);
      } else {
        features = extractFingerprintFeatures(cleanF, W, H);
      }

      resolve({
        maskUrl:maskCanvas.toDataURL("image/png"),
        overlayUrl:ovCanvas.toDataURL("image/png"),
        originalUrl:origCanvas.toDataURL("image/png"),
        skelUrl:skelCanvas.toDataURL("image/png"),
        fullVector: features.fullVector,
        optimizedVector: features.optimizedVector,
        optimizedArray: features.optimizedArray,
        stats: features.stats,
        mode,
      });
    };
    img.onerror=reject;
    img.src=url;
  });
}
// ═══════════════════════════════════════════════════════════════════════════════
// STYLES
// ═══════════════════════════════════════════════════════════════════════════════
const F = { fontFamily: "'Inter','Segoe UI',sans-serif" };
const base = {
  input:  { width:"100%", padding:"10px 12px", border:`1px solid ${C.border}`, borderRadius:8, fontSize:14, outline:"none", boxSizing:"border-box", marginBottom:16, background:C.bg, color:C.text },
  label:  { display:"block", fontWeight:600, fontSize:12, color:C.sub, marginBottom:6, textTransform:"uppercase", letterSpacing:"0.05em" },
  card:   { background:C.surface, border:`1px solid ${C.border}`, borderRadius:14, padding:24 },
  th:     { textAlign:"left", padding:"11px 16px", fontSize:12, fontWeight:700, color:C.sub, textTransform:"uppercase", letterSpacing:"0.05em", background:C.bg, borderBottom:`1px solid ${C.border}` },
  td:     { padding:"13px 16px", fontSize:14, borderBottom:`1px solid ${C.border}`, color:C.text },
};
const mkBtn = (variant="primary", color=C.primary) => ({
  display:"inline-flex", alignItems:"center", justifyContent:"center", gap:6,
  padding:"10px 18px", borderRadius:9, fontWeight:600, fontSize:14, cursor:"pointer",
  border: variant==="ghost" ? `1px solid ${C.border}` : "none",
  background: variant==="primary" ? color : variant==="ghost" ? C.surface : color+"18",
  color: variant==="primary" ? "#fff" : color,
});
const mkBadge = (color=C.primary) => ({ background:color+"18", color, borderRadius:20, padding:"3px 10px", fontSize:12, fontWeight:600 });
const mkChip  = (color=C.success) => ({ background:color+"18", color, borderRadius:20, padding:"4px 12px", fontSize:12, fontWeight:600 });
const mkNav   = (active, ac=C.primary) => ({ display:"flex", alignItems:"center", gap:10, padding:"9px 12px", borderRadius:9, color:active?"#fff":C.sidebarText, background:active?ac:"transparent", cursor:"pointer", fontWeight:active?600:400, fontSize:14, marginBottom:2, border:"none", width:"100%", textAlign:"left" });

// ═══════════════════════════════════════════════════════════════════════════════
// ICONS
// ═══════════════════════════════════════════════════════════════════════════════
const Ic = {
  grid:     <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>,
  scan:     <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M3 7V5a2 2 0 012-2h2"/><path d="M17 3h2a2 2 0 012 2v2"/><path d="M21 17v2a2 2 0 01-2 2h-2"/><path d="M7 21H5a2 2 0 01-2-2v-2"/><line x1="8" y1="12" x2="16" y2="12"/></svg>,
  chart:    <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>,
  clock:    <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>,
  shield:   <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>,
  search:   <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>,
  check:    <svg width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2.5" viewBox="0 0 24 24"><polyline points="20 6 9 17 4 12"/></svg>,
  logout:   <svg width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M9 21H5a2 2 0 01-2-2V5a2 2 0 012-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/></svg>,
  download: <svg width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>,
  user:     <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>,
  steth:    <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M4.8 2.3A.3.3 0 105 2H4a2 2 0 00-2 2v5a6 6 0 006 6 6 6 0 006-6V4a2 2 0 00-2-2h-1a.2.2 0 10.3.3"/><path d="M8 15v1a6 6 0 006 6 6 6 0 006-6v-4"/><circle cx="20" cy="10" r="2"/></svg>,
  back:     <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><line x1="19" y1="12" x2="5" y2="12"/><polyline points="12 19 5 12 12 5"/></svg>,
  plus:     <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>,
  file:     <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>,
  eye:      <svg width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>,
  finger:   <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M12 2a4 4 0 014 4v6a4 4 0 01-8 0V6a4 4 0 014-4z"/><path d="M8 10a4 4 0 000 8h8"/></svg>,
  retine:   <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="4"/><line x1="12" y1="2" x2="12" y2="6"/><line x1="12" y1="18" x2="12" y2="22"/><line x1="2" y1="12" x2="6" y2="12"/><line x1="18" y1="12" x2="22" y2="12"/></svg>,
  db:       <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>,
};

// ═══════════════════════════════════════════════════════════════════════════════
// LOGO
// ═══════════════════════════════════════════════════════════════════════════════
function Logo({ size=32, dark=true }) {
  return (
    <div style={{ display:"flex", alignItems:"center", gap:10 }}>
      <img
        src="/logo-segvision.png"
        alt="SegVision"
        style={{
          width: size * 4,
          height: "auto",
          objectFit: "contain",
          display: "block"
        }}
      />
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// REGISTER
// ═══════════════════════════════════════════════════════════════════════════════
function RegisterPage({ role, onBack, onRegister }) {
  const [step, setStep]   = useState(1);
  const [form, setForm]   = useState({ username:"", password:"", confirm:"", fullName:"", email:"" });
  const [docs, setDocs]   = useState({ rpps:"", ordreNum:"", diplome:null, attestation:null });
  const [err,  setErr]    = useState("");
  const diplomeRef = useRef(); const attestRef = useRef();
  const IsAdmin = role === "Administrateur";
  const set    = (k,v) => setForm(p=>({...p,[k]:v}));
  const setDoc = (k,v) => setDocs(p=>({...p,[k]:v}));

  const v1 = () => {
    if (!form.username.trim() || form.username.length < 4) return "Identifiant trop court (min. 4 caractères).";
    if (!form.fullName.trim()) return "Nom complet requis.";
    if (!form.email.includes("@")) return "Email invalide.";
    if (form.password.length < 6) return "Mot de passe trop court (min. 6 caractères).";
    if (form.password !== form.confirm) return "Les mots de passe ne correspondent pas.";
    return null;
  };
  const v2 = () => {
    if (!docs.rpps.trim() || !/^\d{11}$/.test(docs.rpps.trim())) return "Numéro RPPS invalide (11 chiffres).";
    if (!docs.ordreNum.trim()) return "Numéro d'ordre requis.";
    if (!docs.diplome) return "Diplôme requis.";
    if (!docs.attestation) return "Attestation Ordre requise.";
    return null;
  };

  const next = () => { const e=v1(); if(e){setErr(e);return;} setErr(""); if(IsAdmin) setStep(2); else submit(); };
  const submit = () => { const e=v2(); if(IsAdmin&&e){setErr(e);return;} setErr(""); onRegister({ username:form.username, password:form.password, role, name:form.fullName, validated:!IsAdmin, pendingValidation:IsAdmin }); setStep(3); };

  if (step===3) return (
    <div style={{ minHeight:"100vh", display:"flex", alignItems:"center", justifyContent:"center", background:`linear-gradient(135deg,${C.sidebar} 0%,#1A1A4A 100%)`, ...F }}>
      <div style={{ background:C.surface, borderRadius:20, padding:"48px 40px", width:420, textAlign:"center", boxShadow:"0 24px 64px rgba(0,0,0,0.3)" }}>
        <div style={{ width:72, height:72, background:C.successBg, borderRadius:"50%", display:"flex", alignItems:"center", justifyContent:"center", margin:"0 auto 20px", fontSize:36 }}>✅</div>
        <h2 style={{ fontSize:22, fontWeight:800, marginBottom:10 }}>Compte créé !</h2>
        {IsAdmin
          ? <div style={{ background:C.warningBg, border:`1px solid ${C.warning}30`, borderRadius:10, padding:"14px 16px", marginBottom:24, textAlign:"left" }}>
              <div style={{ fontWeight:700, color:C.warning, marginBottom:6 }}>⏳ Validation en attente</div>
              <div style={{ color:C.sub, fontSize:13, lineHeight:1.6 }}>Votre dossier (RPPS + documents) est en cours de vérification. Accès accordé sous 24–48h.</div>
            </div>
          : <p style={{ color:C.sub, lineHeight:1.6, marginBottom:24 }}>Votre compte Utilisateur est actif. Connectez-vous dès maintenant.</p>
        }
        <button style={{ ...mkBtn("primary", C.primary), width:"100%", padding:"13px" }} onClick={onBack}>{Ic.back} &nbsp;Retour à la connexion</button>
      </div>
    </div>
  );

  return (
    <div style={{ minHeight:"100vh", display:"flex", background:`linear-gradient(135deg,${C.sidebar} 0%,#1A1A4A 100%)`, ...F }}>
      <div style={{ flex:1, display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", padding:48, color:"#fff" }}>
        <Logo size={90} color={IsAdmin?C.primary:C.accent} />
        <h1 style={{ fontSize:28, fontWeight:900, marginTop:28, marginBottom:12, letterSpacing:"-0.02em" }}>{IsAdmin?"Compte Administrateur":"Compte Utilisateur"}</h1>
        <p style={{ color:C.sidebarText, fontSize:14, maxWidth:300, lineHeight:1.7, marginBottom:32 }}>
          {IsAdmin?"Accédez au système d'identification biométrique avancé.":"Enregistrez votre empreinte rétinienne et digitale en toute sécurité."}
        </p>
        {IsAdmin && (
          <div style={{ background:"rgba(255,255,255,0.06)", borderRadius:12, padding:"16px 20px", width:280 }}>
            <div style={{ color:C.muted, fontSize:11, fontWeight:700, textTransform:"uppercase", letterSpacing:"0.1em", marginBottom:10 }}>Documents requis</div>
            {["Numéro RPPS (11 chiffres)","Numéro d'ordre médical","Copie du diplôme","Attestation Ordre des Administrateurs"].map(d=>(
              <div key={d} style={{ display:"flex", gap:8, alignItems:"center", color:C.sidebarText, fontSize:13, marginBottom:7 }}>
                <div style={{ width:18, height:18, borderRadius:"50%", background:C.primary+"40", display:"flex", alignItems:"center", justifyContent:"center", fontSize:9 }}>✓</div>{d}
              </div>
            ))}
          </div>
        )}
      </div>
      <div style={{ width:480, display:"flex", alignItems:"center", justifyContent:"center", padding:32 }}>
        <div style={{ background:C.surface, borderRadius:20, padding:"40px 36px", width:"100%", boxShadow:"0 24px 64px rgba(0,0,0,0.3)" }}>
          <div style={{ display:"flex", alignItems:"center", gap:12, marginBottom:24 }}>
            <button style={{ ...mkBtn("ghost"), padding:"8px 12px" }} onClick={onBack}>{Ic.back}</button>
            <div>
              <h2 style={{ fontSize:18, fontWeight:800, margin:0 }}>{IsAdmin?"Créer un compte Administrateur":"Créer un compte Utilisateur"}</h2>
              {IsAdmin && <div style={{ color:C.muted, fontSize:12, marginTop:2 }}>Étape {step}/2 — {step===1?"Informations":"Justificatifs"}</div>}
            </div>
          </div>
          {IsAdmin && <div style={{ display:"flex", gap:6, marginBottom:24 }}>{[1,2].map(s=><div key={s} style={{ flex:1, height:4, borderRadius:4, background:step>=s?C.primary:C.border, transition:"background 0.3s" }}/>)}</div>}

          {step===1 && <>
            <label style={base.label}>Nom complet</label>
            <input style={base.input} type="text" placeholder={IsAdmin?" Prénom NOM":"Prénom NOM"} value={form.fullName} onChange={e=>set("fullName",e.target.value)} />
            <label style={base.label}>Email</label>
            <input style={base.input} type="email" placeholder="email@exemple.fr" value={form.email} onChange={e=>set("email",e.target.value)} />
            <label style={base.label}>Identifiant</label>
            <input style={base.input} type="text" placeholder="Choisissez un identifiant (min. 4 car.)" value={form.username} onChange={e=>set("username",e.target.value)} />
            <label style={base.label}>Mot de passe</label>
            <input style={base.input} type="password" placeholder="Min. 6 caractères" value={form.password} onChange={e=>set("password",e.target.value)} />
            <label style={base.label}>Confirmer</label>
            <input style={base.input} type="password" placeholder="••••••••" value={form.confirm} onChange={e=>set("confirm",e.target.value)} />
            {err && <div style={{ background:C.redBg, color:C.red, border:`1px solid ${C.red}30`, borderRadius:8, padding:"10px 14px", marginBottom:14, fontSize:13 }}>⚠ {err}</div>}
            <button style={{ ...mkBtn("primary", IsAdmin?C.primary:C.accent), width:"100%", padding:"13px", fontSize:15 }} onClick={next}>
              {IsAdmin ? "Suivant — Justificatifs →" : <>{Ic.plus}&nbsp;Créer mon compte</>}
            </button>
          </>}

          {step===2 && <>
            <div style={{ background:C.warningBg, border:`1px solid ${C.warning}30`, borderRadius:10, padding:"12px 14px", marginBottom:18, fontSize:13 }}>
              <div style={{ fontWeight:700, color:C.warning, marginBottom:4 }}>🔍 Vérification d'identité médicale</div>
              <div style={{ color:C.sub }}>Ces documents sont vérifiés sous 24–48h avant activation.</div>
            </div>
            <label style={base.label}>Numéro RPPS *</label>
            <input style={base.input} type="text" placeholder="11 chiffres" maxLength={11} value={docs.rpps} onChange={e=>setDoc("rpps",e.target.value.replace(/\D/g,""))} />
            <label style={base.label}>Numéro d'ordre *</label>
            <input style={base.input} type="text" placeholder="ex : 75-12345" value={docs.ordreNum} onChange={e=>setDoc("ordreNum",e.target.value)} />
            {[["diplome","Diplôme de Administrateure *",diplomeRef],["attestation","Attestation Ordre des Administrateurs *",attestRef]].map(([key,lbl,ref])=>(
              <div key={key}>
                <label style={base.label}>{lbl}</label>
                <input ref={ref} type="file" accept=".pdf,.jpg,.jpeg,.png" style={{ display:"none" }} onChange={e=>setDoc(key,e.target.files[0])} />
                <div onClick={()=>ref.current.click()} style={{ border:`2px dashed ${docs[key]?C.success:C.border}`, borderRadius:10, padding:"16px", textAlign:"center", cursor:"pointer", background:docs[key]?C.successBg:C.bg, marginBottom:14, transition:"all 0.15s" }}>
                  {docs[key]
                    ? <><div style={{ fontWeight:600, color:C.success, fontSize:13 }}>✅ {docs[key].name}</div><div style={{ color:C.muted, fontSize:11 }}>{(docs[key].size/1024).toFixed(0)} Ko</div></>
                    : <><div style={{ color:C.primary, fontSize:13, fontWeight:600 }}>📎 Cliquer pour uploader</div><div style={{ color:C.muted, fontSize:11 }}>PDF, JPG, PNG</div></>
                  }
                </div>
              </div>
            ))}
            {err && <div style={{ background:C.redBg, color:C.red, border:`1px solid ${C.red}30`, borderRadius:8, padding:"10px 14px", marginBottom:14, fontSize:13 }}>⚠ {err}</div>}
            <div style={{ display:"flex", gap:10 }}>
              <button style={{ ...mkBtn("ghost"), flex:1, padding:"12px" }} onClick={()=>{setErr("");setStep(1);}}>{Ic.back}&nbsp;Retour</button>
              <button style={{ ...mkBtn("primary",C.primary), flex:2, padding:"12px" }} onClick={submit}>{Ic.check}&nbsp;Soumettre le dossier</button>
            </div>
          </>}
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// LOGIN
// ═══════════════════════════════════════════════════════════════════════════════
function LoginPage({ onLogin, users, onRegister }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [err, setErr]     = useState("");
  const [tab, setTab]     = useState("Administrateur");
  const [view, setView]   = useState("login");
  const [retineFile, setRetineFile] = useState(null);
  const [loading, setLoading]       = useState(false);
  const retineRef = useRef();
  const IsAdmin = tab === "Administrateur";

  if (view==="register-Administrateur") return <RegisterPage role="Administrateur" onBack={()=>setView("login")} onRegister={u=>{onRegister(u);setView("login");}} />;
  if (view==="register-client")  return <RegisterPage role="client"  onBack={()=>setView("login")} onRegister={u=>{onRegister(u);setView("login");}} />;

  const login = async () => {
    const u = users[username];
    if (!u || u.password!==password) { setErr("Identifiants incorrects."); return; }
    if (u.role!==tab) { setErr(`Ce compte est de type "${u.role==="Administrateur"?"Administrateur":"Utilisateur"}". Changez d'onglet.`); return; }
    if (u.pendingValidation) { setErr("Compte en attente de validation médicale (24–48h)."); return; }

    // Vérification biométrique : la rétine doit correspondre à celle enrôlée par l'admin
    if (u.role === "client") {
      if (!u.retineVector) { setErr("Aucune rétine enrôlée pour ce compte. Contactez votre administrateur."); return; }
      if (!retineFile)     { setErr("Importez votre image rétinienne pour vous authentifier."); return; }
      setLoading(true); setErr("");
      try {
        const res = await processBiometric(retineFile, "retine");
        if (!vectorsMatch(res.optimizedArray, u.retineVector)) {
          setLoading(false);
          setErr("Rétine non reconnue — accès refusé.");
          return;
        }
      } catch(e) {
        setLoading(false);
        setErr(`Erreur lors de l'analyse rétinienne : ${e.message}`);
        return;
      }
      setLoading(false);
    }

    setErr(""); onLogin({username,...u});
  };

  return (
    <div style={{ minHeight:"100vh", display:"flex", background:`linear-gradient(135deg,${C.sidebar} 0%,#1A1A4A 100%)`, ...F }}>
      {/* Left */}
      <div style={{ flex:1, display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", padding:48 }}>
        <Logo size={90} color={C.primary} />
        
        <p style={{ color:C.sidebarText, fontSize:15, maxWidth:340, lineHeight:1.7, textAlign:"center", marginBottom:40 }}>
          Solution de contrôle d'accès biométrique multimodale par rétine et empreinte digitale
        </p>
        <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:12, width:340 }}>
          {[["👁️","Rétine","Vaisseaux sanguins"],["🫆","Empreinte","Lignes caractéristiques"],["🧬","Vecteur features","16 dimensions"],["🔐","Chiffrement","AES-256-GCM"]].map(([icon,t,d])=>(
            <div key={t} style={{ background:"rgba(255,255,255,0.06)", borderRadius:10, padding:"14px", border:"1px solid rgba(255,255,255,0.08)" }}>
              <div style={{ fontSize:22, marginBottom:6 }}>{icon}</div>
              <div style={{ color:"#fff", fontWeight:700, fontSize:13 }}>{t}</div>
              <div style={{ color:C.muted, fontSize:11, marginTop:2 }}>{d}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Right */}
      <div style={{ width:460, display:"flex", alignItems:"center", justifyContent:"center", padding:32 }}>
        <div style={{ background:C.surface, borderRadius:20, padding:"40px 36px", width:"100%", boxShadow:"0 24px 64px rgba(0,0,0,0.3)" }}>
          <div style={{ display:"flex", background:C.bg, borderRadius:10, padding:4, marginBottom:28, border:`1px solid ${C.border}` }}>
            {[["Administrateur",Ic.user,"Administrateur"],["client",Ic.user,"Utilisateur"]].map(([role,icon,label])=>(
              <button key={role} onClick={()=>{setTab(role);setErr("");}} style={{ flex:1, display:"flex", alignItems:"center", justifyContent:"center", gap:8, padding:"10px", borderRadius:8, border:"none", cursor:"pointer", fontWeight:600, fontSize:14, transition:"all 0.15s", background:tab===role?(role==="Administrateur"?C.primary:C.accent):"transparent", color:tab===role?"#fff":C.sub }}>{icon} {label}</button>
            ))}
          </div>

          <h2 style={{ fontSize:20, fontWeight:800, marginBottom:4 }}>{IsAdmin?"Connexion Administrateur":"Connexion Utilisateur"}</h2>
          <p style={{ color:C.sub, fontSize:13, marginBottom:24 }}>{IsAdmin?"Accès au système biométrique complet.":"Consultez vos analyses biométriques."}</p>

          <label style={base.label}>Identifiant</label>
          <input style={base.input} type="text" placeholder={IsAdmin?"ex : admin":"ex : Utilisateur1"} value={username} onChange={e=>setUsername(e.target.value)} onKeyDown={e=>e.key==="Enter"&&login()} />
          <label style={base.label}>Mot de passe</label>
          <input style={base.input} type="password" placeholder="••••••••" value={password} onChange={e=>setPassword(e.target.value)} onKeyDown={e=>e.key==="Enter"&&login()} />

          {!IsAdmin && (
            <>
              <label style={base.label}>Authentification rétinienne *</label>
              <input ref={retineRef} type="file" accept=".png,.jpg,.jpeg,.bmp" style={{ display:"none" }}
                onChange={e=>setRetineFile(e.target.files[0]||null)} />
              <div onClick={()=>retineRef.current.click()}
                style={{ border:`2px dashed ${retineFile?C.success:C.border}`, borderRadius:10, padding:"16px", textAlign:"center", cursor:"pointer", background:retineFile?C.successBg:C.bg, marginBottom:16, transition:"all 0.15s" }}>
                {retineFile
                  ? <><div style={{ fontSize:20 }}>✅</div><div style={{ color:C.success, fontSize:13, fontWeight:700 }}>{retineFile.name}</div></>
                  : <><div style={{ fontSize:24 }}>👁️</div><div style={{ color:C.accent, fontSize:13 }}>Importer votre image rétinienne</div><div style={{ color:C.muted, fontSize:11 }}>PNG, JPG, BMP</div></>
                }
              </div>
            </>
          )}

          <button style={{ ...mkBtn("primary", IsAdmin?C.primary:C.accent), width:"100%", padding:"13px", fontSize:15, marginBottom:12, opacity:loading?0.6:1 }} onClick={login} disabled={loading}>
            {loading ? <><span style={{ animation:"spin 1s linear infinite", display:"inline-block" }}>⟳</span>&nbsp;Vérification de la rétine...</> : "Se connecter"}
          </button>
          <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
          {err && <div style={{ background:C.redBg, color:C.red, border:`1px solid ${C.red}30`, borderRadius:8, padding:"10px 14px", marginBottom:12, fontSize:13 }}>⚠ {err}</div>}

          <div style={{ borderTop:`1px solid ${C.border}`, paddingTop:14, marginTop:4 }}>
            {IsAdmin ? (
              <>
                <div style={{ color:C.muted, fontSize:12, textAlign:"center", marginBottom:10 }}>Pas encore de compte ?</div>
                <button style={{ ...mkBtn("soft", C.primary), width:"100%", padding:"11px" }} onClick={()=>setView("register-Administrateur")}>
                  {Ic.plus}&nbsp;Créer un compte Administrateur
                </button>
              </>
            ) : (
              <div style={{ display:"flex", gap:10, alignItems:"flex-start", padding:"12px 14px", background:C.bg, borderRadius:10, border:`1px solid ${C.border}`, fontSize:12, color:C.sub, lineHeight:1.5 }}>
                <span style={{ fontSize:16 }}>🔒</span>
                <span>Les comptes utilisateurs sont créés par un administrateur. Contactez votre administrateur pour obtenir vos identifiants.</span>
              </div>
            )}
          </div>

          <div style={{ marginTop:14, padding:"12px 14px", background:C.bg, borderRadius:10, border:`1px solid ${C.border}`, fontSize:12 }}>
            <div style={{ fontWeight:700, color:C.sub, marginBottom:6 }}>{IsAdmin ? "Comptes de démo" : "Pour tester un accès utilisateur"}</div>
            {IsAdmin
              ? <><div style={{ color:C.muted }}>👨‍⚕️ <code>admin</code> / <code>admin123</code></div><div style={{ color:C.muted, marginTop:3 }}>👨‍⚕️ <code>Administrateur</code> / <code>abcd</code></div></>
              : <div style={{ color:C.muted, lineHeight:1.6 }}>Connectez-vous en admin, créez un utilisateur avec son image de rétine, puis revenez ici et importez <strong>la même image</strong> pour obtenir l'accès.</div>
            }
          </div>
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// SHELL
// ═══════════════════════════════════════════════════════════════════════════════
function Shell({ user, page, setPage, navItems, onLogout, sidebarColor, activeColor, topRight, children }) {
  return (
    <div style={{ ...F, display:"flex", minHeight:"100vh", background:C.bg, color:C.text, fontSize:14 }}>
      <aside style={{ width:240, background:sidebarColor, display:"flex", flexDirection:"column", position:"fixed", top:0, left:0, bottom:0, zIndex:100 }}>
        <div style={{ padding:"22px 20px 16px", borderBottom:"1px solid rgba(255,255,255,0.07)" }}>
          <Logo size={45} color={activeColor} />
        </div>
        <nav style={{ flex:1, padding:"14px 12px" }}>
          <div style={{ color:C.muted, fontSize:10, fontWeight:700, letterSpacing:"0.1em", textTransform:"uppercase", padding:"0 8px", marginBottom:8 }}>Navigation</div>
          {navItems.map(n=><button key={n.id} style={mkNav(page===n.id,activeColor)} onClick={()=>setPage(n.id)}>{n.icon} {n.label}</button>)}
        </nav>
        <div style={{ padding:"14px 12px", borderTop:"1px solid rgba(255,255,255,0.07)" }}>
          <div style={{ display:"flex", alignItems:"center", gap:10, padding:"8px", marginBottom:4 }}>
            <div style={{ width:32, height:32, borderRadius:"50%", background:activeColor, display:"flex", alignItems:"center", justifyContent:"center", color:"#fff", fontWeight:700, fontSize:13 }}>{user.name.charAt(0)}</div>
            <div><div style={{ color:"#fff", fontSize:13, fontWeight:600 }}>{user.name}</div><div style={{ color:C.muted, fontSize:11 }}>{user.role==="Administrateur"?"Administrateur":"Utilisateur"}</div></div>
          </div>
          <button style={{ display:"flex", alignItems:"center", gap:6, color:C.muted, background:"none", border:"none", fontSize:13, cursor:"pointer", padding:"8px", width:"100%", borderRadius:6 }} onClick={onLogout}>{Ic.logout} Se déconnecter</button>
        </div>
      </aside>
      <main style={{ marginLeft:240, flex:1, display:"flex", flexDirection:"column" }}>
        <div style={{ background:C.surface, borderBottom:`1px solid ${C.border}`, padding:"0 32px", height:60, display:"flex", alignItems:"center", justifyContent:"space-between", position:"sticky", top:0, zIndex:50 }}>
          <span style={{ fontWeight:700, fontSize:17 }}>{navItems.find(n=>n.id===page)?.label||""}</span>
          <div style={{ display:"flex", gap:10, alignItems:"center" }}>{topRight}<span style={mkBadge(C.success)}>● Opérationnel</span></div>
        </div>
        <div style={{ padding:32, flex:1 }}>{children}</div>
      </main>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// UPLOAD PANEL — Mode rétine seule ou rétine + empreinte (sécurité renforcée)
// ═══════════════════════════════════════════════════════════════════════════════
function UploadPanel({ onResult, accentColor=C.primary, defaultId="", showId=true }) {
  const [securityMode, setSecurityMode] = useState("retine"); // "retine" | "double"
  const [fileRetine,    setFileRetine]    = useState(null);
  const [fileEmpreinte, setFileEmpreinte] = useState(null);
  const [pid,     setPid]     = useState(defaultId);
  const [dragR,   setDragR]   = useState(false);
  const [dragE,   setDragE]   = useState(false);
  const [loading, setLoading] = useState(false);
  const [msg,     setMsg]     = useState("");
  const [err,     setErr]     = useState("");
  const fileRefR = useRef();
  const fileRefE = useRef();

  const canRun = fileRetine && (securityMode === "retine" || fileEmpreinte);

  const run = async () => {
    if (!canRun) return;
    setLoading(true); setErr("");
    try {
      const steps = [
        "Chargement de l'image rétinienne...",
        "Segmentation des vaisseaux...",
        "Squelettisation (1px)...",
        "Extraction OvLen, TI, MedTor, D1, D2...",
        securityMode==="double" ? "Traitement empreinte digitale..." : "Génération des vecteurs...",
        "Finalisation...",
      ];
      for (const s of steps) { setMsg(s); await new Promise(r=>setTimeout(r,350+Math.random()*200)); }

      // Pipeline rétine
      const retineResult = await processBiometric(fileRetine, "retine");

      let empreinteResult = null;
      if (securityMode === "double" && fileEmpreinte) {
        empreinteResult = await processBiometric(fileEmpreinte, "empreinte");
      }
      onResult({
        retine: retineResult,
        empreinte: empreinteResult,
        securityMode,
        UtilisateurId: pid || "Anonyme",
        fileNameRetine: fileRetine.name,
        fileNameEmpreinte: fileEmpreinte?.name || null,
        date: new Date().toLocaleString("fr-FR").slice(0,16),
      });
    } catch(e) { setErr(`Erreur : ${e.message}`); }
    finally { setLoading(false); setMsg(""); }
  };

  const DropZone = ({ file, setFile, drag, setDrag, fileRef, label, icon, accept }) => (
    <>
      <input ref={fileRef} type="file" accept={accept} style={{ display:"none" }}
        onChange={e=>e.target.files[0]&&setFile(e.target.files[0])} />
      <div onClick={()=>fileRef.current.click()}
        onDragOver={e=>{e.preventDefault();setDrag(true);}} onDragLeave={()=>setDrag(false)}
        onDrop={e=>{e.preventDefault();setDrag(false);const f=e.dataTransfer.files[0];if(f)setFile(f);}}
        style={{ border:`2px dashed ${drag?accentColor:file?C.success:C.border}`, borderRadius:10, padding:"20px", textAlign:"center", cursor:"pointer", background:drag?accentColor+"0A":file?C.successBg:C.bg, transition:"all 0.15s", marginBottom:12 }}>
        {file
          ? <><div style={{ fontSize:24, marginBottom:4 }}>✅</div><div style={{ fontWeight:700, color:C.success, fontSize:13 }}>{file.name}</div><div style={{ color:C.muted, fontSize:11 }}>{(file.size/1024).toFixed(1)} Ko</div></>
          : <><div style={{ fontSize:28, marginBottom:6 }}>{icon}</div><div style={{ fontWeight:600, color:accentColor, fontSize:13 }}>{label}</div><div style={{ color:C.muted, fontSize:11 }}>PNG, JPG, BMP</div></>
        }
      </div>
    </>
  );

  return (
    <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:24 }}>
      {/* Left : choix du mode + upload */}
      <div style={base.card}>
        <div style={{ fontWeight:700, fontSize:15, marginBottom:4 }}>Mode d'authentification</div>
        <div style={{ color:C.sub, fontSize:13, marginBottom:16 }}>Rétine seule ou rétine + empreinte (sécurité renforcée)</div>

        {/* Sélection mode */}
        <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:10, marginBottom:20 }}>
          {[
            ["retine","👁️","Rétine seule","Niveau standard"],
            ["double","👁️🫆","Rétine + Empreinte","Sécurité renforcée"],
          ].map(([m,icon,t,d])=>(
            <div key={m} onClick={()=>setSecurityMode(m)}
              style={{ border:`2px solid ${securityMode===m?accentColor:C.border}`, borderRadius:10, padding:"14px", cursor:"pointer", background:securityMode===m?accentColor+"0E":C.bg, textAlign:"center", transition:"all 0.15s" }}>
              <div style={{ fontSize:22, marginBottom:4 }}>{icon}</div>
              <div style={{ fontWeight:700, fontSize:13, color:securityMode===m?accentColor:C.text }}>{t}</div>
              <div style={{ color:C.muted, fontSize:11, marginTop:2 }}>{d}</div>
            </div>
          ))}
        </div>

        {/* Upload rétine */}
        <label style={base.label}>Image rétinienne *</label>
        <DropZone file={fileRetine} setFile={setFileRetine} drag={dragR} setDrag={setDragR}
          fileRef={fileRefR} label="Déposer l'image de rétine" icon="👁️" accept=".png,.jpg,.jpeg,.bmp" />

        {/* Upload empreinte si mode double */}
        {securityMode === "double" && (
          <>
            <label style={base.label}>Image empreinte digitale *</label>
            <DropZone file={fileEmpreinte} setFile={setFileEmpreinte} drag={dragE} setDrag={setDragE}
              fileRef={fileRefE} label="Déposer l'image d'empreinte" icon="🫆" accept=".png,.jpg,.jpeg,.bmp" />
          </>
        )}

        {showId && <>
          <label style={base.label}>Identifiant</label>
          <input style={base.input} type="text" placeholder="Ex : Jean Martin" value={pid} onChange={e=>setPid(e.target.value)} />
        </>}

        {securityMode==="double" && (
          <div style={{ padding:"10px 12px", background:`${accentColor}10`, borderRadius:8, border:`1px solid ${accentColor}30`, fontSize:12, color:accentColor, marginBottom:12 }}>
            🔒 Mode renforcé : les deux biométries doivent correspondre
          </div>
        )}
        <div style={{ padding:"10px 12px", background:C.bg, borderRadius:8, border:`1px solid ${C.border}`, fontSize:12, color:C.muted }}>
          🔒 Images non stockées — seuls les vecteurs optimisés sont conservés
        </div>
      </div>

      {/* Right : pipeline + lancer */}
      <div style={base.card}>
        <div style={{ fontWeight:700, fontSize:15, marginBottom:4 }}>Pipeline de traitement</div>
        <div style={{ color:C.sub, fontSize:13, marginBottom:16 }}>Étapes exécutées automatiquement</div>

        {[
          { n:"1", icon:"🖼️", t:"Prétraitement", d:"Redimensionnement 512×512, normalisation" },
          { n:"2", icon:"🩸", t:"Segmentation vaisseaux", d:"Détection des vaisseaux rétiniens par DoG multi-échelle" },
          { n:"3", icon:"📐", t:"Squelettisation", d:"Amincissement à 1 pixel (Zhang-Suen thinning)" },
          { n:"4", icon:"🧬", t:"Vecteur complet", d:"OvLen, TI, MedTor, D1, D2 + directions + bifurcations" },
          { n:"5", icon:"⚡", t:"Vecteur optimisé", d:securityMode==="double"?"Rétine : OvLen/TI/MedTor/D1/D2 · Empreinte : minuties/orientation":"5 features rétine : OvLen, TI, MedTor, D1, D2" },
        ].map(s=>(
          <div key={s.n} style={{ display:"flex", gap:12, marginBottom:12, padding:"12px", background:C.bg, borderRadius:9, border:`1px solid ${C.border}` }}>
            <div style={{ width:26,height:26,borderRadius:"50%",background:accentColor+"18",color:accentColor,fontWeight:800,fontSize:13,display:"flex",alignItems:"center",justifyContent:"center",flexShrink:0 }}>{s.n}</div>
            <div><div style={{ fontWeight:700,fontSize:13,marginBottom:2 }}>{s.icon} {s.t}</div><div style={{ color:C.sub,fontSize:11,lineHeight:1.5 }}>{s.d}</div></div>
          </div>
        ))}

        <div style={{ padding:"12px 14px", background:accentColor+"0E", borderRadius:9, marginBottom:16, fontSize:13 }}>
          <div style={{ fontWeight:700, color:accentColor, marginBottom:6 }}>Récapitulatif</div>
          <div style={{ color:C.sub }}>Mode : <strong style={{ color:C.text }}>{securityMode==="double"?"👁️🫆 Rétine + Empreinte":"👁️ Rétine seule"}</strong></div>
          <div style={{ color:C.sub, marginTop:3 }}>Rétine : <strong style={{ color:fileRetine?C.text:C.muted }}>{fileRetine?.name||"Non sélectionnée"}</strong></div>
          {securityMode==="double" && <div style={{ color:C.sub, marginTop:3 }}>Empreinte : <strong style={{ color:fileEmpreinte?C.text:C.muted }}>{fileEmpreinte?.name||"Non sélectionnée"}</strong></div>}
        </div>

        {err && <div style={{ background:C.redBg,color:C.red,border:`1px solid ${C.red}30`,borderRadius:8,padding:"10px 14px",marginBottom:12,fontSize:13 }}>⚠ {err}</div>}

        <button style={{ ...mkBtn("primary",accentColor), width:"100%", padding:"14px", fontSize:15, opacity:(!canRun||loading)?0.6:1 }}
          onClick={run} disabled={!canRun||loading}>
          {loading ? <><span style={{ animation:"spin 1s linear infinite", display:"inline-block" }}>⟳</span>&nbsp;{msg}</> : <>{Ic.scan}&nbsp;Lancer l'analyse</>}
        </button>
        {loading && <div style={{ height:5,background:C.border,borderRadius:4,overflow:"hidden",position:"relative",marginTop:10 }}><div style={{ position:"absolute",top:0,left:0,height:"100%",background:accentColor,animation:"progress 3s ease-in-out forwards",borderRadius:4 }}/></div>}
        <style>{`@keyframes spin{to{transform:rotate(360deg)}} @keyframes progress{from{width:0%}to{width:100%}}`}</style>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// RESULTS PANEL — Vecteur complet + Vecteur optimisé
// ═══════════════════════════════════════════════════════════════════════════════
function ResultsPanel({ result, accentColor=C.primary, onNew, onEnroll, onAuth }) {
  const [viewMode, setViewMode] = useState("overlay");
  const [showSkel, setShowSkel] = useState(false);

  if (!result) return (
    <div style={{ textAlign:"center", padding:"80px 0", color:C.sub }}>
      <div style={{ fontSize:52, marginBottom:16 }}>🧬</div>
      <div style={{ fontSize:16, fontWeight:600, marginBottom:8 }}>Aucune analyse effectuée</div>
      <div style={{ marginBottom:24 }}>Importez une image rétinienne pour lancer le pipeline.</div>
      {onNew&&<button style={mkBtn("primary",accentColor)} onClick={onNew}>{Ic.scan}&nbsp;Analyser une image</button>}
    </div>
  );

  const retine = result.retine;
  const empreinte = result.empreinte;
  const isDouble = result.securityMode === "double";

  // Affichage d'un bloc de vecteur
  const VectorBlock = ({ title, data, color, keys }) => (
    <div style={{ marginBottom:16 }}>
      <div style={{ fontWeight:700, fontSize:13, color, marginBottom:8, textTransform:"uppercase", letterSpacing:"0.05em" }}>{title}</div>
      <div style={{ background:"#080F1E", borderRadius:10, padding:"14px 16px", fontFamily:"monospace", fontSize:11, lineHeight:1.9 }}>
        {keys ? keys.map((k,i)=>(
          <div key={k}>
            <span style={{ color:"#94A3B8" }}>{k}: </span>
            <span style={{ color:"#4ADE80", fontWeight:700 }}>{typeof data[k]==='number'?data[k].toFixed(6):data[k]}</span>
          </div>
        )) : Object.entries(data).map(([k,v])=>(
          <div key={k}>
            <span style={{ color:"#94A3B8" }}>{k}: </span>
            <span style={{ color:"#60A5FA" }}>{typeof v==='number'?v.toFixed(6):v}</span>
          </div>
        ))}
      </div>
    </div>
  );

  return (
    <>
      {/* Header */}
      <div style={{ display:"flex", alignItems:"flex-start", justifyContent:"space-between", marginBottom:20, flexWrap:"wrap", gap:12 }}>
        <div>
          <div style={{ display:"flex", alignItems:"center", gap:8, marginBottom:6, flexWrap:"wrap" }}>
            <span style={mkChip(C.success)}>✓ Pipeline complet</span>
            <span style={mkBadge(C.primary)}>👁️ Rétine</span>
            {isDouble && <span style={mkBadge(C.accent)}>🫆 Empreinte</span>}
            {isDouble && <span style={mkBadge(C.warning)}>🔒 Sécurité renforcée</span>}
            <span style={{ color:C.muted, fontSize:13 }}>{result.UtilisateurId}</span>
          </div>
          <h2 style={{ fontSize:18, fontWeight:800 }}>Segmentation · Squelette · Vecteurs biométriques</h2>
        </div>
        <div style={{ display:"flex", gap:8, flexWrap:"wrap" }}>
          {onNew && <button style={mkBtn("ghost")} onClick={onNew}>{Ic.scan}&nbsp;Nouvelle analyse</button>}
          {onEnroll && <button style={mkBtn("soft",C.success)} onClick={()=>onEnroll(result)}>{Ic.plus}&nbsp;Enrôler</button>}
          {onAuth && <button style={mkBtn("primary",accentColor)} onClick={()=>onAuth(result)}>{Ic.search}&nbsp;Authentifier</button>}
        </div>
      </div>

      {/* Images rétine */}
      <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:20, marginBottom:20 }}>
        <div style={base.card}>
          <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:12 }}>
            <div style={{ fontWeight:700, fontSize:14 }}>👁️ Visualisation rétine</div>
            <div style={{ display:"flex", gap:6 }}>
              {[["overlay","Superposition"],["mask","Masque"],["skel","Squelette"],["original","Original"]].map(([v,l])=>(
                <button key={v} style={{ ...mkBtn((showSkel?v==="skel":v===viewMode)&&!showSkel||showSkel&&v==="skel"?"primary":"ghost",accentColor), padding:"4px 9px", fontSize:11 }}
                  onClick={()=>{if(v==="skel"){setShowSkel(true);}else{setShowSkel(false);setViewMode(v);}}}>
                  {l}
                </button>
              ))}
            </div>
          </div>
          <div style={{ background:"#080F1E", borderRadius:10, overflow:"hidden", display:"flex", alignItems:"center", justifyContent:"center", height:260 }}>
            <img key={showSkel?"skel":viewMode}
              src={showSkel?retine.skelUrl:viewMode==="mask"?retine.maskUrl:viewMode==="original"?retine.originalUrl:retine.overlayUrl}
              alt="rétine" style={{ maxWidth:"100%", maxHeight:"100%", objectFit:"contain" }} />
          </div>
          <div style={{ marginTop:8, fontSize:11, color:C.muted, textAlign:"center" }}>
            {showSkel?"📐 Squelette vasculaire (1px d'épaisseur)":viewMode==="overlay"?"🔴 Rouge = vaisseaux":viewMode==="mask"?"⬜ Masque binaire":"🖼️ Original"}
          </div>
          {/* Stats rapides rétine */}
          <div style={{ display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:8, marginTop:12 }}>
            {[
              ["OvLen", retine.stats.OvLen?.toFixed(0)||"-", "Longueur totale"],
              ["TI", retine.stats.TI?.toFixed(3)||"-", "Tortuosité"],
              ["D1", retine.stats.D1?.toFixed(2)||"-", "Diamètre moy."],
            ].map(([k,v,l])=>(
              <div key={k} style={{ background:C.bg, borderRadius:8, padding:"8px", border:`1px solid ${C.border}`, textAlign:"center" }}>
                <div style={{ fontWeight:800, fontSize:16, color:accentColor }}>{v}</div>
                <div style={{ fontSize:10, color:C.muted, marginTop:1 }}>{l}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Empreinte si mode double */}
        {isDouble && empreinte ? (
          <div style={base.card}>
            <div style={{ fontWeight:700, fontSize:14, marginBottom:12 }}>🫆 Visualisation empreinte</div>
            <div style={{ background:"#080F1E", borderRadius:10, overflow:"hidden", display:"flex", alignItems:"center", justifyContent:"center", height:260 }}>
              <img src={empreinte.maskUrl} alt="empreinte" style={{ maxWidth:"100%", maxHeight:"100%", objectFit:"contain" }} />
            </div>
            <div style={{ marginTop:8, fontSize:11, color:C.muted, textAlign:"center" }}>⬜ Crêtes segmentées</div>
            <div style={{ display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:8, marginTop:12 }}>
              {[
                ["Minuties", empreinte.stats.nbMinutiae||"-", "Total"],
                ["Bifurc.", empreinte.stats.nbBifurcations||"-", "Bifurcations"],
                ["Term.", empreinte.stats.nbTerminations||"-", "Terminaisons"],
              ].map(([k,v,l])=>(
                <div key={k} style={{ background:C.bg, borderRadius:8, padding:"8px", border:`1px solid ${C.border}`, textAlign:"center" }}>
                  <div style={{ fontWeight:800, fontSize:16, color:C.accent }}>{v}</div>
                  <div style={{ fontSize:10, color:C.muted, marginTop:1 }}>{l}</div>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div style={{ ...base.card, display:"flex", flexDirection:"column", gap:12 }}>
            <div style={{ fontWeight:700, fontSize:14, marginBottom:4 }}>🧬 Vecteur optimisé rétine</div>
            <div style={{ color:C.sub, fontSize:12, marginBottom:4 }}>5 features — utilisé pour l'authentification</div>
            <VectorBlock title="Features optimisées" data={retine.optimizedVector}
              color={accentColor} keys={["OvLen","TI","MedTor","D1","D2"]} />
            <div style={{ padding:"10px 12px", background:C.primaryLight, borderRadius:8, fontSize:12, color:C.primary }}>
              💡 Ces 5 features sont enregistrées dans la base lors de l'enrôlement
            </div>
          </div>
        )}
      </div>

      {/* Vecteurs complets */}
      <div style={{ display:"grid", gridTemplateColumns: isDouble?"1fr 1fr 1fr 1fr":"1fr 1fr", gap:20 }}>
        {/* Vecteur complet rétine */}
        <div style={base.card}>
          <div style={{ fontWeight:700, fontSize:14, marginBottom:8 }}>📊 Vecteur complet rétine</div>
          <div style={{ color:C.sub, fontSize:12, marginBottom:12 }}>{Object.keys(retine.fullVector).length} features extraites</div>
          <VectorBlock title="Morphologie vasculaire" data={retine.fullVector} color={C.primary}
            keys={["OvLen","TI","MedTor","D1","D2","nbSegments","nbBifurcations","nbTerminations","density"]} />
        </div>

        {/* Vecteur optimisé rétine */}
        <div style={base.card}>
          <div style={{ fontWeight:700, fontSize:14, marginBottom:8 }}>⚡ Vecteur optimisé rétine</div>
          <div style={{ color:C.sub, fontSize:12, marginBottom:12 }}>5 features · utilisé pour l'authentification</div>
          <VectorBlock title="Features optimisées" data={retine.optimizedVector}
            color={accentColor} keys={["OvLen","TI","MedTor","D1","D2"]} />
          <div style={{ marginTop:12, background:"#080F1E", borderRadius:8, padding:"10px 12px", fontFamily:"monospace", fontSize:11, color:"#4ADE80" }}>
            [{retine.optimizedArray.map(v=>v.toFixed(4)).join(", ")}]
          </div>
        </div>

        {/* Vecteurs empreinte si double */}
        {isDouble && empreinte && <>
          <div style={base.card}>
            <div style={{ fontWeight:700, fontSize:14, marginBottom:8 }}>📊 Vecteur complet empreinte</div>
            <div style={{ color:C.sub, fontSize:12, marginBottom:12 }}>{Object.keys(empreinte.fullVector).length} features extraites</div>
            <VectorBlock title="Minuties & orientations" data={empreinte.fullVector} color={C.accent}
              keys={["nbMinutiae","nbBifurcations","nbTerminations","minutiaeDensity","meanOrientation","orientationVariation","density"]} />
          </div>
          <div style={base.card}>
            <div style={{ fontWeight:700, fontSize:14, marginBottom:8 }}>⚡ Vecteur optimisé empreinte</div>
            <div style={{ color:C.sub, fontSize:12, marginBottom:12 }}>6 features · authentification renforcée</div>
            <VectorBlock title="Features optimisées" data={empreinte.optimizedVector} color={C.accent}
              keys={["nbMinutiae","nbBifurcations","nbTerminations","minutiaeDensity","meanOrientation","orientationVariation"]} />
            <div style={{ marginTop:12, background:"#080F1E", borderRadius:8, padding:"10px 12px", fontFamily:"monospace", fontSize:11, color:"#60A5FA" }}>
              [{empreinte.optimizedArray.map(v=>v.toFixed(4)).join(", ")}]
            </div>
          </div>
        </>}
      </div>
    </>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// BASE DE DONNÉES BIOMÉTRIQUE — Enrôlement + Authentification
// ═══════════════════════════════════════════════════════════════════════════════

// Seuils d'authentification
const THRESHOLD_RETINE = 0.05;     // distance euclidienne normalisée rétine
const THRESHOLD_EMPREINTE = 0.05;  // distance euclidienne normalisée empreinte

function BiometricDB({ database, setDatabase, accentColor=C.primary }) {
  const [accessPopup, setAccessPopup] = useState(null);
  const [tab, setTab] = useState("base"); // base | enroll | auth
  const [enrollName, setEnrollName] = useState("");
  const [enrollResult, setEnrollResult] = useState(null);
  const [authResult, setAuthResult] = useState(null);
  const [authFileR, setAuthFileR] = useState(null);
  const [authFileE, setAuthFileE] = useState(null);
  const [authMode, setAuthMode] = useState("retine");
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState("");
  const fileRefR = useRef(); const fileRefE = useRef();
  const fileRefAR = useRef(); const fileRefAE = useRef();

  // ── Enrôlement ────────────────────────────────────────────────────────────
  const handleEnrollFile = async (file, mode, setter) => {
    const result = await processBiometric(file, mode);
    setter(prev => prev ? {...prev, [mode]: result} : {[mode]: result});
  };

  const confirmEnroll = () => {
    if (!enrollName.trim() || !enrollResult?.retine) return;
    const entry = {
      id: Date.now().toString(),
      name: enrollName.trim(),
      date: new Date().toLocaleString("fr-FR").slice(0,16),
      retineVector: enrollResult.retine.optimizedArray,
      empreinteVector: enrollResult.empreinte?.optimizedArray || null,
      hasEmpreinte: !!enrollResult.empreinte,
    };
    setDatabase(prev => [...prev, entry]);
    setEnrollName(""); setEnrollResult(null);
    setTab("base");
    alert(`✅ ${entry.name} enrôlé avec succès !`);
  };

  // ── Authentification ──────────────────────────────────────────────────────
  const runAuth = async () => {
    if (!authFileR || database.length === 0) return;
    setLoading(true); setMsg("Calcul du vecteur rétine..."); setAuthResult(null);
    try {
      const retineRes = await processBiometric(authFileR, "retine");
      let empreinteRes = null;
      if (authMode === "double" && authFileE) {
        setMsg("Calcul du vecteur empreinte...");
        await new Promise(r=>setTimeout(r,300));
        empreinteRes = await processBiometric(authFileE, "empreinte");
      }
      setMsg("Comparaison avec la base...");
      await new Promise(r=>setTimeout(r,300));

      // Comparer avec chaque entrée de la base
      function sameVector(v1, v2) {
        if (!v1 || !v2 || v1.length !== v2.length) return false;

        for (let i = 0; i < v1.length; i++) {
          if (Number(v1[i]).toFixed(4) !== Number(v2[i]).toFixed(4)) {
             return false;
          }
        }

        return true;
      }

      const results = database.map(entry => {
        const retineMatch = sameVector(
          retineRes.optimizedArray,
          entry.retineVector
        );

        let empreinteMatch = false;

        if (authMode === "double") {
          empreinteMatch = sameVector(
            empreinteRes?.optimizedArray,
            entry.empreinteVector
          );
        }

        const globalMatch = authMode === "double"
          ? retineMatch && empreinteMatch
          : retineMatch;

        return {
          ...entry,
          retineDist: retineMatch ? 0 : 1,
          empreinteDist: empreinteMatch ? 0 : 1,
          retineMatch,
          empreinteMatch,
          globalMatch,
          globalScore: globalMatch ? 0 : 1
        };
      });

      results.sort((a, b) => a.globalScore - b.globalScore);
      
      const bestMatch = results.find(r => r.globalMatch);

      if (bestMatch) {
        setAccessPopup({
        status: "autorise",
        name: bestMatch.name,
        });
      } else {
        setAccessPopup({
          status: "refuse",
          name: null,
        });
      }
      setAuthResult({ results, retineVec: retineRes.optimizedArray, empreinteVec: empreinteRes?.optimizedArray });
    } finally { setLoading(false); setMsg(""); }
  };

  return (
    <div>
      {accessPopup && (
        <div style={{
          position:"fixed",
          top:0,
          left:0,
          right:0,
          bottom:0,
          background:"rgba(0,0,0,0.45)",
          display:"flex",
          alignItems:"center",
          justifyContent:"center",
          zIndex:9999
        }}>
          <div style={{
            background:"#fff",
            borderRadius:18,
            padding:"32px",
            width:380,
            textAlign:"center",
            boxShadow:"0 20px 60px rgba(0,0,0,0.25)"
          }}>
           <div style={{ fontSize:46, marginBottom:12 }}>
            {accessPopup.status === "autorise" ? "✅" : "❌"}
            </div>

            <h2 style={{
              fontSize:24,
              fontWeight:900,
              color:accessPopup.status === "autorise" ? C.success : C.red,
              marginBottom:8
            }}>
              {accessPopup.status === "autorise" ? "Accès autorisé" : "Accès refusé"}
            </h2>

            {accessPopup.name && (
              <p style={{ fontSize:16, color:C.text, marginBottom:20 }}>
                Utilisateur reconnu : <strong>{accessPopup.name}</strong>
              </p>
            )}

            <button
              style={{ ...mkBtn("primary", accessPopup.status === "autorise" ? C.success : C.red) }}
              onClick={() => setAccessPopup(null)}
            >
              Fermer
            </button>
          </div>
        </div>
      )}
      {/* Tabs */}
      <div style={{ display:"flex", gap:8, marginBottom:24 }}>
        {[["base","🗄️ Base biométrique"],["enroll","➕ Enrôler"],["auth","🔍 Authentifier"]].map(([t,l])=>(
          <button key={t} style={{ ...mkBtn(tab===t?"primary":"ghost",accentColor), padding:"9px 18px" }} onClick={()=>setTab(t)}>{l}</button>
        ))}
        <div style={{ flex:1 }} />
        <span style={{ ...mkBadge(C.success), display:"flex", alignItems:"center" }}>{database.length} personne(s) enrôlée(s)</span>
      </div>

      {/* BASE */}
      {tab==="base" && (
        <>
          {database.length === 0 ? (
            <div style={{ ...base.card, textAlign:"center", padding:"48px" }}>
              <div style={{ fontSize:40, marginBottom:12 }}>🗄️</div>
              <div style={{ fontWeight:700, fontSize:16, marginBottom:8 }}>Base vide</div>
              <div style={{ color:C.sub, marginBottom:20 }}>Enrôlez des personnes autorisées pour commencer.</div>
              <button style={mkBtn("primary",accentColor)} onClick={()=>setTab("enroll")}>{Ic.plus}&nbsp;Enrôler une personne</button>
            </div>
          ) : (
            <div style={{ background:C.surface, border:`1px solid ${C.border}`, borderRadius:12, overflow:"hidden" }}>
              <table style={{ width:"100%", borderCollapse:"collapse" }}>
                <thead><tr>
                  <th style={base.th}>Nom</th>
                  <th style={base.th}>Date d'enrôlement</th>
                  <th style={base.th}>Rétine</th>
                  <th style={base.th}>Empreinte</th>
                  <th style={base.th}>Vecteur rétine optimisé</th>
                  <th style={base.th}>Action</th>
                </tr></thead>
                <tbody>{database.map((e,i)=>(
                  <tr key={e.id} style={{ background:i%2===0?C.surface:C.bg }}>
                    <td style={{ ...base.td, fontWeight:700 }}>{e.name}</td>
                    <td style={{ ...base.td, fontSize:12 }}>{e.date}</td>
                    <td style={base.td}><span style={mkChip(C.success)}>✓</span></td>
                    <td style={base.td}>{e.hasEmpreinte?<span style={mkChip(C.accent)}>✓</span>:<span style={mkChip(C.muted)}>—</span>}</td>
                    <td style={{ ...base.td, fontFamily:"monospace", fontSize:10, color:C.primary }}>
                      [{e.retineVector.map(v=>v.toFixed(3)).join(", ")}]
                    </td>
                    <td style={base.td}>
                      <button style={{ ...mkBtn("soft",C.red), padding:"5px 10px", fontSize:12 }}
                        onClick={()=>setDatabase(prev=>prev.filter(x=>x.id!==e.id))}>
                        Supprimer
                      </button>
                    </td>
                  </tr>
                ))}</tbody>
              </table>
            </div>
          )}
        </>
      )}

      {/* ENRÔLEMENT */}
      {tab==="enroll" && (
        <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:24 }}>
          <div style={base.card}>
            <div style={{ fontWeight:700, fontSize:15, marginBottom:16 }}>Informations de la personne</div>
            <label style={base.label}>Nom complet *</label>
            <input style={base.input} type="text" placeholder="Ex : Camille Martin"
              value={enrollName} onChange={e=>setEnrollName(e.target.value)} />

            <label style={base.label}>Image rétinienne *</label>
            <input ref={fileRefR} type="file" accept=".png,.jpg,.jpeg,.bmp" style={{ display:"none" }}
              onChange={async e=>{if(e.target.files[0]){setMsg("Traitement rétine...");await handleEnrollFile(e.target.files[0],"retine",setEnrollResult);setMsg("");}}} />
            <div onClick={()=>fileRefR.current.click()} style={{ border:`2px dashed ${enrollResult?.retine?C.success:C.border}`, borderRadius:10, padding:"20px", textAlign:"center", cursor:"pointer", background:enrollResult?.retine?C.successBg:C.bg, marginBottom:14 }}>
              {enrollResult?.retine ? <><div style={{ fontSize:20 }}>✅</div><div style={{ color:C.success, fontSize:13, fontWeight:700 }}>Rétine traitée</div></> : <><div style={{ fontSize:24 }}>👁️</div><div style={{ color:accentColor, fontSize:13 }}>Importer l'image rétinienne</div></>}
            </div>

            <label style={base.label}>Image empreinte (optionnel)</label>
            <input ref={fileRefE} type="file" accept=".png,.jpg,.jpeg,.bmp" style={{ display:"none" }}
              onChange={async e=>{if(e.target.files[0]){setMsg("Traitement empreinte...");await handleEnrollFile(e.target.files[0],"empreinte",setEnrollResult);setMsg("");}}} />
            <div onClick={()=>fileRefE.current.click()} style={{ border:`2px dashed ${enrollResult?.empreinte?C.success:C.border}`, borderRadius:10, padding:"16px", textAlign:"center", cursor:"pointer", background:enrollResult?.empreinte?C.successBg:C.bg, marginBottom:16 }}>
              {enrollResult?.empreinte ? <><div style={{ fontSize:18 }}>✅</div><div style={{ color:C.success, fontSize:13, fontWeight:700 }}>Empreinte traitée</div></> : <><div style={{ fontSize:22 }}>🫆</div><div style={{ color:C.muted, fontSize:13 }}>Importer l'empreinte (optionnel)</div></>}
            </div>

            <button style={{ ...mkBtn("primary",accentColor), width:"100%", padding:"13px", opacity:(!enrollName.trim()||!enrollResult?.retine)?0.5:1 }}
              disabled={!enrollName.trim()||!enrollResult?.retine} onClick={confirmEnroll}>
              {Ic.plus}&nbsp;Enrôler dans la base
            </button>
          </div>

          <div style={base.card}>
            <div style={{ fontWeight:700, fontSize:15, marginBottom:12 }}>Vecteurs générés</div>
            {enrollResult?.retine ? (
              <>
                <div style={{ fontWeight:600, fontSize:13, color:C.primary, marginBottom:6 }}>👁️ Vecteur optimisé rétine [5D]</div>
                <div style={{ background:"#080F1E", borderRadius:8, padding:"12px", fontFamily:"monospace", fontSize:11, color:"#4ADE80", marginBottom:14 }}>
                  {["OvLen","TI","MedTor","D1","D2"].map((k,i)=>(
                    <div key={k}><span style={{ color:"#94A3B8" }}>{k}: </span><span>{enrollResult.retine.optimizedArray[i]?.toFixed(6)}</span></div>
                  ))}
                </div>
              </>
            ) : (
              <div style={{ color:C.muted, fontSize:13, padding:"20px", textAlign:"center" }}>Les vecteurs apparaîtront ici après l'import des images.</div>
            )}
            {enrollResult?.empreinte && (
              <>
                <div style={{ fontWeight:600, fontSize:13, color:C.accent, marginBottom:6 }}>🫆 Vecteur optimisé empreinte [6D]</div>
                <div style={{ background:"#080F1E", borderRadius:8, padding:"12px", fontFamily:"monospace", fontSize:11, color:"#60A5FA", marginBottom:14 }}>
                  {["nbMinutiae","nbBifurcations","nbTerminations","minutiaeDensity","meanOrientation","orientationVariation"].map((k,i)=>(
                    <div key={k}><span style={{ color:"#94A3B8" }}>{k}: </span><span>{enrollResult.empreinte.optimizedArray[i]?.toFixed(6)}</span></div>
                  ))}
                </div>
              </>
            )}
            <div style={{ padding:"12px 14px", background:C.primaryLight, borderRadius:9, fontSize:12, color:C.primary, marginTop:8 }}>
              💡 Seuls les vecteurs optimisés sont stockés dans la base. Les images sont détruites.
            </div>
          </div>
        </div>
      )}

      {/* AUTHENTIFICATION */}
      {tab==="auth" && (
        <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:24 }}>
          <div style={base.card}>
            <div style={{ fontWeight:700, fontSize:15, marginBottom:16 }}>Authentification biométrique</div>

            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:10, marginBottom:16 }}>
              {[["retine","👁️ Rétine seule"],["double","👁️🫆 Rétine + Empreinte"]].map(([m,l])=>(
                <div key={m} onClick={()=>setAuthMode(m)} style={{ border:`2px solid ${authMode===m?accentColor:C.border}`, borderRadius:9, padding:"12px", cursor:"pointer", background:authMode===m?accentColor+"0E":C.bg, textAlign:"center" }}>
                  <div style={{ fontWeight:700, fontSize:13, color:authMode===m?accentColor:C.text }}>{l}</div>
                </div>
              ))}
            </div>

            <label style={base.label}>Image rétinienne *</label>
            <input ref={fileRefAR} type="file" accept=".png,.jpg,.jpeg,.bmp" style={{ display:"none" }}
              onChange={e=>setAuthFileR(e.target.files[0]||null)} />
            <div onClick={()=>fileRefAR.current.click()} style={{ border:`2px dashed ${authFileR?C.success:C.border}`, borderRadius:10, padding:"18px", textAlign:"center", cursor:"pointer", background:authFileR?C.successBg:C.bg, marginBottom:12 }}>
              {authFileR ? <><div style={{ fontSize:18 }}>✅</div><div style={{ color:C.success, fontSize:13 }}>{authFileR.name}</div></> : <><div style={{ fontSize:24 }}>👁️</div><div style={{ color:accentColor, fontSize:13 }}>Importer la rétine</div></>}
            </div>

            {authMode==="double" && (
              <>
                <label style={base.label}>Image empreinte *</label>
                <input ref={fileRefAE} type="file" accept=".png,.jpg,.jpeg,.bmp" style={{ display:"none" }}
                  onChange={e=>setAuthFileE(e.target.files[0]||null)} />
                <div onClick={()=>fileRefAE.current.click()} style={{ border:`2px dashed ${authFileE?C.success:C.border}`, borderRadius:10, padding:"18px", textAlign:"center", cursor:"pointer", background:authFileE?C.successBg:C.bg, marginBottom:12 }}>
                  {authFileE ? <><div style={{ fontSize:18 }}>✅</div><div style={{ color:C.success, fontSize:13 }}>{authFileE.name}</div></> : <><div style={{ fontSize:24 }}>🫆</div><div style={{ color:C.accent, fontSize:13 }}>Importer l'empreinte</div></>}
                </div>
              </>
            )}

            <div style={{ padding:"10px 12px", background:C.bg, borderRadius:8, border:`1px solid ${C.border}`, fontSize:12, color:C.muted, marginBottom:16 }}>
              Seuil rétine : d ≤ {THRESHOLD_RETINE} · Seuil empreinte : d ≤ {THRESHOLD_EMPREINTE}
            </div>

            <button style={{ ...mkBtn("primary",accentColor), width:"100%", padding:"13px", opacity:(!authFileR||loading)?0.5:1 }}
              disabled={!authFileR||loading} onClick={runAuth}>
              {loading?<><span style={{ animation:"spin 1s linear infinite", display:"inline-block" }}>⟳</span>&nbsp;{msg}</>:<>{Ic.search}&nbsp;Lancer l'authentification</>}
            </button>
          </div>

          {/* Résultats authentification */}
          <div style={base.card}>
            <div style={{ fontWeight:700, fontSize:15, marginBottom:12 }}>Résultats</div>
            {!authResult ? (
              <div style={{ color:C.muted, fontSize:13, textAlign:"center", padding:"40px 0" }}>
                <div style={{ fontSize:40, marginBottom:12 }}>🔍</div>
                Les résultats apparaîtront ici après l'authentification.
              </div>
            ) : (
              <>
                {authResult.results.map((r,i)=>{
                  const retinePct = Math.max(0, Math.min(100, (1-r.retineDist/0.5)*100));
                  return (
                    <div key={r.id} style={{ border:`2px solid ${r.globalMatch?C.success:i===0?C.warning:C.border}`, borderRadius:11, padding:"14px", marginBottom:12, background:r.globalMatch?C.successBg:C.surface }}>
                      <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:8 }}>
                        <div style={{ fontWeight:700, fontSize:15 }}>{r.name}</div>
                        <span style={mkChip(r.globalMatch?C.success:C.red)}>{r.globalMatch?"✓ Identifié":"✗ Non identifié"}</span>
                      </div>
                      <div style={{ fontSize:12, color:C.sub, marginBottom:8 }}>
                        👁️ Distance rétine : <strong style={{ color:r.retineMatch?C.success:C.red }}>{r.retineDist.toFixed(4)}</strong>
                        {r.empreinteDist !== null && <> · 🫆 Distance empreinte : <strong style={{ color:r.empreinteMatch?C.success:C.red }}>{r.empreinteDist.toFixed(4)}</strong></>}
                      </div>
                      <div style={{ height:6, background:C.border, borderRadius:4, overflow:"hidden", position:"relative" }}>
                        <div style={{ position:"absolute", top:0, left:0, height:"100%", width:`${retinePct}%`, background:r.retineMatch?C.success:C.red, borderRadius:4 }}/>
                      </div>
                      <div style={{ fontSize:11, color:C.muted, marginTop:4 }}>Score similarité : {retinePct.toFixed(1)}%</div>
                    </div>
                  );
                })}
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}


// ═══════════════════════════════════════════════════════════════════════════════
// CRÉATION DE COMPTE UTILISATEUR (par l'administrateur)
// ═══════════════════════════════════════════════════════════════════════════════
function CreateUserPanel({ users, onCreateUser, database, setDatabase, accentColor=C.primary }) {
  const [form, setForm]       = useState({ prenom:"", nom:"", username:"", password:"", email:"" });
  const [bio, setBio]         = useState(null);   // { retine, empreinte }
  const [busy, setBusy]       = useState("");      // message de traitement en cours
  const [err, setErr]         = useState("");
  const [created, setCreated] = useState(null);    // récap du compte créé
  const refRetine = useRef(); const refEmpreinte = useRef();
  const set = (k,v) => setForm(p=>({...p,[k]:v}));

  const handleFile = async (file, mode) => {
    if (!file) return;
    setErr(""); setBusy(mode==="retine" ? "Analyse rétinienne en cours..." : "Analyse empreinte en cours...");
    try {
      const result = await processBiometric(file, mode);
      setBio(prev => ({ ...(prev||{}), [mode]: result }));
    } catch(e) { setErr(`Erreur lors de l'analyse ${mode} : ${e.message}`); }
    finally { setBusy(""); }
  };

  const validate = () => {
    if (!form.prenom.trim()) return "Le prénom est requis.";
    if (!form.nom.trim()) return "Le nom est requis.";
    if (form.username.trim().length < 4) return "Identifiant trop court (min. 4 caractères).";
    if (users[form.username.trim()]) return "Cet identifiant existe déjà.";
    if (form.password.length < 6) return "Mot de passe trop court (min. 6 caractères).";
    if (form.email && !form.email.includes("@")) return "Email invalide.";
    if (!bio?.retine) return "Une analyse rétinienne est obligatoire (importez l'image de rétine).";
    return null;
  };

  const submit = () => {
    const e = validate();
    if (e) { setErr(e); return; }
    setErr("");
    const fullName = `${form.prenom.trim()} ${form.nom.trim()}`;
    const username = form.username.trim();

    // 1. Compte de connexion (avec vecteurs biométriques attachés pour l'auth au login)
    onCreateUser({
      username,
      password: form.password,
      role: "client",
      name: fullName,
      email: form.email.trim(),
      validated: true,
      retineVector: bio.retine.optimizedArray,
      empreinteVector: bio.empreinte?.optimizedArray || null,
      hasEmpreinte: !!bio.empreinte,
    });

    // 2. Enrôlement biométrique (rétine obligatoire + empreinte optionnelle)
    setDatabase(prev => [...prev, {
      id: Date.now().toString(),
      name: fullName,
      date: new Date().toLocaleString("fr-FR").slice(0,16),
      retineVector: bio.retine.optimizedArray,
      empreinteVector: bio.empreinte?.optimizedArray || null,
      hasEmpreinte: !!bio.empreinte,
    }]);

    setCreated({ username, fullName, hasEmpreinte: !!bio.empreinte });
    setForm({ prenom:"", nom:"", username:"", password:"", email:"" });
    setBio(null);
  };

  // ── Écran de confirmation ───────────────────────────────────────────────────
  if (created) return (
    <div style={{ ...base.card, maxWidth:560, margin:"0 auto", textAlign:"center", padding:"40px 36px" }}>
      <div style={{ width:64, height:64, background:C.successBg, borderRadius:"50%", display:"flex", alignItems:"center", justifyContent:"center", margin:"0 auto 16px", fontSize:32 }}>✅</div>
      <h2 style={{ fontSize:20, fontWeight:800, marginBottom:8 }}>Compte utilisateur créé</h2>
      <p style={{ color:C.sub, fontSize:14, marginBottom:20, lineHeight:1.6 }}>
        <strong>{created.fullName}</strong> peut désormais se connecter avec l'identifiant <code style={{ color:accentColor }}>{created.username}</code>.
      </p>
      <div style={{ background:C.bg, border:`1px solid ${C.border}`, borderRadius:10, padding:"14px 16px", fontSize:13, color:C.sub, marginBottom:24, textAlign:"left" }}>
        👁️ Rétine enrôlée dans la base biométrique{created.hasEmpreinte ? " · 🫆 Empreinte enrôlée" : ""}.
      </div>
      <button style={{ ...mkBtn("primary",accentColor), padding:"12px 22px" }} onClick={()=>setCreated(null)}>{Ic.plus}&nbsp;Créer un autre compte</button>
    </div>
  );

  const DropZone = ({ done, label, icon, inputRef, mode, optional }) => (
    <>
      <input ref={inputRef} type="file" accept=".png,.jpg,.jpeg,.bmp" style={{ display:"none" }}
        onChange={e=>handleFile(e.target.files[0], mode)} />
      <div onClick={()=>inputRef.current.click()}
        style={{ border:`2px dashed ${done?C.success:C.border}`, borderRadius:10, padding:"20px", textAlign:"center", cursor:"pointer", background:done?C.successBg:C.bg, marginBottom:14, transition:"all 0.15s" }}>
        {done
          ? <><div style={{ fontSize:22 }}>✅</div><div style={{ color:C.success, fontSize:13, fontWeight:700 }}>{mode==="retine"?"Rétine analysée":"Empreinte analysée"}</div></>
          : <><div style={{ fontSize:26 }}>{icon}</div><div style={{ color:optional?C.muted:accentColor, fontSize:13 }}>{label}</div><div style={{ color:C.muted, fontSize:11 }}>PNG, JPG, BMP</div></>
        }
      </div>
    </>
  );

  return (
    <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:24 }}>
      {/* Infos utilisateur */}
      <div style={base.card}>
        <div style={{ fontWeight:700, fontSize:15, marginBottom:4 }}>Informations de l'utilisateur</div>
        <div style={{ color:C.sub, fontSize:13, marginBottom:16 }}>Le compte est créé par l'administrateur.</div>

        <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:12 }}>
          <div>
            <label style={base.label}>Prénom *</label>
            <input style={base.input} type="text" placeholder="Ex : Jean" value={form.prenom} onChange={e=>set("prenom",e.target.value)} />
          </div>
          <div>
            <label style={base.label}>Nom *</label>
            <input style={base.input} type="text" placeholder="Ex : Martin" value={form.nom} onChange={e=>set("nom",e.target.value)} />
          </div>
        </div>
        <label style={base.label}>Identifiant *</label>
        <input style={base.input} type="text" placeholder="Min. 4 caractères" value={form.username} onChange={e=>set("username",e.target.value)} />
        <label style={base.label}>Mot de passe *</label>
        <input style={base.input} type="text" placeholder="Min. 6 caractères" value={form.password} onChange={e=>set("password",e.target.value)} />
        <label style={base.label}>Email (optionnel)</label>
        <input style={base.input} type="email" placeholder="email@exemple.fr" value={form.email} onChange={e=>set("email",e.target.value)} />

        {err && <div style={{ background:C.redBg, color:C.red, border:`1px solid ${C.red}30`, borderRadius:8, padding:"10px 14px", marginBottom:14, fontSize:13 }}>⚠ {err}</div>}

        <button style={{ ...mkBtn("primary",accentColor), width:"100%", padding:"13px", fontSize:15, opacity:busy?0.6:1 }}
          onClick={submit} disabled={!!busy}>
          {Ic.plus}&nbsp;Créer le compte utilisateur
        </button>
      </div>

      {/* Analyse rétinienne */}
      <div style={base.card}>
        <div style={{ fontWeight:700, fontSize:15, marginBottom:4 }}>Analyse rétinienne *</div>
        <div style={{ color:C.sub, fontSize:13, marginBottom:16 }}>Importez l'image de rétine de l'utilisateur — l'analyse se lance automatiquement.</div>

        <label style={base.label}>Image rétinienne *</label>
        <DropZone done={!!bio?.retine} label="Importer l'image de rétine" icon="👁️" inputRef={refRetine} mode="retine" />

        <label style={base.label}>Image empreinte (optionnel)</label>
        <DropZone done={!!bio?.empreinte} label="Importer l'empreinte (optionnel)" icon="🫆" inputRef={refEmpreinte} mode="empreinte" optional />

        {busy && (
          <div style={{ display:"flex", alignItems:"center", gap:8, padding:"10px 12px", background:accentColor+"0E", borderRadius:8, fontSize:13, color:accentColor, marginBottom:12 }}>
            <span style={{ animation:"spin 1s linear infinite", display:"inline-block" }}>⟳</span> {busy}
          </div>
        )}

        {bio?.retine && (
          <div style={{ marginBottom:12 }}>
            <div style={{ fontWeight:600, fontSize:13, color:accentColor, marginBottom:6 }}>👁️ Vecteur optimisé rétine [5D]</div>
            <div style={{ background:"#080F1E", borderRadius:8, padding:"12px", fontFamily:"monospace", fontSize:11, color:"#4ADE80" }}>
              [{bio.retine.optimizedArray.map(v=>v.toFixed(4)).join(", ")}]
            </div>
          </div>
        )}

        <div style={{ padding:"10px 12px", background:C.bg, borderRadius:8, border:`1px solid ${C.border}`, fontSize:12, color:C.muted }}>
          🔒 Images non stockées — seuls les vecteurs optimisés sont conservés.
        </div>
        <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// AdministrateurAPP
// ═══════════════════════════════════════════════════════════════════════════════
function AdministrateurApp({ user, users, onCreateUser, onLogout }) {
  const [page, setPage]         = useState("dashboard");
  const [result, setResult]     = useState(null);
  const [database, setDatabase] = useState([
  {
    id: "camille",
    name: "Camille",
    date: "18/06/2026",
    retineVector: [5937.0000, 1.5451, 1.2127, 2.8667, 3.1170],
    empreinteVector: [41501.0000, 41498.0000, 3.0000, 953.3447, -0.0086, 1.9102],
    hasEmpreinte: true,
  },
  {
    id: "steven",
    name: "Steven",
    date: "19/06/2026",
    retineVector: [6273.0000, 1.5488, 1.2000, 1.9130, 1.5993],
    empreinteVector: [66111.0000, 66109.0000, 2.0000, 991.4667, -1.5095, 2.5649],
    hasEmpreinte: true,
  },
  {
    id: "tidar",
    name: "Tidar",
    date: "18/06/2026",
    retineVector: [6413.0000, 1.6210, 1.2000, 2.9683, 3.0340],
    empreinteVector: [60293.0000, 60291.0000, 2.0000, 999.1383, 0.4680, 1.8129],
    hasEmpreinte: true,
  },
  {
    id: "shanice",
    name: "Shanice",
    date: "18/06/2026",
    retineVector: [6154.0000, 1.4684, 1.1770, 2.2308, 1.8462],
    empreinteVector: [73165.0000, 73165.0000, 0.0000, 999.5492, 1.4086, 2.0869],
    hasEmpreinte: true,
  },
  {
    id: "aminata",
    name: "Aminata",
    date: "18/06/2026",
    retineVector: [4941.0000, 1.5814, 1.1142, 2.1613, 1.8156],
    empreinteVector: [69246.0000, 69244.0000, 2.0000, 999.7257, 1.2470, 1.9928],
    hasEmpreinte: true,
  },
]);
  const [history, setHistory]   = useState([]);

  const onResult = (res) => {
    setResult(res);
    setHistory(prev=>[{
      id:`B-${Date.now().toString().slice(-4)}`,
      date: res.date,
      Utilisateur: res.UtilisateurId,
      mode: res.securityMode==="double"?"Rétine + Empreinte":"Rétine",
      action:"Analyse",
      result:"✓ Extrait"
    }, ...prev]);
    setPage("resultats");
  };

  const NAV = [
    { id:"dashboard",  label:"Tableau de bord",    icon:Ic.grid },
    { id:"utilisateurs", label:"Créer un utilisateur", icon:Ic.user },
    { id:"analyse",    label:"Nouvelle analyse",    icon:Ic.scan },
    { id:"resultats",  label:"Résultats",           icon:Ic.chart },
    { id:"biometrie",  label:"Base biométrique",    icon:Ic.db },
    { id:"historique", label:"Historique",          icon:Ic.clock },
    { id:"securite",   label:"Sécurité",            icon:Ic.shield },
  ];

  return (
    <Shell user={user} page={page} setPage={setPage} navItems={NAV} onLogout={onLogout}
      sidebarColor={C.sidebar} activeColor={C.primary} topRight={<span style={mkBadge(C.primary)}>Administrateur</span>}>

      {/* DASHBOARD */}
      {page==="dashboard" && (
        <>
          <h1 style={{ fontSize:22, fontWeight:800, marginBottom:4 }}>Bienvenue, {user.name}</h1>
          <p style={{ color:C.sub, marginBottom:24 }}>Système d'identification biométrique · SegVision</p>
          <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:16, marginBottom:24 }}>
            {[
              {v:String(history.length), l:"Analyses"},
              {v:String(database.length), l:"Personnes enrôlées"},
              {v:"2", l:"Types biométriques"},
              {v:"5+6", l:"Features optimisées"},
            ].map(s=>(
              <div key={s.l} style={base.card}>
                <div style={{ width:36,height:3,background:C.primary,borderRadius:2,marginBottom:12 }}/>
                <div style={{ fontSize:26,fontWeight:800,marginBottom:4 }}>{s.v}</div>
                <div style={{ fontSize:13,color:C.sub }}>{s.l}</div>
              </div>
            ))}
          </div>

          {/* Pipeline */}
          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:20, marginBottom:20 }}>
            <div style={base.card}>
              <div style={{ fontWeight:700, fontSize:15, marginBottom:12 }}>Pipeline biométrique rétine</div>
              {["Segmentation vaisseaux (DoG multi-échelle)","Squelettisation Zhang-Suen (1px)","Extraction OvLen, TI, MedTor, D1, D2","Vecteur optimisé 5D enregistré"].map((s,i)=>(
                <div key={i} style={{ display:"flex",gap:10,alignItems:"center",padding:"8px 0",borderBottom:`1px solid ${C.border}` }}>
                  <div style={{ width:20,height:20,borderRadius:"50%",background:C.primaryLight,color:C.primary,fontWeight:800,fontSize:11,display:"flex",alignItems:"center",justifyContent:"center",flexShrink:0 }}>{i+1}</div>
                  <div style={{ fontSize:13,color:C.text }}>{s}</div>
                </div>
              ))}
            </div>
            <div style={base.card}>
              <div style={{ fontWeight:700, fontSize:15, marginBottom:12 }}>Pipeline biométrique empreinte</div>
              {["Segmentation crêtes digitales","Détection minuties (bifurcations + terminaisons)","Calcul orientation et variation","Vecteur optimisé 6D enregistré"].map((s,i)=>(
                <div key={i} style={{ display:"flex",gap:10,alignItems:"center",padding:"8px 0",borderBottom:`1px solid ${C.border}` }}>
                  <div style={{ width:20,height:20,borderRadius:"50%",background:C.accentLight,color:C.accent,fontWeight:800,fontSize:11,display:"flex",alignItems:"center",justifyContent:"center",flexShrink:0 }}>{i+1}</div>
                  <div style={{ fontSize:13,color:C.text }}>{s}</div>
                </div>
              ))}
            </div>
          </div>

          <div style={{ display:"flex", gap:12, flexWrap:"wrap" }}>
            <button style={mkBtn("primary",C.primary)} onClick={()=>setPage("utilisateurs")}>{Ic.user}&nbsp;Créer un utilisateur</button>
            <button style={mkBtn("soft",C.primary)} onClick={()=>setPage("analyse")}>{Ic.scan}&nbsp;Nouvelle analyse</button>
            <button style={mkBtn("soft",C.success)} onClick={()=>setPage("biometrie")}>{Ic.db}&nbsp;Base biométrique</button>
            <button style={mkBtn("soft",C.primary)} onClick={()=>setPage("biometrie")}>{Ic.search}&nbsp;Authentifier</button>
          </div>
        </>
      )}

      {/* ANALYSE */}
      {page==="analyse" && <UploadPanel onResult={onResult} accentColor={C.primary} showId={true} />}

      {/* RÉSULTATS */}
      {page==="resultats" && (
        <ResultsPanel result={result} accentColor={C.primary}
          onNew={()=>setPage("analyse")}
          onEnroll={r=>{
            // Pré-remplir l'enrôlement avec le résultat courant
            setPage("biometrie");
          }}
          onAuth={()=>setPage("biometrie")}
        />
      )}

      {/* CRÉATION UTILISATEUR */}
      {page==="utilisateurs" && (
        <CreateUserPanel users={users} onCreateUser={onCreateUser} database={database} setDatabase={setDatabase} accentColor={C.primary} />
      )}

      {/* BASE BIOMÉTRIQUE */}
      {page==="biometrie" && (
        <BiometricDB database={database} setDatabase={setDatabase} accentColor={C.primary} />
      )}

      {/* HISTORIQUE */}
      {page==="historique" && (
        <>
          <h2 style={{ fontSize:18,fontWeight:800,marginBottom:20 }}>Historique des opérations</h2>
          {history.length===0
            ? <div style={{ ...base.card, textAlign:"center", padding:"40px", color:C.muted }}>
                <div style={{ fontSize:36, marginBottom:12 }}>📋</div>
                <div>Aucune opération effectuée.</div>
              </div>
            : <div style={{ background:C.surface,border:`1px solid ${C.border}`,borderRadius:12,overflow:"hidden" }}>
                <table style={{ width:"100%",borderCollapse:"collapse" }}>
                  <thead><tr><th style={base.th}>ID</th><th style={base.th}>Date</th><th style={base.th}>Utilisateur</th><th style={base.th}>Mode</th><th style={base.th}>Action</th><th style={base.th}>Résultat</th></tr></thead>
                  <tbody>{history.map((r,i)=>(
                    <tr key={r.id} style={{ background:i%2===0?C.surface:C.bg }}>
                      <td style={base.td}><code style={{ fontSize:12,color:C.primary }}>{r.id}</code></td>
                      <td style={{ ...base.td,fontSize:12 }}>{r.date}</td>
                      <td style={base.td}>{r.Utilisateur}</td>
                      <td style={base.td}>{r.mode}</td>
                      <td style={base.td}>{r.action}</td>
                      <td style={base.td}><span style={mkChip(C.success)}>{r.result}</span></td>
                    </tr>
                  ))}</tbody>
                </table>
              </div>
          }
        </>
      )}

      {/* SÉCURITÉ */}
      {page==="securite" && (
        <>
          <h2 style={{ fontSize:18,fontWeight:800,marginBottom:4 }}>Sécurité & Conformité</h2>
          <p style={{ color:C.sub,fontSize:13,marginBottom:20 }}>Architecture du système biométrique.</p>
          <div style={{ display:"grid",gridTemplateColumns:"1fr 1fr",gap:16 }}>
            {[
              {icon:"🗑️",t:"Pas de stockage d'image",d:"Seuls les vecteurs optimisés sont conservés. L'image est détruite après traitement."},
              {icon:"📐",t:"Squelettisation Zhang-Suen",d:"Les vaisseaux sont réduits à 1px d'épaisseur pour une mesure précise de OvLen, TI, MedTor."},
              {icon:"⚡",t:"Features optimisées",d:"Rétine : OvLen, TI, MedTor, D1, D2 · Empreinte : minuties, orientation, variation."},
              {icon:"🔐",t:"Distance euclidienne normalisée",d:"Seuil rétine : 0.05 · Seuil empreinte : 0.05 · Mode double = les deux doivent correspondre."},
              {icon:"🏥",t:"Validation RPPS",d:"Comptes Administrateur vérifiés via RPPS et Ordre des Administrateurs."},
              {icon:"📋",t:"Audit trail",d:"Chaque analyse, enrôlement et authentification est tracé avec timestamp."},
            ].map(item=>(
              <div key={item.t} style={{ ...base.card,display:"flex",gap:14 }}>
                <div style={{ fontSize:26 }}>{item.icon}</div>
                <div><div style={{ display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:6 }}><div style={{ fontWeight:700,fontSize:14 }}>{item.t}</div><span style={mkChip(C.success)}>Actif</span></div><div style={{ color:C.sub,fontSize:13 }}>{item.d}</div></div>
              </div>
            ))}
          </div>
        </>
      )}
    </Shell>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Utilisateur APP
// ═══════════════════════════════════════════════════════════════════════════════
function UtilisateurApp({ user, onLogout }) {
  const [page,    setPage]    = useState("accueil");
  const [result,  setResult]  = useState(null);
  const [history, setHistory] = useState([]);

  const onResult = (res) => {
    setResult(res);
    setHistory(prev=>[{
      id:`P-${Date.now().toString().slice(-4)}`,
      date: res.date,
      mode: res.securityMode==="double"?"👁️🫆 Rétine + Empreinte":"👁️ Rétine",
      retineDims: 5,
      empreinteDims: res.empreinte ? 6 : null,
    }, ...prev]);
    setPage("resultats");
  };

  const NAV = [
    { id:"accueil",    label:"Accueil",         icon:Ic.grid },
    { id:"analyse",    label:"Mon analyse",      icon:Ic.scan },
    { id:"resultats",  label:"Mes résultats",    icon:Ic.chart },
    { id:"historique", label:"Mon historique",   icon:Ic.clock },
    { id:"infos",      label:"Comment ça marche",icon:Ic.eye },
  ];

  return (
    <Shell user={user} page={page} setPage={setPage} navItems={NAV} onLogout={onLogout}
      sidebarColor={C.clientSidebar} activeColor={C.clientActive} topRight={<span style={mkBadge(C.clientActive)}>Utilisateur</span>}>

      {/* ACCUEIL */}
      {page==="accueil" && (
        <>
          <div style={{ background:`linear-gradient(135deg,${C.clientSidebar} 0%,#3B0D8C 100%)`,borderRadius:16,padding:"36px 40px",marginBottom:24,color:"#fff" }}>
            <div style={{ fontSize:12,color:"#C4B5FD",fontWeight:700,marginBottom:8,textTransform:"uppercase",letterSpacing:"0.08em" }}>SegVision · Biométrie</div>
            <h1 style={{ fontSize:24,fontWeight:900,marginBottom:10 }}>Bonjour, {user.name} 👋</h1>
            <p style={{ color:"#DDD6FE",fontSize:14,maxWidth:500,lineHeight:1.6,marginBottom:20 }}>
              Importez votre image rétinienne (et optionnellement votre empreinte) pour générer vos vecteurs biométriques optimisés.
            </p>
            <button style={{ ...mkBtn("primary","#fff"),color:C.clientSidebar,padding:"11px 22px",fontSize:14,fontWeight:700 }} onClick={()=>setPage("analyse")}>
              {Ic.scan}&nbsp;Lancer mon analyse
            </button>
          </div>

          <div style={{ display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:12,marginBottom:20 }}>
            {[["👁️","Rétine","Vaisseaux segmentés"],["📐","Squelette","1px d'épaisseur"],["🧬","5 features","OvLen,TI,MedTor,D1,D2"],["🔐","Vecteur","Non réversible"]].map(([icon,t,d])=>(
              <div key={t} style={base.card}><div style={{ fontSize:24,marginBottom:8 }}>{icon}</div><div style={{ fontWeight:700,fontSize:13,marginBottom:3 }}>{t}</div><div style={{ color:C.muted,fontSize:11 }}>{d}</div></div>
            ))}
          </div>

          {history.length>0 && (
            <>
              <div style={{ fontWeight:700,fontSize:15,marginBottom:12 }}>Ma dernière analyse</div>
              <div style={{ ...base.card,display:"flex",alignItems:"center",justifyContent:"space-between" }}>
                <div><div style={{ fontWeight:700,fontSize:15,marginBottom:4 }}>{history[0].mode}</div><div style={{ color:C.sub,fontSize:13 }}>{history[0].date} · {history[0].retineDims}D rétine{history[0].empreinteDims?` + ${history[0].empreinteDims}D empreinte`:""}</div></div>
                <button style={mkBtn("primary",C.clientActive)} onClick={()=>setPage("resultats")}>{Ic.eye}&nbsp;Voir</button>
              </div>
            </>
          )}
        </>
      )}

      {/* ANALYSE */}
      {page==="analyse" && <UploadPanel onResult={onResult} accentColor={C.clientActive} showId={true} defaultId={user.name} />}

      {/* RÉSULTATS */}
      {page==="resultats" && <ResultsPanel result={result} accentColor={C.clientActive} onNew={()=>setPage("analyse")} />}

      {/* HISTORIQUE */}
      {page==="historique" && (
        <>
          <h2 style={{ fontSize:18,fontWeight:800,marginBottom:4 }}>Mon historique</h2>
          <p style={{ color:C.sub,fontSize:13,marginBottom:16 }}>{history.length} analyse(s) cette session</p>
          {history.length===0
            ? <div style={{ ...base.card,textAlign:"center",padding:"48px",color:C.muted }}><div style={{ fontSize:40,marginBottom:12 }}>📂</div><div style={{ marginBottom:16 }}>Aucune analyse effectuée.</div><button style={mkBtn("primary",C.clientActive)} onClick={()=>setPage("analyse")}>{Ic.scan}&nbsp;Analyser</button></div>
            : <div style={{ background:C.surface,border:`1px solid ${C.border}`,borderRadius:12,overflow:"hidden" }}>
                <table style={{ width:"100%",borderCollapse:"collapse" }}>
                  <thead><tr><th style={base.th}>ID</th><th style={base.th}>Date</th><th style={base.th}>Mode</th><th style={base.th}>Dimensions</th></tr></thead>
                  <tbody>{history.map((r,i)=>(
                    <tr key={r.id} style={{ background:i%2===0?C.surface:C.bg }}>
                      <td style={base.td}><code style={{ fontSize:12,color:C.clientActive }}>{r.id}</code></td>
                      <td style={{ ...base.td,fontSize:12 }}>{r.date}</td>
                      <td style={base.td}>{r.mode}</td>
                      <td style={base.td}>{r.retineDims}D rétine{r.empreinteDims?` + ${r.empreinteDims}D empreinte`:""}</td>
                    </tr>
                  ))}</tbody>
                </table>
              </div>
          }
        </>
      )}

      {/* INFOS */}
      {page==="infos" && (
        <>
          <h2 style={{ fontSize:18,fontWeight:800,marginBottom:4 }}>Comment ça marche ?</h2>
          <p style={{ color:C.sub,fontSize:13,marginBottom:20 }}>Le pipeline biométrique SegVision étape par étape.</p>
          <div style={{ display:"grid",gridTemplateColumns:"1fr 1fr",gap:16 }}>
            {[
              {icon:"👁️",t:"Image rétinienne",d:"Vous importez une image de votre rétine (fond d'œil). L'image est redimensionnée à 512×512 pixels."},
              {icon:"🩸",t:"Segmentation vaisseaux",d:"L'algorithme détecte les vaisseaux sanguins rétiniens par différence de Gaussiennes multi-échelle."},
              {icon:"📐",t:"Squelettisation",d:"Les vaisseaux sont réduits à 1 pixel d'épaisseur (algorithme Zhang-Suen) pour mesurer précisément leur géométrie."},
              {icon:"🧬",t:"Features rétine",d:"OvLen (longueur vasculaire), TI (tortuosité moyenne), MedTor (tortuosité médiane), D1 (diamètre moyen), D2 (variation diamètre)."},
              {icon:"🫆",t:"Features empreinte (optionnel)",d:"Nombre de minuties, bifurcations, terminaisons, densité, orientation moyenne et variation des orientations."},
              {icon:"🔐",t:"Vecteur optimisé",d:"Seules les features les plus discriminantes sont conservées (5D rétine + 6D empreinte). L'image est détruite."},
            ].map(item=>(
              <div key={item.t} style={base.card}><div style={{ fontSize:28,marginBottom:10 }}>{item.icon}</div><div style={{ fontWeight:700,fontSize:14,marginBottom:8 }}>{item.t}</div><div style={{ color:C.sub,fontSize:13,lineHeight:1.6 }}>{item.d}</div></div>
            ))}
          </div>
        </>
      )}
    </Shell>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// ROOT
// ═══════════════════════════════════════════════════════════════════════════════
export default function App() {
  const [user,  setUser]  = useState(null);
  const [users, setUsers] = useState({ ...INITIAL_USERS });
  const register = (u) => setUsers(prev=>({...prev,[u.username]:u}));

  if (!user) return <LoginPage onLogin={setUser} users={users} onRegister={register} />;
  if (user.role==="Administrateur") return <AdministrateurApp user={user} users={users} onCreateUser={register} onLogout={()=>setUser(null)} />;
  return <UtilisateurApp user={user} onLogout={()=>setUser(null)} />;
}
