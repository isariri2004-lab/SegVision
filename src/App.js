import React, { useState, useRef, useCallback, useEffect } from "react";

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
  return Math.sqrt(a.reduce((sum, value, index) => sum + (value - b[index]) ** 2, 0));
}

function cosineSimilarity(a, b) {
  if (!a || !b || a.length !== b.length) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) || 1);
}

function normalizeVector(vector) {
  const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0)) || 1;
  return vector.map(value => value / norm);
}

function validVector(vector, expectedLength) {
  return Array.isArray(vector) &&
    vector.length === expectedLength &&
    vector.every(value => Number.isFinite(Number(value)));
}

// Compare deux signatures rétiniennes avec des tolérances propres à chaque feature.
// La même image donne 100 %. Deux acquisitions très proches du même œil peuvent
// être acceptées sans exiger une égalité numérique impossible au pixel près.
function compareRetinaVectors(a, b) {
  if (!validVector(a, 5) || !validVector(b, 5)) {
    return { match:false, similarity:0, distance:Infinity, within:0, deltas:[] };
  }

  const av = a.map(Number);
  const bv = b.map(Number);
  const average = (x, y, floor=1e-6) => Math.max((Math.abs(x) + Math.abs(y)) / 2, floor);

  const deltas = [
    Math.abs(av[0] - bv[0]) / average(av[0], bv[0], 1) / 0.04, // OvLen : ±4 %
    Math.abs(av[1] - bv[1]) / 0.03,                             // TI
    Math.abs(av[2] - bv[2]) / 0.03,                             // MedTor
    Math.abs(av[3] - bv[3]) / average(av[3], bv[3], 0.5) / 0.10, // D1 : ±10 %
    Math.abs(av[4] - bv[4]) / average(av[4], bv[4], 0.5) / 0.12, // D2 : ±12 %
  ];

  const weights = [0.35, 0.20, 0.15, 0.15, 0.15];
  const distance = deltas.reduce(
    (sum, delta, index) => sum + Math.min(delta, 4) * weights[index],
    0
  );
  const within = deltas.filter(delta => delta <= 1).length;
  const similarity = Math.max(0, Math.min(100, 100 * Math.exp(-0.8 * distance)));
  const exact = av.every((value, index) => value.toFixed(4) === bv[index].toFixed(4));
  const match = exact || (within >= 4 && distance <= 0.85 && similarity >= 50);

  return { match, similarity, distance, within, deltas, exact };
}

function compareFingerprintVectors(a, b) {
  if (!validVector(a, 6) || !validVector(b, 6)) {
    return { match:false, similarity:0, distance:Infinity, within:0, deltas:[] };
  }

  const av = a.map(Number);
  const bv = b.map(Number);
  const average = (x, y, floor=1e-6) => Math.max((Math.abs(x) + Math.abs(y)) / 2, floor);

  const deltas = [
    Math.abs(av[0] - bv[0]) / average(av[0], bv[0], 10) / 0.08,
    Math.abs(av[1] - bv[1]) / average(av[1], bv[1], 10) / 0.10,
    Math.abs(av[2] - bv[2]) / average(av[2], bv[2], 3) / 0.15,
    Math.abs(av[3] - bv[3]) / average(av[3], bv[3], 1) / 0.08,
    Math.abs(av[4] - bv[4]) / 0.15,
    Math.abs(av[5] - bv[5]) / 0.15,
  ];

  const weights = [0.25, 0.18, 0.12, 0.18, 0.14, 0.13];
  const distance = deltas.reduce(
    (sum, delta, index) => sum + Math.min(delta, 4) * weights[index],
    0
  );
  const within = deltas.filter(delta => delta <= 1).length;
  const similarity = Math.max(0, Math.min(100, 100 * Math.exp(-0.8 * distance)));
  const exact = av.every((value, index) => value.toFixed(4) === bv[index].toFixed(4));
  const match = exact || (within >= 5 && distance <= 0.85 && similarity >= 50);

  return { match, similarity, distance, within, deltas, exact };
}

function vectorsMatch(a, b) {
  return compareRetinaVectors(a, b).match;
}

// ─── Segmentation biométrique ────────────────────────────────────────────────
async function processBiometric(file, mode) {
return new Promise((resolve, reject) => {
const img = new Image();
const url = URL.createObjectURL(file);


img.onload = () => {
  try {
    const W = 512;
    const H = 512;
    const N = W * H;

    const origCanvas = document.createElement("canvas");
    origCanvas.width = W;
    origCanvas.height = H;
    const oCtx = origCanvas.getContext("2d", { willReadFrequently: true });
    if (!oCtx) throw new Error("Canvas 2D indisponible.");

    // Conserve les proportions de l'image au lieu de l'étirer.
    oCtx.fillStyle = "#000";
    oCtx.fillRect(0, 0, W, H);
    oCtx.imageSmoothingEnabled = true;
    oCtx.imageSmoothingQuality = "high";

    const scale = Math.min(W / img.width, H / img.height);
    const drawW = Math.max(1, Math.round(img.width * scale));
    const drawH = Math.max(1, Math.round(img.height * scale));
    const drawX = Math.floor((W - drawW) / 2);
    const drawY = Math.floor((H - drawH) / 2);

    oCtx.drawImage(img, drawX, drawY, drawW, drawH);
    URL.revokeObjectURL(url);

    const raw = oCtx.getImageData(0, 0, W, H).data;

    const countMask = (mask) => {
      let count = 0;
      for (let i = 0; i < mask.length; i++) {
        count += mask[i] ? 1 : 0;
      }
      return count;
    };

    const percentile = (values, q) => {
      if (!values.length) return 0;

      const sorted = Array.from(values).sort((a, b) => a - b);
      const index = Math.min(
        sorted.length - 1,
        Math.max(0, Math.floor((sorted.length - 1) * q))
      );

      return sorted[index];
    };

    function gauss(src, sigma) {
      const radius = Math.max(1, Math.ceil(sigma * 3));
      const size = radius * 2 + 1;
      const kernel = new Float32Array(size);
      let kernelSum = 0;

      for (let i = 0; i < size; i++) {
        const x = i - radius;
        const value = Math.exp(-(x * x) / (2 * sigma * sigma));
        kernel[i] = value;
        kernelSum += value;
      }

      for (let i = 0; i < size; i++) {
        kernel[i] /= kernelSum;
      }

      const tmp = new Float32Array(N);
      const out = new Float32Array(N);

      for (let y = 0; y < H; y++) {
        const row = y * W;

        for (let x = 0; x < W; x++) {
          let sum = 0;

          for (let k = -radius; k <= radius; k++) {
            const xx = Math.min(W - 1, Math.max(0, x + k));
            sum += src[row + xx] * kernel[k + radius];
          }

          tmp[row + x] = sum;
        }
      }

      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          let sum = 0;

          for (let k = -radius; k <= radius; k++) {
            const yy = Math.min(H - 1, Math.max(0, y + k));
            sum += tmp[yy * W + x] * kernel[k + radius];
          }

          out[y * W + x] = sum;
        }
      }

      return out;
    }

    const binaryDilate = (mask, radius = 1) => {
      if (radius <= 0) return new Uint8Array(mask);

      const horizontal = new Uint8Array(N);
      const out = new Uint8Array(N);

      for (let y = 0; y < H; y++) {
        const row = y * W;

        for (let x = 0; x < W; x++) {
          let value = 0;

          for (let dx = -radius; dx <= radius; dx++) {
            const xx = x + dx;

            if (xx >= 0 && xx < W && mask[row + xx]) {
              value = 1;
              break;
            }
          }

          horizontal[row + x] = value;
        }
      }

      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          let value = 0;

          for (let dy = -radius; dy <= radius; dy++) {
            const yy = y + dy;

            if (yy >= 0 && yy < H && horizontal[yy * W + x]) {
              value = 1;
              break;
            }
          }

          out[y * W + x] = value;
        }
      }

      return out;
    };

    const binaryErode = (mask, radius = 1) => {
      if (radius <= 0) return new Uint8Array(mask);

      const horizontal = new Uint8Array(N);
      const out = new Uint8Array(N);

      for (let y = 0; y < H; y++) {
        const row = y * W;

        for (let x = 0; x < W; x++) {
          let value = 1;

          for (let dx = -radius; dx <= radius; dx++) {
            const xx = x + dx;

            if (xx < 0 || xx >= W || !mask[row + xx]) {
              value = 0;
              break;
            }
          }

          horizontal[row + x] = value;
        }
      }

      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          let value = 1;

          for (let dy = -radius; dy <= radius; dy++) {
            const yy = y + dy;

            if (yy < 0 || yy >= H || !horizontal[yy * W + x]) {
              value = 0;
              break;
            }
          }

          out[y * W + x] = value;
        }
      }

      return out;
    };

    const binaryClose = (mask, radius = 1) =>
      binaryErode(binaryDilate(mask, radius), radius);

    const fillHoles = (mask) => {
      const outside = new Uint8Array(N);
      const queue = new Int32Array(N);
      let head = 0;
      let tail = 0;

      const pushIfBackground = (index) => {
        if (!mask[index] && !outside[index]) {
          outside[index] = 1;
          queue[tail++] = index;
        }
      };

      for (let x = 0; x < W; x++) {
        pushIfBackground(x);
        pushIfBackground((H - 1) * W + x);
      }

      for (let y = 0; y < H; y++) {
        pushIfBackground(y * W);
        pushIfBackground(y * W + W - 1);
      }

      while (head < tail) {
        const p = queue[head++];
        const x = p % W;
        const y = (p - x) / W;

        if (x > 0) pushIfBackground(p - 1);
        if (x < W - 1) pushIfBackground(p + 1);
        if (y > 0) pushIfBackground(p - W);
        if (y < H - 1) pushIfBackground(p + W);
      }

      const out = new Uint8Array(mask);

      for (let i = 0; i < N; i++) {
        if (!mask[i] && !outside[i]) {
          out[i] = 1;
        }
      }

      return out;
    };

    const largestConnectedComponent = (mask) => {
      const seen = new Uint8Array(N);
      const stack = new Int32Array(N);
      let best = [];

      for (let start = 0; start < N; start++) {
        if (!mask[start] || seen[start]) continue;

        const component = [];
        let size = 0;

        stack[size++] = start;
        seen[start] = 1;

        while (size > 0) {
          const p = stack[--size];
          component.push(p);

          const x = p % W;
          const y = (p - x) / W;

          for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
              if (!dx && !dy) continue;

              const nx = x + dx;
              const ny = y + dy;

              if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;

              const np = ny * W + nx;

              if (mask[np] && !seen[np]) {
                seen[np] = 1;
                stack[size++] = np;
              }
            }
          }
        }

        if (component.length > best.length) {
          best = component;
        }
      }

      const out = new Uint8Array(N);

      for (const index of best) {
        out[index] = 1;
      }

      return out;
    };

    const removeSmallComponents = (mask, minArea) => {
      const seen = new Uint8Array(N);
      const stack = new Int32Array(N);
      const out = new Uint8Array(N);

      for (let start = 0; start < N; start++) {
        if (!mask[start] || seen[start]) continue;

        const component = [];
        let size = 0;

        stack[size++] = start;
        seen[start] = 1;

        while (size > 0) {
          const p = stack[--size];
          component.push(p);

          const x = p % W;
          const y = (p - x) / W;

          for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
              if (!dx && !dy) continue;

              const nx = x + dx;
              const ny = y + dy;

              if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;

              const np = ny * W + nx;

              if (mask[np] && !seen[np]) {
                seen[np] = 1;
                stack[size++] = np;
              }
            }
          }
        }

        if (component.length >= minArea) {
          for (const index of component) {
            out[index] = 1;
          }
        }
      }

      return out;
    };

    const buildLocalStats = (src, radius) => {
      const stride = W + 1;
      const integral = new Float64Array((W + 1) * (H + 1));
      const integralSq = new Float64Array((W + 1) * (H + 1));

      for (let y = 1; y <= H; y++) {
        let rowSum = 0;
        let rowSqSum = 0;

        for (let x = 1; x <= W; x++) {
          const value = src[(y - 1) * W + (x - 1)];

          rowSum += value;
          rowSqSum += value * value;

          const index = y * stride + x;

          integral[index] = integral[index - stride] + rowSum;
          integralSq[index] = integralSq[index - stride] + rowSqSum;
        }
      }

      const mean = new Float32Array(N);
      const std = new Float32Array(N);

      for (let y = 0; y < H; y++) {
        const y0 = Math.max(0, y - radius);
        const y1 = Math.min(H - 1, y + radius);

        for (let x = 0; x < W; x++) {
          const x0 = Math.max(0, x - radius);
          const x1 = Math.min(W - 1, x + radius);

          const a = y0 * stride + x0;
          const b = y0 * stride + (x1 + 1);
          const c = (y1 + 1) * stride + x0;
          const d = (y1 + 1) * stride + (x1 + 1);

          const area = (x1 - x0 + 1) * (y1 - y0 + 1);

          const sum =
            integral[d] - integral[b] - integral[c] + integral[a];

          const sumSq =
            integralSq[d] -
            integralSq[b] -
            integralSq[c] +
            integralSq[a];

          const m = sum / area;
          const variance = Math.max(0, sumSq / area - m * m);
          const index = y * W + x;

          mean[index] = m;
          std[index] = Math.sqrt(variance);
        }
      }

      return { mean, std };
    };

    const hysteresis = (
      score,
      roi,
      lowThreshold,
      highThreshold
    ) => {
      const accepted = new Uint8Array(N);
      const queue = new Int32Array(N);

      let head = 0;
      let tail = 0;

      for (let i = 0; i < N; i++) {
        if (roi[i] && score[i] >= highThreshold) {
          accepted[i] = 1;
          queue[tail++] = i;
        }
      }

      while (head < tail) {
        const p = queue[head++];
        const x = p % W;
        const y = (p - x) / W;

        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            if (!dx && !dy) continue;

            const nx = x + dx;
            const ny = y + dy;

            if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;

            const np = ny * W + nx;

            if (
              !accepted[np] &&
              roi[np] &&
              score[np] >= lowThreshold
            ) {
              accepted[np] = 1;
              queue[tail++] = np;
            }
          }
        }
      }

      return accepted;
    };

    const luminance = new Float32Array(N);
    const green = new Float32Array(N);
    const gray = new Float32Array(N);

    for (let i = 0; i < N; i++) {
      const r = raw[i * 4] / 255;
      const g = raw[i * 4 + 1] / 255;
      const b = raw[i * 4 + 2] / 255;

      green[i] = g;
      gray[i] = r * 0.299 + g * 0.587 + b * 0.114;
      luminance[i] = (r + g + b) / 3;
    }

    let clean = new Uint8Array(N);
    let analysisRoi = new Uint8Array(N);

    if (mode === "retine") {
      // Création d'un meilleur masque du champ rétinien.
      const fovCandidate = new Uint8Array(N);

      for (let i = 0; i < N; i++) {
        const r = raw[i * 4] / 255;
        const g = raw[i * 4 + 1] / 255;
        const b = raw[i * 4 + 2] / 255;

        const maxChannel = Math.max(r, g, b);
        const chroma = maxChannel - Math.min(r, g, b);

        fovCandidate[i] =
          maxChannel > 0.045 &&
          (luminance[i] > 0.025 || chroma > 0.025)
            ? 1
            : 0;
      }

      let fov = largestConnectedComponent(fovCandidate);
      let fovArea = countMask(fov);

      // Si aucun bord noir n'est détecté, on utilise presque tout le cadre.
      if (fovArea < N * 0.22) {
        fov = new Uint8Array(N);

        for (let y = 10; y < H - 10; y++) {
          for (let x = 10; x < W - 10; x++) {
            fov[y * W + x] = 1;
          }
        }
      } else {
        fov = fillHoles(binaryClose(fov, 3));
      }

      // Retire le bord de l'image rétinienne.
      fov = binaryErode(fov, 8);

      analysisRoi = fov;
      fovArea = Math.max(1, countMask(fov));

      // Canal vert inversé : les vaisseaux deviennent plus clairs.
      const invertedGreen = new Float32Array(N);

      for (let i = 0; i < N; i++) {
        invertedGreen[i] = 1 - green[i];
      }

      // Correction de l'éclairage.
      const background = gauss(invertedGreen, 22);
      const local = buildLocalStats(invertedGreen, 15);

      const shadeCorrected = new Float32Array(N);
      const positiveZ = new Float32Array(N);
      const shadeValues = [];

      for (let i = 0; i < N; i++) {
        if (!fov[i]) continue;

        const shade = Math.max(
          0,
          invertedGreen[i] - background[i]
        );

        const z = Math.max(
          0,
          (invertedGreen[i] - local.mean[i]) /
            Math.max(0.018, local.std[i])
        );

        shadeCorrected[i] = shade;
        positiveZ[i] = Math.min(4, z) / 4;

        if (shade > 0) {
          shadeValues.push(shade);
        }
      }

      const shadeScale = Math.max(
        1e-5,
        percentile(shadeValues, 0.99)
      );

      const enhanced = new Float32Array(N);

      for (let i = 0; i < N; i++) {
        if (!fov[i]) continue;

        const normalizedShade = Math.min(
          1,
          shadeCorrected[i] / shadeScale
        );

        enhanced[i] =
          0.62 * normalizedShade + 0.38 * positiveZ[i];
      }

      // Filtre de Frangi multi-échelle.
      const vesselness = new Float32Array(N);
      const scales = [0.8, 1.2, 1.8, 2.6, 3.6, 5.0];
      const beta = 0.55;
      const beta2 = 2 * beta * beta;

      for (const sigma of scales) {
        const blurred = gauss(enhanced, sigma);
        const rb = new Float32Array(N);
        const structure = new Float32Array(N);

        let maxStructure = 1e-6;
        const sigma2 = sigma * sigma;

        for (let y = 1; y < H - 1; y++) {
          for (let x = 1; x < W - 1; x++) {
            const i = y * W + x;

            if (!fov[i]) continue;

            const dxx =
              (blurred[i + 1] -
                2 * blurred[i] +
                blurred[i - 1]) *
              sigma2;

            const dyy =
              (blurred[i + W] -
                2 * blurred[i] +
                blurred[i - W]) *
              sigma2;

            const dxy =
              (blurred[i + W + 1] -
                blurred[i + W - 1] -
                blurred[i - W + 1] +
                blurred[i - W - 1]) *
              0.25 *
              sigma2;

            const delta = Math.sqrt(
              (dxx - dyy) ** 2 + 4 * dxy * dxy
            );

            let lambda1 = (dxx + dyy + delta) / 2;
            let lambda2 = (dxx + dyy - delta) / 2;

            if (Math.abs(lambda1) > Math.abs(lambda2)) {
              const tmp = lambda1;
              lambda1 = lambda2;
              lambda2 = tmp;
            }

            if (lambda2 >= 0) continue;

            const s = Math.sqrt(
              lambda1 * lambda1 + lambda2 * lambda2
            );

            rb[i] = lambda1 / (lambda2 || -1e-6);
            structure[i] = s;

        
        if (s > maxStructure) {
          maxStructure = s;
        }
      }
    }

    const c = Math.max(1e-6, 0.5 * maxStructure);
    const c2 = 2 * c * c;



        for (let i = 0; i < N; i++) {
          if (structure[i] <= 0) continue;

          const response =
            Math.exp(-(rb[i] * rb[i]) / beta2) *
            (1 -
              Math.exp(
                -(structure[i] * structure[i]) / c2
              ));

          if (response > vesselness[i]) {
            vesselness[i] = response;
          }
        }
      }

      // Double seuil pour conserver les vaisseaux fins connectés.
      const values = [];

      for (let i = 0; i < N; i++) {
        if (fov[i] && vesselness[i] > 0) {
          values.push(vesselness[i]);
        }
      }

      let high = percentile(values, 0.91);
      let low = percentile(values, 0.73);

      let mask = hysteresis(
        vesselness,
        fov,
        low,
        high
      );

      let density = countMask(mask) / fovArea;

      // Ajustement automatique selon la qualité de l'image.
      if (density < 0.018) {
        high = percentile(values, 0.86);
        low = percentile(values, 0.64);

        mask = hysteresis(
          vesselness,
          fov,
          low,
          high
        );
      } else if (density > 0.19) {
        high = percentile(values, 0.95);
        low = percentile(values, 0.84);

        mask = hysteresis(
          vesselness,
          fov,
          low,
          high
        );
      }

      mask = binaryClose(mask, 1);
      clean = removeSmallComponents(mask, 10);

      for (let i = 0; i < N; i++) {
        clean[i] = clean[i] && fov[i] ? 1 : 0;
      }
    } else {
      // Segmentation de l'empreinte digitale.
      const localGray = buildLocalStats(gray, 10);
      const stdValues = [];

      for (let i = 0; i < N; i++) {
        if (luminance[i] > 0.02) {
          stdValues.push(localGray.std[i]);
        }
      }

      const varianceThreshold = Math.max(
        0.025,
        percentile(stdValues, 0.48)
      );

      const roiCandidate = new Uint8Array(N);

      for (let i = 0; i < N; i++) {
        roiCandidate[i] =
          luminance[i] > 0.02 &&
          localGray.std[i] >= varianceThreshold
            ? 1
            : 0;
      }

      let roi = largestConnectedComponent(roiCandidate);

      if (countMask(roi) < N * 0.12) {
        roi = new Uint8Array(N);

        for (let y = 12; y < H - 12; y++) {
          for (let x = 12; x < W - 12; x++) {
            roi[y * W + x] = 1;
          }
        }
      } else {
        roi = fillHoles(binaryClose(roi, 5));
        roi = binaryErode(roi, 5);
      }

      analysisRoi = roi;

      const roiArea = Math.max(1, countMask(roi));
      const ridgeLocal = new Float32Array(N);
      const invertedGray = new Float32Array(N);

      for (let i = 0; i < N; i++) {
        invertedGray[i] = 1 - gray[i];

        if (!roi[i]) continue;

        const z =
          (localGray.mean[i] - gray[i]) /
          Math.max(0.025, localGray.std[i]);

        ridgeLocal[i] = Math.min(
          1,
          Math.max(0, z / 2.5)
        );
      }

      // Différence de Gaussiennes multi-échelle.
      const dog = new Float32Array(N);

      for (const sigma of [0.7, 1.0, 1.4, 1.9]) {
        const fine = gauss(invertedGray, sigma);
        const coarse = gauss(
          invertedGray,
          sigma * 2.2
        );

        for (let i = 0; i < N; i++) {
          if (!roi[i]) continue;

          const response = Math.max(
            0,
            fine[i] - coarse[i]
          );

          if (response > dog[i]) {
            dog[i] = response;
          }
        }
      }

      const dogValues = [];

      for (let i = 0; i < N; i++) {
        if (roi[i] && dog[i] > 0) {
          dogValues.push(dog[i]);
        }
      }

      const dogScale = Math.max(
        1e-5,
        percentile(dogValues, 0.98)
      );

      const ridgeScore = new Float32Array(N);
      const scoreValues = [];

      for (let i = 0; i < N; i++) {
        if (!roi[i]) continue;

        const normalizedDog = Math.min(
          1,
          dog[i] / dogScale
        );

        ridgeScore[i] =
          0.68 * ridgeLocal[i] +
          0.32 * normalizedDog;

        scoreValues.push(ridgeScore[i]);
      }

      let high = percentile(scoreValues, 0.66);
      let low = percentile(scoreValues, 0.49);

      let mask = hysteresis(
        ridgeScore,
        roi,
        low,
        high
      );

      let density = countMask(mask) / roiArea;

      if (density < 0.16) {
        high = percentile(scoreValues, 0.60);
        low = percentile(scoreValues, 0.43);

        mask = hysteresis(
          ridgeScore,
          roi,
          low,
          high
        );
      } else if (density > 0.46) {
        high = percentile(scoreValues, 0.73);
        low = percentile(scoreValues, 0.58);

        mask = hysteresis(
          ridgeScore,
          roi,
          low,
          high
        );
      }

      mask = binaryClose(mask, 1);
      clean = removeSmallComponents(mask, 12);

      for (let i = 0; i < N; i++) {
        clean[i] = clean[i] && roi[i] ? 1 : 0;
      }
    }

    // Création de l'image du masque.
    const maskCanvas = document.createElement("canvas");
    maskCanvas.width = W;
    maskCanvas.height = H;

    const mCtx = maskCanvas.getContext("2d");
    const mData = mCtx.createImageData(W, H);

    for (let i = 0; i < N; i++) {
      const value = clean[i] ? 255 : 0;

      mData.data[i * 4] = value;
      mData.data[i * 4 + 1] = value;
      mData.data[i * 4 + 2] = value;
      mData.data[i * 4 + 3] = 255;
    }

    mCtx.putImageData(mData, 0, 0);

    // Superposition du masque sur l'image d'origine.
    const ovCanvas = document.createElement("canvas");
    ovCanvas.width = W;
    ovCanvas.height = H;

    const ovCtx = ovCanvas.getContext("2d");
    ovCtx.drawImage(origCanvas, 0, 0);

    const ovData = ovCtx.getImageData(0, 0, W, H);
    const overlay = ovData.data;

    const [highlightR, highlightG, highlightB] =
      mode === "retine"
        ? [255, 70, 70]
        : [60, 160, 255];

    for (let i = 0; i < N; i++) {
      if (!clean[i]) continue;

      overlay[i * 4] = Math.min(
        255,
        overlay[i * 4] * 0.25 +
          highlightR * 0.75
      );

      overlay[i * 4 + 1] = Math.min(
        255,
        overlay[i * 4 + 1] * 0.25 +
          highlightG * 0.75
      );

      overlay[i * 4 + 2] = Math.min(
        255,
        overlay[i * 4 + 2] * 0.25 +
          highlightB * 0.75
      );
    }

    ovCtx.putImageData(ovData, 0, 0);

    const cleanF = new Uint8Array(clean);
    const skel = skeletonize(cleanF, W, H);

    // Création de l'image du squelette.
    const skelCanvas = document.createElement("canvas");
    skelCanvas.width = W;
    skelCanvas.height = H;

    const sCtx = skelCanvas.getContext("2d");
    const sData = sCtx.createImageData(W, H);

    for (let i = 0; i < N; i++) {
      const value = skel[i] ? 255 : 0;

      sData.data[i * 4] = value;
      sData.data[i * 4 + 1] = value;
      sData.data[i * 4 + 2] = value;
      sData.data[i * 4 + 3] = 255;
    }

    sCtx.putImageData(sData, 0, 0);

    // Les minuties de l'empreinte sont calculées sur le squelette.
    const features =
      mode === "retine"
        ? extractRetineFeatures(
            cleanF,
            skel,
            W,
            H
          )
        : extractFingerprintFeatures(
            skel,
            W,
            H
          );

    const roiPixels = Math.max(
      1,
      countMask(analysisRoi)
    );

    resolve({
      maskUrl: maskCanvas.toDataURL("image/png"),
      overlayUrl: ovCanvas.toDataURL("image/png"),
      originalUrl: origCanvas.toDataURL("image/png"),
      skelUrl: skelCanvas.toDataURL("image/png"),
      fullVector: features.fullVector,
      optimizedVector: features.optimizedVector,
      optimizedArray: features.optimizedArray,
      stats: {
        ...features.stats,
        maskDensity: parseFloat(
          (
            (countMask(clean) / roiPixels) *
            100
          ).toFixed(2)
        ),
        segmentationVersion: 2,
      },
      mode,
      segmentationVersion: 2,
    });
  } catch (error) {
    URL.revokeObjectURL(url);
    reject(error);
  }
};

img.onerror = () => {
  URL.revokeObjectURL(url);

  reject(
    new Error(
      "Impossible de lire l'image sélectionnée."
    )
  );
};

img.src = url;

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
    if (u.disabled) { setErr("Ce compte a été suspendu par un administrateur."); return; }
    if (u.pendingValidation) { setErr("Compte en attente de validation par un administrateur."); return; }

    // Vérification biométrique : la rétine doit correspondre à celle enrôlée par l'admin
    if (u.role === "client") {
      if (!u.retineVector) { setErr("Aucune rétine enrôlée pour ce compte. Contactez votre administrateur."); return; }
      if (!retineFile)     { setErr("Importez votre image rétinienne pour vous authentifier."); return; }
      setLoading(true); setErr("");
      try {
        const res = await processBiometric(retineFile, "retine");
        const comparison = compareRetinaVectors(res.optimizedArray, u.retineVector);
        if (!comparison.match) {
          setLoading(false);
          setErr(`Rétine non reconnue — similarité ${comparison.similarity.toFixed(1)} %.`);
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
const [securityMode, setSecurityMode] = useState("empreinte");
const [fileRetine, setFileRetine] = useState(null);
const [fileEmpreinte, setFileEmpreinte] = useState(null);
const [pid, setPid] = useState(defaultId);
const [dragR, setDragR] = useState(false);
const [dragE, setDragE] = useState(false);
const [loading, setLoading] = useState(false);
const [msg, setMsg] = useState("");
const [err, setErr] = useState("");

const fileRefR = useRef();
const fileRefE = useRef();

const canRun =
!!fileEmpreinte &&
(securityMode === "empreinte" || !!fileRetine);

const run = async () => {
if (!canRun) return;

setLoading(true);
setErr("");

try {
  const steps = [
    "Chargement de l'empreinte digitale...",
    "Segmentation des crêtes digitales...",
    "Squelettisation et détection des minuties...",
    "Extraction du vecteur optimisé empreinte...",
    securityMode === "double"
      ? "Traitement Premium de la rétine..."
      : "Génération de la signature biométrique...",
    "Finalisation...",
  ];

  for (const step of steps) {
    setMsg(step);

    await new Promise(resolve =>
      setTimeout(resolve, 350 + Math.random() * 200)
    );
  }

  // La fonction de segmentation actuelle est conservée.
  const empreinteResult = await processBiometric(
    fileEmpreinte,
    "empreinte"
  );

  let retineResult = null;

  if (securityMode === "double" && fileRetine) {
    retineResult = await processBiometric(
      fileRetine,
      "retine"
    );
  }

  onResult({
    empreinte: empreinteResult,
    retine: retineResult,
    securityMode,
    UtilisateurId: pid || "Anonyme",
    fileNameEmpreinte: fileEmpreinte.name,
    fileNameRetine: fileRetine?.name || null,
    date: new Date()
      .toLocaleString("fr-FR")
      .slice(0, 16),
  });
} catch (error) {
  setErr(`Erreur : ${error.message}`);
} finally {
  setLoading(false);
  setMsg("");
}

};

const DropZone = ({
file,
setFile,
drag,
setDrag,
fileRef,
label,
icon,
accept,
}) => (
<>
<input
ref={fileRef}
type="file"
accept={accept}
style={{ display: "none" }}
onChange={event => {
const selectedFile = event.target.files[0];

      if (selectedFile) {
        setFile(selectedFile);
      }
    }}
  />

  <div
    onClick={() => fileRef.current.click()}
    onDragOver={event => {
      event.preventDefault();
      setDrag(true);
    }}
    onDragLeave={() => setDrag(false)}
    onDrop={event => {
      event.preventDefault();
      setDrag(false);

      const droppedFile =
        event.dataTransfer.files[0];

      if (droppedFile) {
        setFile(droppedFile);
      }
    }}
    style={{
      border: `2px dashed ${
        drag
          ? accentColor
          : file
          ? C.success
          : C.border
      }`,
      borderRadius: 10,
      padding: "20px",
      textAlign: "center",
      cursor: "pointer",
      background: drag
        ? accentColor + "0A"
        : file
        ? C.successBg
        : C.bg,
      transition: "all 0.15s",
      marginBottom: 12,
    }}
  >
    {file ? (
      <>
        <div
          style={{
            fontSize: 24,
            marginBottom: 4,
          }}
        >
          ✅
        </div>

        <div
          style={{
            fontWeight: 700,
            color: C.success,
            fontSize: 13,
          }}
        >
          {file.name}
        </div>

        <div
          style={{
            color: C.muted,
            fontSize: 11,
          }}
        >
          {(file.size / 1024).toFixed(1)} Ko
        </div>
      </>
    ) : (
      <>
        <div
          style={{
            fontSize: 28,
            marginBottom: 6,
          }}
        >
          {icon}
        </div>

        <div
          style={{
            fontWeight: 600,
            color: accentColor,
            fontSize: 13,
          }}
        >
          {label}
        </div>

        <div
          style={{
            color: C.muted,
            fontSize: 11,
          }}
        >
          PNG, JPG, BMP
        </div>
      </>
    )}
  </div>
</>

);

return (
<div
style={{
display: "grid",
gridTemplateColumns: "1fr 1fr",
gap: 24,
}}
>

<div
style={{
fontWeight: 700,
fontSize: 15,
marginBottom: 4,
}}
>
Mode d'authentification



    <div
      style={{
        color: C.sub,
        fontSize: 13,
        marginBottom: 16,
      }}
    >
      Empreinte seule ou empreinte + rétine
      en mode Premium
    </div>

    <div
      style={{
        display: "grid",
        gridTemplateColumns: "1fr 1fr",
        gap: 10,
        marginBottom: 20,
      }}
    >
      {[
        [
          "empreinte",
          "🫆",
          "Empreinte seule",
          "Niveau standard",
        ],
        [
          "double",
          "🫆 👁️",
          "Empreinte + Rétine",
          "Mode Premium",
        ],
      ].map(
        ([
          mode,
          icon,
          title,
          description,
        ]) => (
          <div
            key={mode}
            onClick={() =>
              setSecurityMode(mode)
            }
            style={{
              border: `2px solid ${
                securityMode === mode
                  ? accentColor
                  : C.border
              }`,
              borderRadius: 10,
              padding: "14px",
              cursor: "pointer",
              background:
                securityMode === mode
                  ? accentColor + "0E"
                  : C.bg,
              textAlign: "center",
              transition: "all 0.15s",
            }}
          >
            <div
              style={{
                fontSize: 22,
                marginBottom: 4,
              }}
            >
              {icon}
            </div>

            <div
              style={{
                fontWeight: 700,
                fontSize: 13,
                color:
                  securityMode === mode
                    ? accentColor
                    : C.text,
              }}
            >
              {title}
            </div>

            <div
              style={{
                color: C.muted,
                fontSize: 11,
                marginTop: 2,
              }}
            >
              {description}
            </div>
          </div>
        )
      )}
    </div>

    <label style={base.label}>
      Image empreinte digitale *
    </label>

    <DropZone
      file={fileEmpreinte}
      setFile={setFileEmpreinte}
      drag={dragE}
      setDrag={setDragE}
      fileRef={fileRefE}
      label="Déposer l'image d'empreinte"
      icon="🫆"
      accept=".png,.jpg,.jpeg,.bmp"
    />

    {securityMode === "double" && (
      <>
        <label style={base.label}>
          Image rétinienne *
        </label>

        <DropZone
          file={fileRetine}
          setFile={setFileRetine}
          drag={dragR}
          setDrag={setDragR}
          fileRef={fileRefR}
          label="Déposer l'image de rétine"
          icon="👁️"
          accept=".png,.jpg,.jpeg,.bmp"
        />
      </>
    )}

    {showId && (
      <>
        <label style={base.label}>
          Identifiant
        </label>

        <input
          style={base.input}
          type="text"
          placeholder="Ex : Jean Martin"
          value={pid}
          onChange={event =>
            setPid(event.target.value)
          }
        />
      </>
    )}

    {securityMode === "double" && (
      <div
        style={{
          padding: "10px 12px",
          background: `${accentColor}10`,
          borderRadius: 8,
          border: `1px solid ${accentColor}30`,
          fontSize: 12,
          color: accentColor,
          marginBottom: 12,
        }}
      >
        ⭐ Mode Premium : l'empreinte et la
        rétine doivent toutes les deux
        correspondre.
      </div>
    )}

    <div
      style={{
        padding: "10px 12px",
        background: C.bg,
        borderRadius: 8,
        border: `1px solid ${C.border}`,
        fontSize: 12,
        color: C.muted,
      }}
    >
      🔒 Images non stockées — seuls les
      vecteurs optimisés sont conservés
    </div>
  </div>

  <div style={base.card}>
    <div
      style={{
        fontWeight: 700,
        fontSize: 15,
        marginBottom: 4,
      }}
    >
      Pipeline de traitement
    </div>

    <div
      style={{
        color: C.sub,
        fontSize: 13,
        marginBottom: 16,
      }}
    >
      Étapes exécutées automatiquement
    </div>

    {[
      {
        n: "1",
        icon: "🖼️",
        t: "Prétraitement",
        d: "Redimensionnement 512×512 et normalisation",
      },
      {
        n: "2",
        icon: "🫆",
        t: "Segmentation empreinte",
        d: "Détection des crêtes digitales avec le pipeline existant",
      },
      {
        n: "3",
        icon: "📐",
        t: "Squelettisation",
        d: "Amincissement à 1 pixel et détection des minuties",
      },
      {
        n: "4",
        icon: "🧬",
        t: "Vecteur empreinte",
        d: "Minuties, bifurcations, terminaisons et orientations",
      },
      {
        n: "5",
        icon:
          securityMode === "double"
            ? "👁️"
            : "⚡",
        t:
          securityMode === "double"
            ? "Analyse rétinienne Premium"
            : "Signature optimisée",
        d:
          securityMode === "double"
            ? "Segmentation des vaisseaux, squelette et vecteur rétinien optimisé"
            : "Vecteur optimisé empreinte utilisé pour l'authentification",
      },
    ].map(step => (
      <div
        key={step.n}
        style={{
          display: "flex",
          gap: 12,
          marginBottom: 12,
          padding: "12px",
          background: C.bg,
          borderRadius: 9,
          border: `1px solid ${C.border}`,
        }}
      >
        <div
          style={{
            width: 26,
            height: 26,
            borderRadius: "50%",
            background: accentColor + "18",
            color: accentColor,
            fontWeight: 800,
            fontSize: 13,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            flexShrink: 0,
          }}
        >
          {step.n}
        </div>

        <div>
          <div
            style={{
              fontWeight: 700,
              fontSize: 13,
              marginBottom: 2,
            }}
          >
            {step.icon} {step.t}
          </div>

          <div
            style={{
              color: C.sub,
              fontSize: 11,
              lineHeight: 1.5,
            }}
          >
            {step.d}
          </div>
        </div>
      </div>
    ))}

    <div
      style={{
        padding: "12px 14px",
        background: accentColor + "0E",
        borderRadius: 9,
        marginBottom: 16,
        fontSize: 13,
      }}
    >
      <div
        style={{
          fontWeight: 700,
          color: accentColor,
          marginBottom: 6,
        }}
      >
        Récapitulatif
      </div>

      <div style={{ color: C.sub }}>
        Mode :{" "}
        <strong style={{ color: C.text }}>
          {securityMode === "double"
            ? "⭐ Empreinte + Rétine"
            : "Empreinte seule"}
        </strong>
      </div>

      <div
        style={{
          color: C.sub,
          marginTop: 3,
        }}
      >
        Empreinte :{" "}
        <strong
          style={{
            color: fileEmpreinte
              ? C.text
              : C.muted,
          }}
        >
          {fileEmpreinte?.name ||
            "Non sélectionnée"}
        </strong>
      </div>

      {securityMode === "double" && (
        <div
          style={{
            color: C.sub,
            marginTop: 3,
          }}
        >
          Rétine :{" "}
          <strong
            style={{
              color: fileRetine
                ? C.text
                : C.muted,
            }}
          >
            {fileRetine?.name ||
              "Non sélectionnée"}
          </strong>
        </div>
      )}
    </div>

    {err && (
      <div
        style={{
          background: C.redBg,
          color: C.red,
          border: `1px solid ${C.red}30`,
          borderRadius: 8,
          padding: "10px 14px",
          marginBottom: 12,
          fontSize: 13,
        }}
      >
        ⚠ {err}
      </div>
    )}

    <button
      style={{
        ...mkBtn("primary", accentColor),
        width: "100%",
        padding: "14px",
        fontSize: 15,
        opacity:
          !canRun || loading ? 0.6 : 1,
      }}
      onClick={run}
      disabled={!canRun || loading}
    >
      {loading ? (
        <>
          <span
            style={{
              animation:
                "spin 1s linear infinite",
              display: "inline-block",
            }}
          >
            ⟳
          </span>
          &nbsp;{msg}
        </>
      ) : (
        <>
          {Ic.scan}&nbsp;Lancer l'analyse
        </>
      )}
    </button>

    {loading && (
      <div
        style={{
          height: 5,
          background: C.border,
          borderRadius: 4,
          overflow: "hidden",
          position: "relative",
          marginTop: 10,
        }}
      >
        <div
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            height: "100%",
            background: accentColor,
            animation:
              "progress 3s ease-in-out forwards",
            borderRadius: 4,
          }}
        />
      </div>
    )}

    <style>
      {`
        @keyframes spin {
          to {
            transform: rotate(360deg);
          }
        }

        @keyframes progress {
          from {
            width: 0%;
          }

          to {
            width: 100%;
          }
        }
      `}
    </style>
  </div>
</div>

);
}
function ResultsPanel({
result,
accentColor = C.primary,
onNew,
onEnroll,
onAuth,
}) {
const [
fingerViewMode,
setFingerViewMode,
] = useState("overlay");

const [
retinaViewMode,
setRetinaViewMode,
] = useState("overlay");

if (!result) {
return (
<div
style={{
textAlign: "center",
padding: "80px 0",
color: C.sub,
}}
>
<div
style={{
fontSize: 52,
marginBottom: 16,
}}
>
🧬



    <div
      style={{
        fontSize: 16,
        fontWeight: 600,
        marginBottom: 8,
      }}
    >
      Aucune analyse effectuée
    </div>

    <div style={{ marginBottom: 24 }}>
      Importez une empreinte digitale pour
      lancer le pipeline.
    </div>

    {onNew && (
      <button
        style={mkBtn(
          "primary",
          accentColor
        )}
        onClick={onNew}
      >
        {Ic.scan}&nbsp;Analyser une image
      </button>
    )}
  </div>
);

}

const empreinte = result.empreinte;
const retine = result.retine;
const isDouble =
result.securityMode === "double";

const retinaKeys = [
"OvLen",
"TI",
"MedTor",
"D1",
"D2",
];

const fingerprintKeys = [
"nbMinutiae",
"nbBifurcations",
"nbTerminations",
"minutiaeDensity",
"meanOrientation",
"orientationVariation",
];

const FullVectorBlock = ({
title,
data,
keys,
color,
}) => (

<div
style={{
fontWeight: 700,
fontSize: 13,
color,
marginBottom: 8,
textTransform: "uppercase",
letterSpacing: "0.05em",
}}
>
{title}



  <div
    style={{
      background: "#080F1E",
      borderRadius: 10,
      padding: "14px 16px",
      fontFamily: "monospace",
      fontSize: 11,
      lineHeight: 1.9,
    }}
  >
    {keys.map(key => (
      <div key={key}>
        <span style={{ color: "#94A3B8" }}>
          {key}:{" "}
        </span>

        <span style={{ color: "#60A5FA" }}>
          {typeof data[key] === "number"
            ? data[key].toFixed(6)
            : data[key]}
        </span>
      </div>
    ))}
  </div>
</div>

);

const OptimizedVector = ({
title,
vector,
keys,
color,
}) => (

<div
style={{
fontWeight: 800,
fontSize: 16,
marginBottom: 5,
}}
>
{title}



  <div
    style={{
      color: C.sub,
      fontSize: 12,
      marginBottom: 18,
    }}
  >
    Signature utilisée pour
    l'authentification biométrique
  </div>

  <div
    style={{
      display: "grid",
      gridTemplateColumns:
        "repeat(2,minmax(0,1fr))",
      gap: 12,
      marginBottom: 18,
    }}
  >
    {keys.map((key, index) => (
      <div
        key={key}
        style={{
          background: C.bg,
          border: `1px solid ${C.border}`,
          borderRadius: 10,
          padding: "14px",
        }}
      >
        <div
          style={{
            color: C.muted,
            fontSize: 11,
            fontWeight: 700,
            marginBottom: 6,
          }}
        >
          {key}
        </div>

        <div
          style={{
            color,
            fontFamily: "monospace",
            fontSize: 18,
            fontWeight: 800,
            wordBreak: "break-all",
          }}
        >
          {Number(
            vector[index]
          ).toFixed(6)}
        </div>
      </div>
    ))}
  </div>

  <div
    style={{
      background: "#080F1E",
      borderRadius: 10,
      padding: "16px",
      fontFamily: "monospace",
      fontSize: 12,
      color: "#4ADE80",
      lineHeight: 1.7,
      wordBreak: "break-word",
    }}
  >
    [
    {vector
      .map(value =>
        Number(value).toFixed(4)
      )
      .join(", ")}
    ]
  </div>
</div>

);

const ImageViewer = ({
title,
data,
viewMode,
setViewMode,
color,
legend,
}) => (

<div
style={{
display: "flex",
alignItems: "center",
justifyContent: "space-between",
marginBottom: 12,
gap: 10,
flexWrap: "wrap",
}}
>
<div
style={{
fontWeight: 700,
fontSize: 14,
}}
>
{title}



    <div
      style={{
        display: "flex",
        gap: 6,
        flexWrap: "wrap",
      }}
    >
      {[
        ["original", "Original"],
        ["overlay", "Superposition"],
        ["mask", "Masque"],
        ["skel", "Squelette"],
      ].map(([value, label]) => (
        <button
          key={value}
          style={{
            ...mkBtn(
              viewMode === value
                ? "primary"
                : "ghost",
              color
            ),
            padding: "4px 9px",
            fontSize: 11,
          }}
          onClick={() =>
            setViewMode(value)
          }
        >
          {label}
        </button>
      ))}
    </div>
  </div>

  <div
    style={{
      background: "#080F1E",
      borderRadius: 10,
      overflow: "hidden",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      height: 300,
    }}
  >
    <img
      key={viewMode}
      src={
        viewMode === "original"
          ? data.originalUrl
          : viewMode === "mask"
          ? data.maskUrl
          : viewMode === "skel"
          ? data.skelUrl
          : data.overlayUrl
      }
      alt={title}
      style={{
        maxWidth: "100%",
        maxHeight: "100%",
        objectFit: "contain",
      }}
    />
  </div>

  <div
    style={{
      marginTop: 8,
      fontSize: 11,
      color: C.muted,
      textAlign: "center",
    }}
  >
    {viewMode === "original"
      ? "🖼️ Image originale"
      : viewMode === "mask"
      ? "⬜ Masque binaire"
      : viewMode === "skel"
      ? "📐 Squelette à 1 pixel"
      : legend}
  </div>
</div>

);

return (
<>
<div
style={{
display: "flex",
alignItems: "flex-start",
justifyContent: "space-between",
marginBottom: 20,
flexWrap: "wrap",
gap: 12,
}}
>

<div
style={{
display: "flex",
alignItems: "center",
gap: 8,
marginBottom: 6,
flexWrap: "wrap",
}}
>

✓ Pipeline complet



        <span style={mkBadge(C.accent)}>
          🫆 Empreinte
        </span>

        {isDouble && (
          <span style={mkBadge(C.primary)}>
            👁️ Rétine
          </span>
        )}

        {isDouble && (
          <span style={mkBadge(C.warning)}>
            ⭐ Mode Premium
          </span>
        )}

        <span
          style={{
            color: C.muted,
            fontSize: 13,
          }}
        >
          {result.UtilisateurId}
        </span>
      </div>

      <h2
        style={{
          fontSize: 18,
          fontWeight: 800,
        }}
      >
        Segmentation · Squelette ·
        Signature biométrique
      </h2>
    </div>

    <div
      style={{
        display: "flex",
        gap: 8,
        flexWrap: "wrap",
      }}
    >
      {onNew && (
        <button
          style={mkBtn("ghost")}
          onClick={onNew}
        >
          {Ic.scan}&nbsp;Nouvelle analyse
        </button>
      )}

      {onEnroll && (
        <button
          style={mkBtn(
            "soft",
            C.success
          )}
          onClick={() =>
            onEnroll(result)
          }
        >
          {Ic.plus}&nbsp;Enrôler
        </button>
      )}

      {onAuth && (
        <button
          style={mkBtn(
            "primary",
            accentColor
          )}
          onClick={() =>
            onAuth(result)
          }
        >
          {Ic.search}&nbsp;Authentifier
        </button>
      )}
    </div>
  </div>

  <div
    style={{
      display: "grid",
      gridTemplateColumns:
        "minmax(0,1fr) minmax(380px,1fr)",
      gap: 20,
      alignItems: "stretch",
    }}
  >
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 20,
      }}
    >
      <ImageViewer
        title="🫆 Visualisation empreinte"
        data={empreinte}
        viewMode={fingerViewMode}
        setViewMode={setFingerViewMode}
        color={C.accent}
        legend="🔵 Bleu = crêtes digitales détectées"
      />

      <div style={base.card}>
        <div
          style={{
            display: "grid",
            gridTemplateColumns:
              "repeat(3,1fr)",
            gap: 8,
            marginBottom: 16,
          }}
        >
          {[
            [
              empreinte.stats
                .nbMinutiae ?? "-",
              "Minuties",
            ],
            [
              empreinte.stats
                .nbBifurcations ?? "-",
              "Bifurcations",
            ],
            [
              empreinte.stats
                .nbTerminations ?? "-",
              "Terminaisons",
            ],
          ].map(([value, label]) => (
            <div
              key={label}
              style={{
                background: C.bg,
                borderRadius: 8,
                padding: "10px",
                border: `1px solid ${C.border}`,
                textAlign: "center",
              }}
            >
              <div
                style={{
                  fontWeight: 800,
                  fontSize: 16,
                  color: C.accent,
                }}
              >
                {value}
              </div>

              <div
                style={{
                  fontSize: 10,
                  color: C.muted,
                  marginTop: 1,
                }}
              >
                {label}
              </div>
            </div>
          ))}
        </div>

        <FullVectorBlock
          title="Détails empreinte"
          data={empreinte.fullVector}
          color={C.accent}
          keys={[
            "nbMinutiae",
            "nbBifurcations",
            "nbTerminations",
            "minutiaeDensity",
            "meanOrientation",
            "orientationVariation",
            "density",
          ]}
        />
      </div>

      {isDouble && retine && (
        <>
          <ImageViewer
            title="👁️ Visualisation rétine Premium"
            data={retine}
            viewMode={retinaViewMode}
            setViewMode={setRetinaViewMode}
            color={C.primary}
            legend="🔴 Rouge = vaisseaux rétiniens détectés"
          />

          <div style={base.card}>
            <div
              style={{
                display: "grid",
                gridTemplateColumns:
                  "repeat(3,1fr)",
                gap: 8,
                marginBottom: 16,
              }}
            >
              {[
                [
                  retine.stats.OvLen?.toFixed(
                    0
                  ) || "-",
                  "Longueur totale",
                ],
                [
                  retine.stats.TI?.toFixed(
                    3
                  ) || "-",
                  "Tortuosité",
                ],
                [
                  retine.stats.D1?.toFixed(
                    2
                  ) || "-",
                  "Diamètre moyen",
                ],
              ].map(([value, label]) => (
                <div
                  key={label}
                  style={{
                    background: C.bg,
                    borderRadius: 8,
                    padding: "10px",
                    border: `1px solid ${C.border}`,
                    textAlign: "center",
                  }}
                >
                  <div
                    style={{
                      fontWeight: 800,
                      fontSize: 16,
                      color: C.primary,
                    }}
                  >
                    {value}
                  </div>

                  <div
                    style={{
                      fontSize: 10,
                      color: C.muted,
                      marginTop: 1,
                    }}
                  >
                    {label}
                  </div>
                </div>
              ))}
            </div>

            <FullVectorBlock
              title="Morphologie vasculaire"
              data={retine.fullVector}
              color={C.primary}
              keys={[
                "OvLen",
                "TI",
                "MedTor",
                "D1",
                "D2",
                "nbSegments",
                "nbBifurcations",
                "nbTerminations",
                "density",
              ]}
            />
          </div>
        </>
      )}
    </div>

    <div
      style={{
        ...base.card,
        display: "flex",
        flexDirection: "column",
        justifyContent: "space-between",
        minHeight: 620,
      }}
    >
      <div>
        <OptimizedVector
          title="🫆 Vecteur optimisé empreinte"
          vector={empreinte.optimizedArray}
          keys={fingerprintKeys}
          color={C.accent}
        />

        {isDouble && retine && (
          <div
            style={{
              borderTop: `1px solid ${C.border}`,
              marginTop: 26,
              paddingTop: 26,
            }}
          >
            <OptimizedVector
              title="👁️ Vecteur optimisé rétine Premium"
              vector={retine.optimizedArray}
              keys={retinaKeys}
              color={C.primary}
            />
          </div>
        )}
      </div>

      <div
        style={{
          marginTop: 28,
          padding: "14px 16px",
          background: C.primaryLight,
          borderRadius: 10,
          color: C.primary,
          fontSize: 12,
          lineHeight: 1.6,
        }}
      >
        {isDouble
          ? "⭐ Mode Premium : la signature de l'empreinte et la signature rétinienne sont toutes les deux générées avec les pipelines de segmentation existants."
          : "🔒 Mode standard : seule la signature optimisée de l'empreinte est générée et utilisée."}
      </div>
    </div>
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
    if (authMode === "double" && !authFileE) return;

    setLoading(true);
    setMsg("Calcul du vecteur rétine...");
    setAuthResult(null);

    try {
      const retineRes = await processBiometric(authFileR, "retine");
      let empreinteRes = null;

      if (authMode === "double") {
        setMsg("Calcul du vecteur empreinte...");
        empreinteRes = await processBiometric(authFileE, "empreinte");
      }

      setMsg("Comparaison avec la base...");

      const results = database.map(entry => {
        const retinaComparison = compareRetinaVectors(
          retineRes.optimizedArray,
          entry.retineVector
        );

        const fingerprintComparison = authMode === "double"
          ? compareFingerprintVectors(
              empreinteRes?.optimizedArray,
              entry.empreinteVector
            )
          : null;

        const globalMatch = authMode === "double"
          ? retinaComparison.match && fingerprintComparison?.match
          : retinaComparison.match;

        const globalSimilarity = authMode === "double"
          ? (retinaComparison.similarity + (fingerprintComparison?.similarity || 0)) / 2
          : retinaComparison.similarity;

        return {
          ...entry,
          retineDist: retinaComparison.distance,
          empreinteDist: fingerprintComparison?.distance ?? null,
          retineSimilarity: retinaComparison.similarity,
          empreinteSimilarity: fingerprintComparison?.similarity ?? null,
          retineMatch: retinaComparison.match,
          empreinteMatch: fingerprintComparison?.match ?? false,
          globalMatch,
          globalSimilarity,
          globalScore: 100 - globalSimilarity,
        };
      });

      results.sort((a, b) => b.globalSimilarity - a.globalSimilarity);
      const bestMatch = results.find(result => result.globalMatch);

      setAccessPopup(bestMatch
        ? { status:"autorise", name:bestMatch.name, similarity:bestMatch.globalSimilarity }
        : { status:"refuse", name:null, similarity:results[0]?.globalSimilarity || 0 }
      );

      setAuthResult({
        results,
        retineVec:retineRes.optimizedArray,
        empreinteVec:empreinteRes?.optimizedArray || null,
      });
    } catch (error) {
      setAccessPopup({ status:"refuse", name:null, similarity:0 });
      setAuthResult(null);
    } finally {
      setLoading(false);
      setMsg("");
    }
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
              <p style={{ fontSize:16, color:C.text, marginBottom:8 }}>
                Utilisateur reconnu : <strong>{accessPopup.name}</strong>
              </p>
            )}
            <p style={{ fontSize:13, color:C.sub, marginBottom:20 }}>
              Similarité biométrique : <strong>{Number(accessPopup.similarity || 0).toFixed(1)}%</strong>
            </p>

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
              Comparaison multi-critères : longueur vasculaire, tortuosité et diamètres. La même image produit 100 % de similarité.
            </div>

            <button style={{ ...mkBtn("primary",accentColor), width:"100%", padding:"13px", opacity:(!authFileR || (authMode==="double"&&!authFileE) || loading)?0.5:1 }}
              disabled={!authFileR || (authMode==="double"&&!authFileE) || loading} onClick={runAuth}>
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
                  const retinePct = r.retineSimilarity;
                  return (
                    <div key={r.id} style={{ border:`2px solid ${r.globalMatch?C.success:i===0?C.warning:C.border}`, borderRadius:11, padding:"14px", marginBottom:12, background:r.globalMatch?C.successBg:C.surface }}>
                      <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:8 }}>
                        <div style={{ fontWeight:700, fontSize:15 }}>{r.name}</div>
                        <span style={mkChip(r.globalMatch?C.success:C.red)}>{r.globalMatch?"✓ Identifié":"✗ Non identifié"}</span>
                      </div>
                      <div style={{ fontSize:12, color:C.sub, marginBottom:8 }}>
                        👁️ Similarité rétine : <strong style={{ color:r.retineMatch?C.success:C.red }}>{r.retineSimilarity.toFixed(1)}%</strong>
                        {r.empreinteSimilarity !== null && <> · 🫆 Similarité empreinte : <strong style={{ color:r.empreinteMatch?C.success:C.red }}>{r.empreinteSimilarity.toFixed(1)}%</strong></>}
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
      username,
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
// GESTION DES COMPTES — administrateurs et utilisateurs
// ═══════════════════════════════════════════════════════════════════════════════
function AccountManagementPanel({
  users,
  currentUser,
  onCreateUser,
  onUpdateUser,
  onDeleteUser,
  database,
  setDatabase,
  accentColor=C.primary,
  initialTab="clients",
}) {
  const [tab, setTab] = useState(initialTab);
  const [clientView, setClientView] = useState("list");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [adminForm, setAdminForm] = useState({ name:"", username:"", email:"", password:"" });
  const [editing, setEditing] = useState(null);

  const accounts = Object.entries(users);
  const admins = accounts.filter(([, account]) => account.role === "Administrateur");
  const clients = accounts.filter(([, account]) => account.role === "client");
  const activeAdmins = admins.filter(([, account]) => !account.disabled && !account.pendingValidation);

  useEffect(() => {
    setTab(initialTab);
    setClientView("list");
    setError("");
  }, [initialTab]);

  const notify = message => {
    setSuccess(message);
    setError("");
    window.setTimeout(() => setSuccess(""), 2600);
  };

  const createAdmin = () => {
    const username = adminForm.username.trim();
    if (!adminForm.name.trim()) return setError("Le nom complet est requis.");
    if (username.length < 4) return setError("Identifiant trop court (4 caractères minimum).");
    if (users[username]) return setError("Cet identifiant existe déjà.");
    if (adminForm.email && !adminForm.email.includes("@")) return setError("Adresse email invalide.");
    if (adminForm.password.length < 6) return setError("Le mot de passe doit contenir au moins 6 caractères.");

    onCreateUser({
      username,
      password:adminForm.password,
      role:"Administrateur",
      name:adminForm.name.trim(),
      email:adminForm.email.trim(),
      validated:true,
      pendingValidation:false,
      disabled:false,
      createdAt:new Date().toLocaleString("fr-FR").slice(0,16),
    });
    setAdminForm({ name:"", username:"", email:"", password:"" });
    notify("Administrateur créé et activé.");
  };

  const approve = username => {
    onUpdateUser(username, { pendingValidation:false, validated:true, disabled:false });
    notify(`Le compte ${username} a été validé.`);
  };

  const toggleAccount = (username, account) => {
    if (username === currentUser.username) {
      setError("Vous ne pouvez pas suspendre votre propre compte.");
      return;
    }
    if (account.role === "Administrateur" && !account.disabled && !account.pendingValidation && activeAdmins.length <= 1) {
      setError("Impossible de suspendre le dernier administrateur actif.");
      return;
    }
    onUpdateUser(username, { disabled:!account.disabled });
    notify(account.disabled ? `Compte ${username} réactivé.` : `Compte ${username} suspendu.`);
  };

  const resetPassword = username => {
    const password = window.prompt(`Nouveau mot de passe pour ${username} (6 caractères minimum) :`);
    if (password === null) return;
    if (password.length < 6) {
      setError("Mot de passe trop court.");
      return;
    }
    onUpdateUser(username, { password });
    notify(`Mot de passe de ${username} réinitialisé.`);
  };

  const startEdit = (username, account) => {
    setEditing({
      originalUsername:username,
      username,
      name:account.name || "",
      email:account.email || "",
      password:"",
      role:account.role,
    });
    setError("");
  };

  const saveEdit = () => {
    if (!editing) return;
    const nextUsername = editing.username.trim();
    const nextName = editing.name.trim();
    const nextEmail = editing.email.trim();

    if (nextUsername.length < 4) return setError("Identifiant trop court (4 caractères minimum).");
    if (!nextName) return setError("Le nom complet est requis.");
    if (nextEmail && !nextEmail.includes("@")) return setError("Adresse email invalide.");
    if (nextUsername !== editing.originalUsername && users[nextUsername]) return setError("Ce nouvel identifiant existe déjà.");
    if (editing.password && editing.password.length < 6) return setError("Le nouveau mot de passe doit contenir au moins 6 caractères.");

    const changes = {
      name:nextName,
      email:nextEmail,
    };
    if (editing.password) changes.password = editing.password;

    onUpdateUser(editing.originalUsername, changes, nextUsername);

    if (editing.role === "client") {
      setDatabase(previous => previous.map(entry => {
        const linkedByUsername = entry.username === editing.originalUsername;
        const legacyNameMatch = !entry.username && entry.name === users[editing.originalUsername]?.name;
        return linkedByUsername || legacyNameMatch
          ? { ...entry, username:nextUsername, name:nextName }
          : entry;
      }));
    }

    setEditing(null);
    notify(`Les informations de ${nextUsername} ont été modifiées.`);
  };

  const removeAccount = (username, account) => {
    if (username === currentUser.username) {
      setError("Vous ne pouvez pas supprimer votre propre compte.");
      return;
    }
    if (account.role === "Administrateur" && !account.disabled && !account.pendingValidation && activeAdmins.length <= 1) {
      setError("Impossible de supprimer le dernier administrateur actif.");
      return;
    }
    if (!window.confirm(`Supprimer définitivement le compte ${username} ?`)) return;
    onDeleteUser(username);
    if (account.role === "client") {
      setDatabase(previous => previous.filter(entry => entry.username !== username));
    }
    notify(`Compte ${username} supprimé.`);
  };

  const statusBadge = account => {
    if (account.pendingValidation) return <span style={mkChip(C.warning)}>En attente</span>;
    if (account.disabled) return <span style={mkChip(C.red)}>Suspendu</span>;
    return <span style={mkChip(C.success)}>Actif</span>;
  };

  const AccountTable = ({ rows }) => (
    <div style={{ background:C.surface, border:`1px solid ${C.border}`, borderRadius:12, overflow:"auto" }}>
      <table style={{ width:"100%", borderCollapse:"collapse", minWidth:940 }}>
        <thead>
          <tr>
            <th style={base.th}>Identifiant</th>
            <th style={base.th}>Nom</th>
            <th style={base.th}>Email</th>
            <th style={base.th}>Biométrie</th>
            <th style={base.th}>Statut</th>
            <th style={base.th}>Actions</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([username, account], index) => (
            <tr key={username} style={{ background:index % 2 === 0 ? C.surface : C.bg }}>
              <td style={base.td}>
                <code style={{ color:accentColor, fontWeight:700 }}>{username}</code>
                {username === currentUser.username && <span style={{ ...mkBadge(C.primary), marginLeft:8 }}>Vous</span>}
              </td>
              <td style={{ ...base.td, fontWeight:700 }}>{account.name}</td>
              <td style={{ ...base.td, color:C.sub }}>{account.email || "—"}</td>
              <td style={base.td}>
                {account.role === "client"
                  ? <span style={mkChip(account.retineVector ? C.success : C.warning)}>{account.retineVector ? "Rétine enrôlée" : "Non enrôlée"}</span>
                  : <span style={mkBadge(C.primary)}>Compte admin</span>}
              </td>
              <td style={base.td}>{statusBadge(account)}</td>
              <td style={base.td}>
                <div style={{ display:"flex", gap:7, flexWrap:"wrap" }}>
                  {account.pendingValidation && (
                    <button style={{ ...mkBtn("soft",C.success), padding:"6px 10px", fontSize:12 }} onClick={() => approve(username)}>Valider</button>
                  )}
                  <button style={{ ...mkBtn("soft",C.primary), padding:"6px 10px", fontSize:12 }} onClick={() => startEdit(username, account)}>Modifier</button>
                  <button style={{ ...mkBtn("soft",account.disabled ? C.success : C.warning), padding:"6px 10px", fontSize:12 }} onClick={() => toggleAccount(username, account)}>
                    {account.disabled ? "Réactiver" : "Suspendre"}
                  </button>
                  <button style={{ ...mkBtn("soft",C.primary), padding:"6px 10px", fontSize:12 }} onClick={() => resetPassword(username)}>Mot de passe</button>
                  <button style={{ ...mkBtn("soft",C.red), padding:"6px 10px", fontSize:12 }} onClick={() => removeAccount(username, account)}>Supprimer</button>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );

  return (
    <>
      {editing && (
        <div style={{ position:"fixed", inset:0, background:"rgba(15,23,42,0.55)", display:"flex", alignItems:"center", justifyContent:"center", zIndex:9999, padding:20 }}>
          <div style={{ ...base.card, width:460, maxWidth:"100%", boxShadow:"0 24px 70px rgba(0,0,0,0.28)" }}>
            <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:18 }}>
              <div>
                <div style={{ fontWeight:900, fontSize:18 }}>Modifier le compte</div>
                <div style={{ color:C.sub, fontSize:12, marginTop:3 }}>{editing.role === "Administrateur" ? "Administrateur" : "Utilisateur"}</div>
              </div>
              <button style={{ ...mkBtn("ghost"), padding:"7px 10px" }} onClick={() => setEditing(null)}>✕</button>
            </div>
            <label style={base.label}>Nom complet *</label>
            <input style={base.input} value={editing.name} onChange={event => setEditing(previous => ({ ...previous, name:event.target.value }))} />
            <label style={base.label}>Identifiant *</label>
            <input style={base.input} value={editing.username} onChange={event => setEditing(previous => ({ ...previous, username:event.target.value }))} />
            <label style={base.label}>Email</label>
            <input style={base.input} type="email" value={editing.email} onChange={event => setEditing(previous => ({ ...previous, email:event.target.value }))} />
            <label style={base.label}>Nouveau mot de passe (optionnel)</label>
            <input style={base.input} type="password" value={editing.password} onChange={event => setEditing(previous => ({ ...previous, password:event.target.value }))} placeholder="Laisser vide pour ne pas changer" />
            <div style={{ display:"flex", gap:10, justifyContent:"flex-end" }}>
              <button style={mkBtn("ghost")} onClick={() => setEditing(null)}>Annuler</button>
              <button style={mkBtn("primary",accentColor)} onClick={saveEdit}>Enregistrer</button>
            </div>
          </div>
        </div>
      )}

      <div style={{ display:"flex", alignItems:"flex-start", justifyContent:"space-between", gap:16, flexWrap:"wrap", marginBottom:20 }}>
        <div>
          <h2 style={{ fontSize:19, fontWeight:800, marginBottom:4 }}>
            {tab === "admins" ? "Gestion des administrateurs" : "Gestion des utilisateurs"}
          </h2>
          <p style={{ color:C.sub, fontSize:13 }}>
            {tab === "admins"
              ? "Créer, valider, modifier, suspendre et supprimer les comptes administrateurs."
              : "Ajouter, modifier, suspendre, réactiver ou supprimer les comptes utilisateurs."}
          </p>
        </div>
        <div style={{ display:"flex", gap:8, flexWrap:"wrap" }}>
          <span style={mkBadge(C.primary)}>{activeAdmins.length} admin(s) actif(s)</span>
          <span style={mkBadge(C.accent)}>{clients.length} utilisateur(s)</span>
          <span style={mkBadge(C.warning)}>{admins.filter(([, account]) => account.pendingValidation).length} demande(s)</span>
        </div>
      </div>

      {error && <div style={{ background:C.redBg, color:C.red, border:`1px solid ${C.red}30`, borderRadius:9, padding:"11px 14px", marginBottom:14 }}>⚠ {error}</div>}
      {success && <div style={{ background:C.successBg, color:C.success, border:`1px solid ${C.success}30`, borderRadius:9, padding:"11px 14px", marginBottom:14 }}>✓ {success}</div>}

      {tab === "admins" && (
        <div style={{ display:"grid", gridTemplateColumns:"minmax(300px,0.8fr) minmax(0,1.7fr)", gap:20 }}>
          <div style={base.card}>
            <div style={{ fontWeight:800, fontSize:15, marginBottom:5 }}>Créer un administrateur</div>
            <div style={{ color:C.sub, fontSize:12, marginBottom:16 }}>Le compte est actif immédiatement.</div>
            <label style={base.label}>Nom complet *</label>
            <input style={base.input} value={adminForm.name} onChange={event => setAdminForm(previous => ({ ...previous, name:event.target.value }))} placeholder="Prénom NOM" />
            <label style={base.label}>Identifiant *</label>
            <input style={base.input} value={adminForm.username} onChange={event => setAdminForm(previous => ({ ...previous, username:event.target.value }))} placeholder="4 caractères minimum" />
            <label style={base.label}>Email</label>
            <input style={base.input} type="email" value={adminForm.email} onChange={event => setAdminForm(previous => ({ ...previous, email:event.target.value }))} placeholder="email@exemple.fr" />
            <label style={base.label}>Mot de passe *</label>
            <input style={base.input} type="password" value={adminForm.password} onChange={event => setAdminForm(previous => ({ ...previous, password:event.target.value }))} placeholder="6 caractères minimum" />
            <button style={{ ...mkBtn("primary",accentColor), width:"100%", padding:"12px" }} onClick={createAdmin}>{Ic.plus}&nbsp;Créer l'administrateur</button>
          </div>
          {admins.length ? <AccountTable rows={admins} /> : <div style={{ ...base.card, textAlign:"center", padding:48, color:C.muted }}>Aucun administrateur.</div>}
        </div>
      )}

      {tab === "clients" && (
        <>
          <div style={{ display:"flex", gap:8, marginBottom:18 }}>
            <button style={mkBtn(clientView === "list" ? "primary" : "ghost",accentColor)} onClick={() => setClientView("list")}>Liste des utilisateurs</button>
            <button style={mkBtn(clientView === "create" ? "primary" : "ghost",accentColor)} onClick={() => setClientView("create")}>{Ic.plus}&nbsp;Ajouter un utilisateur</button>
          </div>
          {clientView === "create" ? (
            <CreateUserPanel
              users={users}
              onCreateUser={account => {
                onCreateUser(account);
                notify(`Utilisateur ${account.username} créé.`);
              }}
              database={database}
              setDatabase={setDatabase}
              accentColor={accentColor}
            />
          ) : clients.length ? (
            <AccountTable rows={clients} />
          ) : (
            <div style={{ ...base.card, textAlign:"center", padding:"48px", color:C.muted }}>
              <div style={{ fontSize:38, marginBottom:10 }}>👥</div>
              <div style={{ marginBottom:16 }}>Aucun compte utilisateur.</div>
              <button style={mkBtn("primary",accentColor)} onClick={() => setClientView("create")}>{Ic.plus}&nbsp;Ajouter le premier utilisateur</button>
            </div>
          )}
        </>
      )}
    </>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPARAISON DIRECTE DE DEUX RÉTINES
// ═══════════════════════════════════════════════════════════════════════════════
function RetinaComparisonPanel({ accentColor=C.primary }) {
  const [fileA, setFileA] = useState(null);
  const [fileB, setFileB] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const refA = useRef();
  const refB = useRef();

  const run = async () => {
    if (!fileA || !fileB) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const [retinaA, retinaB] = await Promise.all([
        processBiometric(fileA, "retine"),
        processBiometric(fileB, "retine"),
      ]);
      const comparison = compareRetinaVectors(retinaA.optimizedArray, retinaB.optimizedArray);
      setResult({ retinaA, retinaB, comparison });
    } catch (exception) {
      setError(`Comparaison impossible : ${exception.message}`);
    } finally {
      setLoading(false);
    }
  };

  const Drop = ({ file, setFile, inputRef, label }) => (
    <>
      <input ref={inputRef} type="file" accept=".png,.jpg,.jpeg,.bmp" style={{ display:"none" }} onChange={event => setFile(event.target.files[0] || null)} />
      <div onClick={() => inputRef.current.click()} style={{ border:`2px dashed ${file ? C.success : C.border}`, borderRadius:11, padding:"28px 18px", textAlign:"center", cursor:"pointer", background:file ? C.successBg : C.bg }}>
        <div style={{ fontSize:30, marginBottom:6 }}>{file ? "✅" : "👁️"}</div>
        <div style={{ fontWeight:700, color:file ? C.success : accentColor, fontSize:13 }}>{file ? file.name : label}</div>
      </div>
    </>
  );

  return (
    <>
      <h2 style={{ fontSize:19, fontWeight:800, marginBottom:4 }}>Comparer deux rétines</h2>
      <p style={{ color:C.sub, fontSize:13, marginBottom:20 }}>Les deux images passent exactement par le même masque, la même segmentation et le même calcul de signature.</p>

      <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:20, marginBottom:20 }}>
        <div style={base.card}>
          <div style={{ fontWeight:700, marginBottom:12 }}>Rétine A</div>
          <Drop file={fileA} setFile={setFileA} inputRef={refA} label="Importer la première rétine" />
        </div>
        <div style={base.card}>
          <div style={{ fontWeight:700, marginBottom:12 }}>Rétine B</div>
          <Drop file={fileB} setFile={setFileB} inputRef={refB} label="Importer la seconde rétine" />
        </div>
      </div>

      {error && <div style={{ background:C.redBg, color:C.red, borderRadius:9, padding:"11px 14px", marginBottom:14 }}>⚠ {error}</div>}
      <button disabled={!fileA || !fileB || loading} onClick={run} style={{ ...mkBtn("primary",accentColor), padding:"12px 22px", opacity:!fileA || !fileB || loading ? 0.5 : 1, marginBottom:20 }}>
        {loading ? "Analyse des deux rétines..." : "Comparer les signatures"}
      </button>

      {result && (
        <div style={{ ...base.card, border:`2px solid ${result.comparison.match ? C.success : C.red}` }}>
          <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", gap:12, flexWrap:"wrap", marginBottom:18 }}>
            <div>
              <div style={{ fontWeight:900, fontSize:20, color:result.comparison.match ? C.success : C.red }}>
                {result.comparison.match ? "✓ Signatures compatibles" : "✗ Signatures différentes"}
              </div>
              <div style={{ color:C.sub, fontSize:13, marginTop:4 }}>
                Similarité calculée : {result.comparison.similarity.toFixed(1)} %
              </div>
            </div>
            <span style={mkChip(result.comparison.match ? C.success : C.red)}>
              {result.comparison.exact ? "Correspondance exacte" : result.comparison.match ? "Correspondance tolérée" : "Non reconnu"}
            </span>
          </div>
          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:14 }}>
            {[result.retinaA.optimizedArray, result.retinaB.optimizedArray].map((vector, index) => (
              <div key={index} style={{ background:"#080F1E", borderRadius:9, padding:"14px", fontFamily:"monospace", color:index === 0 ? "#4ADE80" : "#60A5FA", fontSize:12, wordBreak:"break-word" }}>
                <div style={{ color:"#94A3B8", marginBottom:7 }}>Vecteur {index === 0 ? "A" : "B"}</div>
                [{vector.map(value => Number(value).toFixed(4)).join(", ")}]
              </div>
            ))}
          </div>
        </div>
      )}
    </>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// AdministrateurAPP
// ═══════════════════════════════════════════════════════════════════════════════
function AdministrateurApp({ user, users, onCreateUser, onUpdateUser, onDeleteUser, onLogout }) {
  const [page, setPage] = useState("dashboard");
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);

  const [database, setDatabase] = useState(() => {
    const fallback = [
      { id:"camille", name:"Camille", date:"18/06/2026", retineVector:[5937,1.5451,1.2127,2.8667,3.1170], empreinteVector:[41501,41498,3,953.3447,-0.0086,1.9102], hasEmpreinte:true },
      { id:"steven", name:"Steven", date:"19/06/2026", retineVector:[6273,1.5488,1.2,1.913,1.5993], empreinteVector:[66111,66109,2,991.4667,-1.5095,2.5649], hasEmpreinte:true },
      { id:"tidar", name:"Tidar", date:"18/06/2026", retineVector:[6413,1.621,1.2,2.9683,3.034], empreinteVector:[60293,60291,2,999.1383,0.468,1.8129], hasEmpreinte:true },
      { id:"shanice", name:"Shanice", date:"18/06/2026", retineVector:[6154,1.4684,1.177,2.2308,1.8462], empreinteVector:[73165,73165,0,999.5492,1.4086,2.0869], hasEmpreinte:true },
      { id:"aminata", name:"Aminata", date:"18/06/2026", retineVector:[4941,1.5814,1.1142,2.1613,1.8156], empreinteVector:[69246,69244,2,999.7257,1.247,1.9928], hasEmpreinte:true },
    ];
    try {
      const saved = localStorage.getItem("segvision_biometric_database_v2");
      return saved ? JSON.parse(saved) : fallback;
    } catch (error) {
      return fallback;
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem("segvision_biometric_database_v2", JSON.stringify(database));
    } catch (error) {
      console.error("Impossible d'enregistrer la base biométrique", error);
    }
  }, [database]);

  const onResult = res => {
    setResult(res);
    setHistory(previous => [{
      id:`B-${Date.now().toString().slice(-4)}`,
      date:res.date,
      Utilisateur:res.UtilisateurId,
      mode:res.securityMode === "double" ? "Rétine + Empreinte" : "Rétine",
      action:"Analyse",
      result:"✓ Extrait",
    }, ...previous]);
    setPage("resultats");
  };

  const adminCount = Object.values(users).filter(account => account.role === "Administrateur" && !account.disabled && !account.pendingValidation).length;
  const clientCount = Object.values(users).filter(account => account.role === "client").length;

  const NAV = [
    { id:"dashboard", label:"Tableau de bord", icon:Ic.grid },
    { id:"admins", label:"Gestion des administrateurs", icon:Ic.shield },
    { id:"utilisateurs", label:"Gestion des utilisateurs", icon:Ic.user },
    { id:"analyse", label:"Nouvelle analyse", icon:Ic.scan },
    { id:"resultats", label:"Résultats", icon:Ic.chart },
    { id:"comparaison", label:"Comparer 2 rétines", icon:Ic.search },
    { id:"biometrie", label:"Base biométrique", icon:Ic.db },
    { id:"historique", label:"Historique", icon:Ic.clock },
    { id:"securite", label:"Sécurité", icon:Ic.shield },
  ];

  return (
    <Shell
      user={user}
      page={page}
      setPage={setPage}
      navItems={NAV}
      onLogout={onLogout}
      sidebarColor={C.sidebar}
      activeColor={C.primary}
      topRight={<span style={mkBadge(C.primary)}>Administrateur</span>}
    >
      {page === "dashboard" && (
        <>
          <h1 style={{ fontSize:22, fontWeight:800, marginBottom:4 }}>Bienvenue, {user.name}</h1>
          <p style={{ color:C.sub, marginBottom:24 }}>Administration des accès et du système biométrique SegVision.</p>
          <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:16, marginBottom:24 }}>
            {[
              { value:String(adminCount), label:"Administrateurs actifs" },
              { value:String(clientCount), label:"Comptes utilisateurs" },
              { value:String(database.length), label:"Personnes enrôlées" },
              { value:String(history.length), label:"Opérations cette session" },
            ].map(card => (
              <div key={card.label} style={base.card}>
                <div style={{ width:36, height:3, background:C.primary, borderRadius:2, marginBottom:12 }} />
                <div style={{ fontSize:26, fontWeight:800, marginBottom:4 }}>{card.value}</div>
                <div style={{ fontSize:13, color:C.sub }}>{card.label}</div>
              </div>
            ))}
          </div>

          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:20, marginBottom:20 }}>
            <div style={base.card}>
              <div style={{ fontWeight:700, fontSize:15, marginBottom:12 }}>Administration des accès</div>
              {["Valider les demandes administrateur","Suspendre ou réactiver un compte","Réinitialiser les mots de passe","Supprimer les accès obsolètes"].map((step,index) => (
                <div key={step} style={{ display:"flex", gap:10, alignItems:"center", padding:"8px 0", borderBottom:`1px solid ${C.border}` }}>
                  <div style={{ width:20, height:20, borderRadius:"50%", background:C.primaryLight, color:C.primary, fontWeight:800, fontSize:11, display:"flex", alignItems:"center", justifyContent:"center" }}>{index + 1}</div>
                  <div style={{ fontSize:13 }}>{step}</div>
                </div>
              ))}
            </div>
            <div style={base.card}>
              <div style={{ fontWeight:700, fontSize:15, marginBottom:12 }}>Contrôle biométrique</div>
              {["Enrôler une signature rétinienne","Comparer deux rétines","Authentifier contre la base","Supprimer une identité biométrique"].map((step,index) => (
                <div key={step} style={{ display:"flex", gap:10, alignItems:"center", padding:"8px 0", borderBottom:`1px solid ${C.border}` }}>
                  <div style={{ width:20, height:20, borderRadius:"50%", background:C.accentLight, color:C.accent, fontWeight:800, fontSize:11, display:"flex", alignItems:"center", justifyContent:"center" }}>{index + 1}</div>
                  <div style={{ fontSize:13 }}>{step}</div>
                </div>
              ))}
            </div>
          </div>

          <div style={{ display:"flex", gap:12, flexWrap:"wrap" }}>
            <button style={mkBtn("primary",C.primary)} onClick={() => setPage("admins")}>{Ic.shield}&nbsp;Gérer les administrateurs</button>
            <button style={mkBtn("soft",C.primary)} onClick={() => setPage("utilisateurs")}>{Ic.user}&nbsp;Gérer les utilisateurs</button>
            <button style={mkBtn("soft",C.accent)} onClick={() => setPage("comparaison")}>{Ic.search}&nbsp;Comparer deux rétines</button>
            <button style={mkBtn("soft",C.success)} onClick={() => setPage("biometrie")}>{Ic.db}&nbsp;Base biométrique</button>
          </div>
        </>
      )}

      {page === "admins" && (
        <AccountManagementPanel
          users={users}
          currentUser={user}
          onCreateUser={onCreateUser}
          onUpdateUser={onUpdateUser}
          onDeleteUser={onDeleteUser}
          database={database}
          setDatabase={setDatabase}
          accentColor={C.primary}
          initialTab="admins"
        />
      )}

      {page === "utilisateurs" && (
        <AccountManagementPanel
          users={users}
          currentUser={user}
          onCreateUser={onCreateUser}
          onUpdateUser={onUpdateUser}
          onDeleteUser={onDeleteUser}
          database={database}
          setDatabase={setDatabase}
          accentColor={C.primary}
          initialTab="clients"
        />
      )}

      {page === "analyse" && <UploadPanel onResult={onResult} accentColor={C.primary} showId={true} />}

      {page === "resultats" && (
        <ResultsPanel
          result={result}
          accentColor={C.primary}
          onNew={() => setPage("analyse")}
          onEnroll={() => setPage("biometrie")}
          onAuth={() => setPage("biometrie")}
        />
      )}

      {page === "comparaison" && <RetinaComparisonPanel accentColor={C.primary} />}

      {page === "biometrie" && <BiometricDB database={database} setDatabase={setDatabase} accentColor={C.primary} />}

      {page === "historique" && (
        <>
          <h2 style={{ fontSize:18, fontWeight:800, marginBottom:20 }}>Historique des opérations</h2>
          {history.length === 0 ? (
            <div style={{ ...base.card, textAlign:"center", padding:"40px", color:C.muted }}>
              <div style={{ fontSize:36, marginBottom:12 }}>📋</div>
              <div>Aucune opération effectuée pendant cette session.</div>
            </div>
          ) : (
            <div style={{ background:C.surface, border:`1px solid ${C.border}`, borderRadius:12, overflow:"hidden" }}>
              <table style={{ width:"100%", borderCollapse:"collapse" }}>
                <thead><tr><th style={base.th}>ID</th><th style={base.th}>Date</th><th style={base.th}>Utilisateur</th><th style={base.th}>Mode</th><th style={base.th}>Action</th><th style={base.th}>Résultat</th></tr></thead>
                <tbody>{history.map((row,index) => (
                  <tr key={row.id} style={{ background:index % 2 === 0 ? C.surface : C.bg }}>
                    <td style={base.td}><code style={{ fontSize:12, color:C.primary }}>{row.id}</code></td>
                    <td style={{ ...base.td, fontSize:12 }}>{row.date}</td>
                    <td style={base.td}>{row.Utilisateur}</td>
                    <td style={base.td}>{row.mode}</td>
                    <td style={base.td}>{row.action}</td>
                    <td style={base.td}><span style={mkChip(C.success)}>{row.result}</span></td>
                  </tr>
                ))}</tbody>
              </table>
            </div>
          )}
        </>
      )}

      {page === "securite" && (
        <>
          <h2 style={{ fontSize:18, fontWeight:800, marginBottom:4 }}>Sécurité & conformité</h2>
          <p style={{ color:C.sub, fontSize:13, marginBottom:20 }}>État des contrôles du prototype.</p>
          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:16 }}>
            {[
              { icon:"🗑️", title:"Pas de stockage d'image", description:"Les images ne sont pas placées dans la base. Seules les signatures numériques sont conservées." },
              { icon:"👥", title:"Gestion des comptes", description:"Validation, suspension, réactivation, réinitialisation et suppression des accès." },
              { icon:"👁️", title:"Comparaison rétinienne", description:"Comparaison multi-critères avec égalité exacte pour la même image et tolérance contrôlée entre acquisitions proches." },
              { icon:"🫆", title:"Double authentification biométrique", description:"En mode renforcé, la rétine et l'empreinte doivent toutes les deux correspondre." },
              { icon:"💾", title:"Persistance locale", description:"Comptes et signatures restent dans le navigateur après rechargement du site." },
              { icon:"📋", title:"Journal de session", description:"Les analyses effectuées pendant la session sont listées dans l'historique." },
            ].map(item => (
              <div key={item.title} style={{ ...base.card, display:"flex", gap:14 }}>
                <div style={{ fontSize:26 }}>{item.icon}</div>
                <div>
                  <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", gap:10, marginBottom:6 }}>
                    <div style={{ fontWeight:700, fontSize:14 }}>{item.title}</div>
                    <span style={mkChip(C.success)}>Actif</span>
                  </div>
                  <div style={{ color:C.sub, fontSize:13 }}>{item.description}</div>
                </div>
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
return (
<div
style={{
minHeight: "100vh",
background: C.bg,
...F,
}}
>
<header
style={{
height: 72,
padding: "0 24px",
display: "flex",
alignItems: "center",
justifyContent: "space-between",
background: C.clientSidebar,
boxSizing: "border-box",
}}
> <Logo size={38} />

    <button
      onClick={onLogout}
      style={{
        ...mkBtn("ghost"),
        color: "#FFFFFF",
        background: "rgba(255,255,255,0.10)",
        border: "1px solid rgba(255,255,255,0.25)",
      }}
    >
      {Ic.logout}
      Déconnexion
    </button>
  </header>

  <main
    style={{
      minHeight: "calc(100vh - 72px)",
      padding: 24,
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      boxSizing: "border-box",
    }}
  >
    <div
      style={{
        width: "100%",
        maxWidth: 700,
        padding: 48,
        background: C.surface,
        border: `1px solid ${C.border}`,
        borderRadius: 20,
        textAlign: "center",
        boxShadow: "0 20px 60px rgba(0,0,0,0.15)",
        boxSizing: "border-box",
      }}
    >
      <div
        style={{
          width: 80,
          height: 80,
          margin: "0 auto 24px",
          borderRadius: "50%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: C.successBg,
          color: C.success,
          fontSize: 38,
          fontWeight: 800,
        }}
      >
        ✓
      </div>

      <h1
        style={{
          margin: "0 0 14px",
          color: C.text,
          fontSize: 34,
          fontWeight: 900,
        }}
      >
        Bonjour {user.username}
      </h1>

      <p
        style={{
          margin: 0,
          color: C.sub,
          fontSize: 20,
          lineHeight: 1.6,
        }}
      >
        Vous avez accès au site.
      </p>
    </div>
  </main>
</div>

);
}

// ═══════════════════════════════════════════════════════════════════════════════
// ROOT
// ═══════════════════════════════════════════════════════════════════════════════
export default function App() {
  const [user, setUser] = useState(null);
  const [users, setUsers] = useState(() => {
    try {
      const saved = localStorage.getItem("segvision_users");
      return saved ? JSON.parse(saved) : { ...INITIAL_USERS };
    } catch (error) {
      return { ...INITIAL_USERS };
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem("segvision_users", JSON.stringify(users));
    } catch (error) {
      console.error("Impossible d'enregistrer les comptes", error);
    }
  }, [users]);

  const register = account => {
    const normalized = {
      disabled:false,
      pendingValidation:false,
      validated:true,
      ...account,
      username:account.username,
    };
    setUsers(previous => ({ ...previous, [normalized.username]:normalized }));
  };

  const updateUser = (username, changes, nextUsername=username) => {
    setUsers(previous => {
      if (!previous[username]) return previous;
      const updated = {
        ...previous[username],
        ...changes,
        username:nextUsername,
      };
      const next = { ...previous };
      delete next[username];
      next[nextUsername] = updated;
      return next;
    });

    setUser(current => current?.username === username
      ? { ...current, ...changes, username:nextUsername }
      : current
    );
  };

  const deleteUser = username => {
    setUsers(previous => {
      const next = { ...previous };
      delete next[username];
      return next;
    });
  };

  if (!user) {
    return <LoginPage onLogin={setUser} users={users} onRegister={register} />;
  }

  if (user.role === "Administrateur") {
    return (
      <AdministrateurApp
        user={user}
        users={users}
        onCreateUser={register}
        onUpdateUser={updateUser}
        onDeleteUser={deleteUser}
        onLogout={() => setUser(null)}
      />
    );
  }

  return <UtilisateurApp user={user} onLogout={() => setUser(null)} />;
}
