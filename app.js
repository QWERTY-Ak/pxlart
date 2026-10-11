/* ==========================================================================
   Pixelify — image → pixel art
   --------------------------------------------------------------------------
   The algorithm:
     1. Load the image and cap its working size (keeps everything fast).
     2. DOWNSAMPLE: split it into a cols × rows grid and average every cell
        (in linear light, so colors don't get muddy/dark).
     3. ADJUST: boost contrast + saturation — pixel art reads better punchy.
     4. PALETTE: either pick the N best colors with k-means clustering in the
        perceptual OKLab color space, or use a fixed retro palette.
     5. MAP + DITHER: snap every cell to a palette color — directly, with an
        ordered (Bayer) pattern, or with error diffusion (Floyd–Steinberg /
        Atkinson) so gradients survive a tiny palette.
     6. RENDER: draw each cell as a solid square (nearest-neighbour upscale),
        optionally with grid lines on top.
   ========================================================================== */
(() => {
  'use strict';

  const DEFAULTS = Object.freeze({
    gridSize: 48,
    palette: 'auto',
    colorCount: 16,
    dither: 'none',
    ditherStrength: 85,
    saturation: 120,
    contrast: 110,
    showGrid: false,
    gridColor: '#0b0b14',
    gridOpacity: 45,
    exportScale: 16,
  });

  const MAX_SOURCE_SIDE = 1024; // working resolution cap
  const MAX_EXPORT_SIDE = 8192; // browser canvas safety limit
  const SAMPLE_URL = 'assets/sample.jpg';

  // ---------------------------------------------------------- palettes ----
  const hexList = (str) => {
    const seen = new Set();
    return str
      .trim()
      .split(/\s+/)
      .map((h) => h.replace('#', '').toLowerCase())
      .filter((h) => !seen.has(h) && seen.add(h))
      .map((h) => ({ r: parseInt(h.slice(0, 2), 16), g: parseInt(h.slice(2, 4), 16), b: parseInt(h.slice(4, 6), 16) }));
  };

  // `luma: true` → match by brightness only (great for monochrome palettes,
  // so a red apple becomes a dark green instead of "the closest green hue").
  const PALETTES = [
    { id: 'auto', name: 'Auto', kind: 'auto', desc: 'Best colors picked automatically from your image.' },
    {
      id: 'gameboy', name: 'Game Boy', kind: 'preset', luma: true,
      desc: 'The 4 iconic greens of the original 1989 Game Boy.',
      colors: hexList('0f380f 306230 8bac0f 9bbc0f'),
    },
    {
      id: 'pico8', name: 'PICO-8', kind: 'preset',
      desc: 'The 16-color palette of the PICO-8 fantasy console.',
      colors: hexList('000000 1d2b53 7e2553 008751 ab5236 5f574f c2c3c7 fff1e8 ff004d ffa300 ffec27 00e436 29adff 83769c ff77a8 ffccaa'),
    },
    {
      id: 'nes', name: 'NES', kind: 'preset',
      desc: 'The colors the Nintendo Entertainment System could display.',
      colors: hexList(`
        7c7c7c 0000fc 0000bc 4428bc 940084 a80020 a81000 881400 503000 007800 006800 005800 004058
        bcbcbc 0078f8 0058f8 6844fc d800cc e40058 f83800 e45c10 ac7c00 00b800 00a800 00a844 008888
        f8f8f8 3cbcfc 6888fc 9878f8 f878f8 f85898 f87858 fca044 f8b800 b8f818 58d854 58f898 00e8d8 787878
        fcfcfc a4e4fc b8b8f8 d8b8f8 f8b8f8 f8a4c0 f0d0b0 fce0a8 f8d878 d8f878 b8f8b8 b8f8d8 00fcfc f8d8f8
        000000`),
    },
    {
      id: 'c64', name: 'Commodore 64', kind: 'preset',
      desc: 'The 16 colors of the Commodore 64 (Pepto palette).',
      colors: hexList('000000 ffffff 68372b 70a4b2 6f3d86 588d43 352879 b8c76f 6f4f25 433900 9a6759 444444 6c6c6c 9ad284 6c5eb5 959595'),
    },
    {
      id: 'cga', name: 'CGA', kind: 'preset',
      desc: 'Black, cyan, magenta & white — 1980s IBM PC graphics.',
      colors: hexList('000000 55ffff ff55ff ffffff'),
    },
    {
      id: 'sweetie16', name: 'Sweetie 16', kind: 'preset',
      desc: 'A much-loved modern 16-color palette by GrafxKid.',
      colors: hexList('1a1c2c 5d275d b13e53 ef7d57 ffcd75 a7f070 38b764 257179 29366f 3b5dc9 41a6f6 73eff7 f4f4f4 94b0c2 566c86 333c57'),
    },
    {
      id: 'endesga32', name: 'Endesga 32', kind: 'preset',
      desc: 'A rich 32-color palette by Endesga — great for scenes.',
      colors: hexList('be4a2f d77643 ead4aa e4a672 b86f50 733e39 3e2731 a22633 e43b44 f77622 feae34 fee761 63c74d 3e8948 265c42 193c3e 124e89 0099db 2ce8f5 ffffff c0cbdc 8b9bb4 5a6988 3a4466 262b44 181425 ff0044 68386c b55088 f6757a e8b796 c28569'),
    },
    {
      id: '1bit', name: '1-bit', kind: 'preset', luma: true,
      desc: 'Pure black and white — pair it with dithering!',
      colors: hexList('000000 ffffff'),
    },
    { id: 'full', name: 'Full color', kind: 'full', desc: 'No color reduction — every cell keeps its averaged color.' },
  ];
  const PALETTE_BY_ID = Object.fromEntries(PALETTES.map((p) => [p.id, p]));

  // --------------------------------------------------------- dithering ----
  function bayerMatrix(n) {
    if (n === 1) return [0];
    const h = n / 2;
    const prev = bayerMatrix(h);
    const m = new Array(n * n);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < h; x++) {
        const v = prev[y * h + x] * 4;
        m[y * n + x] = v;
        m[y * n + x + h] = v + 2;
        m[(y + h) * n + x] = v + 3;
        m[(y + h) * n + x + h] = v + 1;
      }
    }
    return m;
  }

  const DITHERS = {
    none: { chip: 'None', name: 'No dithering', desc: 'Flat areas of color — the cleanest pixel-art look.' },
    bayer2: { chip: '2×2', name: 'Bayer 2×2', type: 'ordered', size: 2, desc: 'Tiny checkerboard pattern — very chunky, very retro.' },
    bayer4: { chip: '4×4', name: 'Bayer 4×4', type: 'ordered', size: 4, desc: 'The classic crosshatch pattern of old consoles & PCs.' },
    bayer8: { chip: '8×8', name: 'Bayer 8×8', type: 'ordered', size: 8, desc: 'A finer ordered pattern for smoother gradients.' },
    floyd: {
      chip: 'Floyd', name: 'Floyd–Steinberg', type: 'diffusion',
      desc: 'Error diffusion — organic, detailed, slightly noisy.',
      // [dx, dy, weight]
      kernel: [[1, 0, 7 / 16], [-1, 1, 3 / 16], [0, 1, 5 / 16], [1, 1, 1 / 16]],
    },
    atkinson: {
      chip: 'Atkinson', name: 'Atkinson', type: 'diffusion',
      desc: 'The classic Macintosh look — crisp, high-contrast texture.',
      kernel: [[1, 0, 1 / 8], [2, 0, 1 / 8], [-1, 1, 1 / 8], [0, 1, 1 / 8], [1, 1, 1 / 8], [0, 2, 1 / 8]],
    },
  };
  for (const d of Object.values(DITHERS)) {
    if (d.type === 'ordered') d.matrix = bayerMatrix(d.size);
  }

  // ---------------------------------------------------------------- DOM ----
  const $ = (id) => document.getElementById(id);
  const els = {
    fileInput: $('file-input'),
    uploadBtn: $('upload-btn'),
    downloadBtn: $('download-btn'),
    resetBtn: $('reset-btn'),
    compareBtn: $('compare-btn'),
    dropzone: $('dropzone'),
    stageBody: $('stage-body'),
    canvasWrap: $('canvas-wrap'),
    canvas: $('pixel-canvas'),
    original: $('original-img'),
    palette: $('palette'),
    toast: $('toast'),

    gridSize: $('grid-size'),
    gridSizeValue: $('grid-size-value'),
    gridPresets: $('grid-presets'),
    gridDimsHint: $('grid-dims-hint'),

    palettePicker: $('palette-picker'),
    paletteDesc: $('palette-desc'),
    colorCount: $('color-count'),
    colorCountValue: $('color-count-value'),
    colorCountField: $('color-count-field'),

    ditherOptions: $('dither-options'),
    ditherDesc: $('dither-desc'),
    ditherFields: $('dither-fields'),
    ditherStrength: $('dither-strength'),
    ditherStrengthValue: $('dither-strength-value'),
    ditherStrengthField: $('dither-strength-field'),

    saturation: $('saturation'),
    saturationValue: $('saturation-value'),
    contrast: $('contrast'),
    contrastValue: $('contrast-value'),

    showGrid: $('show-grid'),
    gridLineFields: $('grid-line-fields'),
    gridColor: $('grid-color'),
    gridOpacity: $('grid-opacity'),
    gridOpacityValue: $('grid-opacity-value'),

    exportScale: $('export-scale'),
    exportScaleValue: $('export-scale-value'),
    exportSizeHint: $('export-size-hint'),

    statGrid: $('stat-grid'),
    statColors: $('stat-colors'),
    statTime: $('stat-time'),
  };

  const state = {
    settings: { ...DEFAULTS },
    source: null, // { canvas, w, h }
    fileName: 'image',
    objectUrl: null,
    cache: null, // downsampled grid, keyed by grid size
    autoPalCache: null, // cached auto palette { key, pal }
    result: null, // final pixel art
  };

  // ------------------------------------------------------ color helpers ----
  const SRGB_TO_LINEAR = new Float32Array(256);
  for (let i = 0; i < 256; i++) {
    const c = i / 255;
    SRGB_TO_LINEAR[i] = c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  }

  const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
  const srgbFToLinear = (c) => SRGB_TO_LINEAR[Math.round(clamp01(c) * 255)];
  const linearToSrgbF = (v) => (v <= 0 ? 0 : v >= 1 ? 1 : v <= 0.0031308 ? v * 12.92 : 1.055 * v ** (1 / 2.4) - 0.055);
  const linearToSrgb = (v) => Math.round(linearToSrgbF(v) * 255);

  // OKLab — a perceptual color space, so "distance" matches what eyes see.
  function linearToOklab(lr, lg, lb, out, o) {
    const l = Math.cbrt(0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb);
    const m = Math.cbrt(0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb);
    const s = Math.cbrt(0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb);
    out[o] = 0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s;
    out[o + 1] = 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s;
    out[o + 2] = 0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s;
  }

  // From 0–255 ints (fast, via lookup table)
  const srgbToOklab = (r, g, b, out, o) => linearToOklab(SRGB_TO_LINEAR[r], SRGB_TO_LINEAR[g], SRGB_TO_LINEAR[b], out, o);
  // From 0–1 floats (instant via lookup table without Math.pow)
  const srgbFToOklab = (r, g, b, out, o) => {
    const ir = Math.round(clamp01(r) * 255);
    const ig = Math.round(clamp01(g) * 255);
    const ib = Math.round(clamp01(b) * 255);
    linearToOklab(SRGB_TO_LINEAR[ir], SRGB_TO_LINEAR[ig], SRGB_TO_LINEAR[ib], out, o);
  };

  function oklabToSrgb(L, a, b) {
    const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
    const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
    const s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3;
    return [
      linearToSrgb(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s),
      linearToSrgb(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s),
      linearToSrgb(-0.0041960785 * l - 0.7034186147 * m + 1.707614701 * s),
    ];
  }

  // Gray (0–1 sRGB) with the same perceived brightness as the given color.
  const grayOf = (r, g, b) =>
    linearToSrgbF(0.2126 * srgbFToLinear(r) + 0.7152 * srgbFToLinear(g) + 0.0722 * srgbFToLinear(b));

  const toHex = (r, g, b) => '#' + [r, g, b].map((v) => v.toString(16).padStart(2, '0')).join('');

  const lightness = (c) => {
    const t = [0, 0, 0];
    srgbToOklab(c.r, c.g, c.b, t, 0);
    return t[0];
  };

  // Small deterministic PRNG so the same settings always give the same palette.
  function mulberry32(seed) {
    return () => {
      seed |= 0;
      seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // ---------------------------------------------------- 1. load source ----
  const sampleCanvas = document.createElement('canvas');
  const sampleCtx = sampleCanvas.getContext('2d', { willReadFrequently: true });

  function prepareSource(img) {
    const scale = Math.min(1, MAX_SOURCE_SIDE / Math.max(img.naturalWidth, img.naturalHeight));
    const w = Math.max(1, Math.round(img.naturalWidth * scale));
    const h = Math.max(1, Math.round(img.naturalHeight * scale));
    const c = document.createElement('canvas');
    c.width = w;
    c.height = h;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(img, 0, 0, w, h);
    return { canvas: c, w, h };
  }

  function gridDimensions(src, size) {
    if (src.w >= src.h) {
      return { cols: size, rows: Math.max(1, Math.round((size * src.h) / src.w)) };
    }
    return { cols: Math.max(1, Math.round((size * src.w) / src.h)), rows: size };
  }

  // ------------------------------------------------------ 2. downsample ----
  // Fast hardware-accelerated canvas downsample with area smoothing
  function downsample(src, cols, rows) {
    sampleCanvas.width = cols;
    sampleCanvas.height = rows;
    sampleCtx.imageSmoothingEnabled = true;
    sampleCtx.imageSmoothingQuality = 'high';
    sampleCtx.clearRect(0, 0, cols, rows);
    sampleCtx.drawImage(src.canvas, 0, 0, cols, rows);
    const imgData = sampleCtx.getImageData(0, 0, cols, rows).data;

    const n = cols * rows;
    const rgb = new Uint8ClampedArray(n * 3);
    const opaque = new Uint8Array(n);

    for (let i = 0, p = 0; i < n; i++, p += 4) {
      if (imgData[p + 3] >= 64) {
        opaque[i] = 1;
        rgb[i * 3] = imgData[p];
        rgb[i * 3 + 1] = imgData[p + 1];
        rgb[i * 3 + 2] = imgData[p + 2];
      }
    }
    return { cols, rows, rgb, opaque };
  }

  // ---------------------------------------------------------- 3. adjust ----
  function adjust(base, saturation, contrast) {
    const rgb = new Uint8ClampedArray(base.rgb); // clamps + rounds for us
    const s = saturation / 100;
    const k = contrast / 100;
    for (let i = 0, p = 0; i < base.opaque.length; i++, p += 3) {
      if (!base.opaque[i]) continue;
      const r = (rgb[p] - 128) * k + 128;
      const g = (rgb[p + 1] - 128) * k + 128;
      const b = (rgb[p + 2] - 128) * k + 128;
      const lum = 0.299 * r + 0.587 * g + 0.114 * b;
      rgb[p] = lum + (r - lum) * s;
      rgb[p + 1] = lum + (g - lum) * s;
      rgb[p + 2] = lum + (b - lum) * s;
    }
    return rgb;
  }

  // ---------------------------------------------------- 4a. auto palette ----
  // Fast k-means++ clustering in OKLab with point subsampling (runs in ~1-2ms)
  function autoPalette(rgb, opaque, k) {
    const indices = [];
    for (let i = 0; i < opaque.length; i++) if (opaque[i]) indices.push(i);
    const n = indices.length;
    if (n === 0) return [];

    // Subsample points so k-means is instant even on dense grids
    const maxSamples = 800;
    let sampleIndices = indices;
    if (n > maxSamples) {
      sampleIndices = [];
      const step = n / maxSamples;
      for (let i = 0; i < maxSamples; i++) {
        sampleIndices.push(indices[Math.floor(i * step)]);
      }
    }
    const sampleN = sampleIndices.length;

    const pts = new Float32Array(sampleN * 3);
    for (let j = 0; j < sampleN; j++) {
      const p = sampleIndices[j] * 3;
      srgbToOklab(rgb[p], rgb[p + 1], rgb[p + 2], pts, j * 3);
    }

    k = Math.min(k, sampleN);
    const rand = mulberry32(0x9e3779b9);
    const centers = new Float32Array(k * 3);
    const dist = new Float32Array(sampleN).fill(Infinity);

    // --- k-means++ seeding: spread initial centers far apart.
    const c0 = Math.floor(rand() * sampleN);
    centers.set(pts.subarray(c0 * 3, c0 * 3 + 3), 0);
    let found = 1;
    for (; found < k; found++) {
      const pc = (found - 1) * 3;
      let sum = 0;
      for (let j = 0; j < sampleN; j++) {
        const q = j * 3;
        const dL = pts[q] - centers[pc], da = pts[q + 1] - centers[pc + 1], db = pts[q + 2] - centers[pc + 2];
        const d = dL * dL + da * da + db * db;
        if (d < dist[j]) dist[j] = d;
        sum += dist[j];
      }
      if (sum <= 1e-12) break; // fewer distinct colors than k
      let target = rand() * sum;
      let pick = sampleN - 1;
      for (let j = 0; j < sampleN; j++) {
        target -= dist[j];
        if (target <= 0) { pick = j; break; }
      }
      centers.set(pts.subarray(pick * 3, pick * 3 + 3), found * 3);
    }
    k = found;

    // --- Lloyd iterations (10 iterations max for quick response).
    const labels = new Int32Array(sampleN).fill(-1);
    const sums = new Float64Array(k * 3);
    const counts = new Int32Array(k);

    for (let iter = 0; iter < 10; iter++) {
      let changed = 0;
      for (let j = 0; j < sampleN; j++) {
        const q = j * 3;
        let best = 0, bestD = Infinity;
        for (let c = 0; c < k; c++) {
          const cc = c * 3;
          const dL = pts[q] - centers[cc], da = pts[q + 1] - centers[cc + 1], db = pts[q + 2] - centers[cc + 2];
          const d = dL * dL + da * da + db * db;
          if (d < bestD) { bestD = d; best = c; }
        }
        if (labels[j] !== best) { labels[j] = best; changed++; }
      }
      if (changed === 0) break;

      sums.fill(0);
      counts.fill(0);
      for (let j = 0; j < sampleN; j++) {
        const c = labels[j], q = j * 3;
        sums[c * 3] += pts[q];
        sums[c * 3 + 1] += pts[q + 1];
        sums[c * 3 + 2] += pts[q + 2];
        counts[c]++;
      }
      for (let c = 0; c < k; c++) {
        if (counts[c] > 0) {
          centers[c * 3] = sums[c * 3] / counts[c];
          centers[c * 3 + 1] = sums[c * 3 + 1] / counts[c];
          centers[c * 3 + 2] = sums[c * 3 + 2] / counts[c];
        }
      }
    }

    // --- Convert centers back to sRGB, dropping unused / duplicate colors.
    const used = new Uint8Array(k);
    for (let j = 0; j < sampleN; j++) used[labels[j]] = 1;
    const seen = new Set();
    const colors = [];
    for (let c = 0; c < k; c++) {
      if (!used[c]) continue;
      const [r, g, b] = oklabToSrgb(centers[c * 3], centers[c * 3 + 1], centers[c * 3 + 2]);
      const key = (r << 16) | (g << 8) | b;
      if (seen.has(key)) continue;
      seen.add(key);
      colors.push({ r, g, b });
    }
    return colors;
  }

  // ------------------------------------------------- 4b. prepare palette ----
  // Precomputes what the mapper needs: each color as 0–1 sRGB ("proxy", used
  // for dithering error) and as OKLab (used for nearest-color matching).
  // In luma mode the proxies are grays, so matching is by brightness only.
  function preparePalette(colors, luma = false) {
    const count = colors.length;
    const proxy = new Float32Array(count * 3);
    const lab = new Float32Array(count * 3);
    colors.forEach((c, i) => {
      let r = c.r / 255, g = c.g / 255, b = c.b / 255;
      if (luma) r = g = b = grayOf(r, g, b);
      proxy[i * 3] = r;
      proxy[i * 3 + 1] = g;
      proxy[i * 3 + 2] = b;
      srgbFToOklab(r, g, b, lab, i * 3);
    });

    // How far apart palette colors typically are → amplitude for ordered
    // dithering, so the pattern just bridges the gap between neighbours.
    let spread = 0;
    if (count > 1) {
      const nn = [];
      for (let i = 0; i < count; i++) {
        let best = Infinity;
        for (let j = 0; j < count; j++) {
          if (i === j) continue;
          const dr = proxy[i * 3] - proxy[j * 3];
          const dg = proxy[i * 3 + 1] - proxy[j * 3 + 1];
          const db = proxy[i * 3 + 2] - proxy[j * 3 + 2];
          const d = dr * dr + dg * dg + db * db;
          if (d > 1e-9 && d < best) best = d;
        }
        if (best < Infinity) nn.push(Math.sqrt(best));
      }
      nn.sort((a, b) => a - b);
      const median = nn.length ? nn[nn.length >> 1] : 0;
      spread = Math.min(1, Math.max(0.04, median / Math.sqrt(3)));
    }
    return { colors, count, proxy, lab, spread, luma };
  }

  const presetCache = new Map();
  function presetPalette(preset) {
    if (!presetCache.has(preset.id)) presetCache.set(preset.id, preparePalette(preset.colors, !!preset.luma));
    return presetCache.get(preset.id);
  }

  // ----------------------------------------------------- 5. map + dither ----
  function mapToPalette(base, rgb, pal, dither, strength) {
    const { cols, rows, opaque } = base;
    const n = cols * rows;
    const assign = new Int16Array(n).fill(-1);
    if (!pal.count) return assign;

    // Working buffer: 0–1 sRGB floats (grays in luma mode).
    const work = new Float32Array(n * 3);
    for (let i = 0, p = 0; i < n; i++, p += 3) {
      if (!opaque[i]) continue;
      let r = rgb[p] / 255, g = rgb[p + 1] / 255, b = rgb[p + 2] / 255;
      if (pal.luma) r = g = b = grayOf(r, g, b);
      work[p] = r;
      work[p + 1] = g;
      work[p + 2] = b;
    }

    const tmp = new Float32Array(3);
    const lab = pal.lab;
    const proxy = pal.proxy;

    // Fast 5-bit memoized nearest-color lookup cache (32x32x32 = 32,768 entries)
    const colorCache = new Int16Array(32768).fill(-1);

    const computeNearest = pal.luma
      ? (v) => {
          let best = 0, bestD = Infinity;
          for (let c = 0, o = 0; c < pal.count; c++, o += 3) {
            const d = Math.abs(v - proxy[o]);
            if (d < bestD) { bestD = d; best = c; }
          }
          return best;
        }
      : (r, g, b) => {
          srgbFToOklab(r, g, b, tmp, 0);
          const L = tmp[0], A = tmp[1], B = tmp[2];
          let best = 0, bestD = Infinity;
          for (let c = 0, o = 0; c < pal.count; c++, o += 3) {
            const dl = L - lab[o], da = A - lab[o + 1], db = B - lab[o + 2];
            const d = dl * dl + da * da + db * db;
            if (d < bestD) { bestD = d; best = c; }
          }
          return best;
        };

    const nearest = (r, g, b) => {
      const cr = (clamp01(r) * 31 + 0.5) | 0;
      const cg = (clamp01(g) * 31 + 0.5) | 0;
      const cb = (clamp01(b) * 31 + 0.5) | 0;
      const key = (cr << 10) | (cg << 5) | cb;
      let idx = colorCache[key];
      if (idx === -1) {
        idx = computeNearest(cr / 31, cg / 31, cb / 31);
        colorCache[key] = idx;
      }
      return idx;
    };

    const type = strength > 0 ? dither.type : undefined;

    if (type === 'ordered') {
      // Nudge each cell up/down by a repeating threshold pattern before matching.
      const N = dither.size, M = dither.matrix, NN = N * N;
      const amp = pal.spread * strength;
      for (let y = 0; y < rows; y++) {
        for (let x = 0; x < cols; x++) {
          const i = y * cols + x;
          if (!opaque[i]) continue;
          const off = ((M[(y % N) * N + (x % N)] + 0.5) / NN - 0.5) * amp;
          const p = i * 3;
          assign[i] = nearest(work[p] + off, work[p + 1] + off, work[p + 2] + off);
        }
      }
    } else if (type === 'diffusion') {
      // Push each cell's rounding error onto its unvisited neighbours.
      // Serpentine scan (alternate direction per row) avoids diagonal streaks.
      const K = dither.kernel;
      for (let y = 0; y < rows; y++) {
        const dir = y & 1 ? -1 : 1;
        for (let s = 0; s < cols; s++) {
          const x = dir === 1 ? s : cols - 1 - s;
          const i = y * cols + x;
          if (!opaque[i]) continue;
          const p = i * 3;
          const r = clamp01(work[p]), g = clamp01(work[p + 1]), b = clamp01(work[p + 2]);
          const c = nearest(r, g, b);
          assign[i] = c;
          const er = (r - proxy[c * 3]) * strength;
          const eg = (g - proxy[c * 3 + 1]) * strength;
          const eb = (b - proxy[c * 3 + 2]) * strength;
          for (let k = 0; k < K.length; k++) {
            const nx = x + K[k][0] * dir, ny = y + K[k][1];
            if (nx < 0 || nx >= cols || ny >= rows) continue;
            const j = ny * cols + nx;
            if (!opaque[j]) continue;
            const w = K[k][2];
            work[j * 3] += er * w;
            work[j * 3 + 1] += eg * w;
            work[j * 3 + 2] += eb * w;
          }
        }
      }
    } else {
      for (let i = 0, p = 0; i < n; i++, p += 3) {
        if (opaque[i]) assign[i] = nearest(work[p], work[p + 1], work[p + 2]);
      }
    }
    return assign;
  }

  function countUniqueColors(rgb, opaque) {
    const set = new Set();
    for (let i = 0, p = 0; i < opaque.length; i++, p += 3) {
      if (opaque[i]) set.add((rgb[p] << 16) | (rgb[p + 1] << 8) | rgb[p + 2]);
    }
    return set.size;
  }

  // ---------------------------------------------------------- pipeline ----
  function process() {
    if (!state.source) return;
    const t0 = performance.now();
    const s = state.settings;

    // Downsampling only depends on grid size, so cache it.
    if (!state.cache || state.cache.gridSize !== s.gridSize) {
      const { cols, rows } = gridDimensions(state.source, s.gridSize);
      state.cache = { gridSize: s.gridSize, base: downsample(state.source, cols, rows) };
      state.autoPalCache = null; // invalidates auto palette when grid dimensions change
    }
    const base = state.cache.base;
    const { cols, rows, opaque } = base;
    const rgb = adjust(base, s.saturation, s.contrast);
    const preset = PALETTE_BY_ID[s.palette] || PALETTE_BY_ID.auto;

    const pixels = new Uint8ClampedArray(cols * rows * 4);
    let palette = null;
    let colorCount;

    if (preset.kind === 'full') {
      for (let i = 0; i < opaque.length; i++) {
        if (!opaque[i]) continue;
        pixels[i * 4] = rgb[i * 3];
        pixels[i * 4 + 1] = rgb[i * 3 + 1];
        pixels[i * 4 + 2] = rgb[i * 3 + 2];
        pixels[i * 4 + 3] = 255;
      }
      colorCount = countUniqueColors(rgb, opaque);
    } else {
      let pal;
      if (preset.kind === 'auto') {
        const autoKey = `${s.colorCount}_${s.saturation}_${s.contrast}`;
        if (!state.autoPalCache || state.autoPalCache.key !== autoKey) {
          state.autoPalCache = {
            key: autoKey,
            pal: preparePalette(autoPalette(rgb, opaque, s.colorCount)),
          };
        }
        pal = state.autoPalCache.pal;
      } else {
        pal = presetPalette(preset);
      }

      const assign = mapToPalette(base, rgb, pal, DITHERS[s.dither] || DITHERS.none, s.ditherStrength / 100);
      const usage = new Int32Array(pal.count);
      for (let i = 0; i < assign.length; i++) {
        const ci = assign[i];
        if (ci < 0) continue;
        const c = pal.colors[ci];
        pixels[i * 4] = c.r;
        pixels[i * 4 + 1] = c.g;
        pixels[i * 4 + 2] = c.b;
        pixels[i * 4 + 3] = 255;
        usage[ci]++;
      }
      palette = pal.colors.map((c, i) => ({ ...c, count: usage[i] }));
      if (preset.kind === 'auto') palette = palette.filter((c) => c.count > 0);
      colorCount = palette.filter((c) => c.count > 0).length;
    }

    // The 1-pixel-per-cell image, which we scale up crisply when rendering.
    const tiny = document.createElement('canvas');
    tiny.width = cols;
    tiny.height = rows;
    tiny.getContext('2d').putImageData(new ImageData(pixels, cols, rows), 0, 0);

    state.result = { cols, rows, tiny, palette, colorCount };

    renderDisplay();
    renderPalette();
    updateStats(performance.now() - t0);
  }

  // ------------------------------------------------------------ render ----
  function renderTo(canvas, result, cell) {
    const s = state.settings;
    const W = result.cols * cell;
    const H = result.rows * cell;
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext('2d');
    ctx.imageSmoothingEnabled = false; // nearest neighbour = hard pixel edges
    ctx.clearRect(0, 0, W, H);
    ctx.drawImage(result.tiny, 0, 0, W, H);

    if (s.showGrid && cell >= 4) {
      const lw = Math.max(1, Math.round(cell / 14));
      const half = Math.floor(lw / 2);
      ctx.globalAlpha = s.gridOpacity / 100;
      ctx.fillStyle = s.gridColor;
      ctx.beginPath(); // one path → intersections aren't double-darkened
      for (let x = 0; x <= result.cols; x++) {
        ctx.rect(Math.min(Math.max(x * cell - half, 0), W - lw), 0, lw, H);
      }
      for (let y = 0; y <= result.rows; y++) {
        ctx.rect(0, Math.min(Math.max(y * cell - half, 0), H - lw), W, lw);
      }
      ctx.fill();
      ctx.globalAlpha = 1;
    }
  }

  // Fit the preview to the stage using an integer cell size (in device
  // pixels), so every pixel-art square is exactly the same size on screen.
  function renderDisplay() {
    const r = state.result;
    if (!r) return;
    const dpr = window.devicePixelRatio || 1;
    const box = els.stageBody.getBoundingClientRect();
    const pad = box.width < 600 ? 24 : 56;
    const fitCss = Math.min((box.width - pad) / r.cols, (box.height - pad) / r.rows);
    const cell = Math.max(1, Math.floor(fitCss * dpr));
    renderTo(els.canvas, r, cell);
    els.canvas.style.width = (r.cols * cell) / dpr + 'px';
    els.canvas.style.height = (r.rows * cell) / dpr + 'px';
  }

  let lastPaletteHexList = '';
  function renderPalette() {
    const r = state.result;
    if (!r) {
      els.palette.replaceChildren();
      lastPaletteHexList = '';
      return;
    }
    if (!r.palette) {
      if (lastPaletteHexList !== 'full') {
        els.palette.replaceChildren();
        const p = document.createElement('div');
        p.className = 'palette-empty';
        p.textContent = `Full color mode — ${r.colorCount.toLocaleString()} unique colors.`;
        els.palette.append(p);
        lastPaletteHexList = 'full';
      }
      return;
    }

    const total = r.palette.reduce((a, c) => a + c.count, 0) || 1;
    const sorted = r.palette
      .map((c) => ({ ...c, L: lightness(c) }))
      .sort((a, b) => a.L - b.L);

    const currentHexList = sorted.map((c) => toHex(c.r, c.g, c.b)).join(',');
    const swatches = els.palette.querySelectorAll('.swatch');

    // If swatch elements match current palette, update stats in place without DOM re-creation
    if (currentHexList === lastPaletteHexList && swatches.length === sorted.length) {
      sorted.forEach((c, i) => {
        const sw = swatches[i];
        const hex = toHex(c.r, c.g, c.b);
        sw.className = 'swatch' + (c.count ? '' : ' is-unused');
        sw.title = `${hex.toUpperCase()} · ${((c.count / total) * 100).toFixed(1)}%`;
      });
      return;
    }

    // Reconstruct swatches only when colors actually change
    lastPaletteHexList = currentHexList;
    els.palette.replaceChildren();
    const frag = document.createDocumentFragment();
    sorted.forEach((c) => {
      const hex = toHex(c.r, c.g, c.b);
      const sw = document.createElement('button');
      sw.type = 'button';
      sw.className = 'swatch' + (c.count ? '' : ' is-unused');
      sw.style.background = hex;
      sw.title = `${hex.toUpperCase()} · ${((c.count / total) * 100).toFixed(1)}%`;
      sw.setAttribute('aria-label', `Copy ${hex}`);
      sw.addEventListener('click', () => copyHex(hex));
      frag.append(sw);
    });
    els.palette.append(frag);
  }

  function updateStats(ms) {
    const r = state.result;
    els.statGrid.textContent = r ? `${r.cols}×${r.rows}` : '—';
    els.statColors.textContent = r ? r.colorCount.toLocaleString() : '—';
    els.statTime.textContent = ms != null ? `${Math.max(1, Math.round(ms))}ms` : '—';
    if (r) els.gridDimsHint.textContent = `${r.cols} × ${r.rows} = ${(r.cols * r.rows).toLocaleString()} pixels. Fewer = chunkier.`;
    updateExportHint();
  }

  function exportCellSize() {
    const r = state.result;
    if (!r) return state.settings.exportScale;
    const maxCell = Math.floor(MAX_EXPORT_SIDE / Math.max(r.cols, r.rows));
    return Math.max(1, Math.min(state.settings.exportScale, maxCell));
  }

  function updateExportHint() {
    const r = state.result;
    if (!r) {
      els.exportSizeHint.textContent = 'Output: —';
      return;
    }
    const cell = exportCellSize();
    const capped = cell < state.settings.exportScale ? ' (capped)' : '';
    els.exportSizeHint.textContent = `Output: ${r.cols * cell} × ${r.rows * cell} px${capped}`;
  }

  // ----------------------------------------------------------- loading ----
  function loadFile(file) {
    if (!file || !file.type.startsWith('image/')) {
      toast('That doesn\u2019t look like an image file.');
      return;
    }
    if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
    state.objectUrl = URL.createObjectURL(file);
    loadUrl(state.objectUrl, file.name);
  }

  function loadUrl(url, name, { silent = false } = {}) {
    const img = new Image();
    img.decoding = 'async';
    img.onload = () => {
      try {
        state.source = prepareSource(img);
      } catch (err) {
        // Happens when opening via file:// in some browsers (tainted canvas).
        if (!silent) toast('Could not read that image.');
        console.warn(err);
        return;
      }
      state.cache = null;
      state.autoPalCache = null;
      lastPaletteHexList = '';
      state.fileName = (name || 'image').replace(/\.[^.]+$/, '') || 'image';
      els.original.src = url;
      document.body.classList.add('has-image');
      els.downloadBtn.disabled = false;
      els.compareBtn.disabled = false;
      // Wait a frame so the stage has its final layout before fitting.
      requestAnimationFrame(process);
    };
    img.onerror = () => {
      if (!silent) toast('Could not load that image.');
    };
    img.src = url;
  }

  // ---------------------------------------------------------- download ----
  function download() {
    const r = state.result;
    if (!r) return;
    const out = document.createElement('canvas');
    renderTo(out, r, exportCellSize());
    out.toBlob((blob) => {
      if (!blob) return toast('Export failed — try a smaller pixel size.');
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `${state.fileName}-pixel-${r.cols}x${r.rows}.png`;
      document.body.append(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 2000);
      toast('Pixel art downloaded ✨');
    }, 'image/png');
  }

  // ------------------------------------------------------------- utils ----
  let toastTimer;
  function toast(msg) {
    els.toast.textContent = msg;
    els.toast.classList.add('is-visible');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => els.toast.classList.remove('is-visible'), 2400);
  }

  async function copyHex(hex) {
    try {
      await navigator.clipboard.writeText(hex.toUpperCase());
      toast(`Copied ${hex.toUpperCase()}`);
    } catch {
      toast(hex.toUpperCase());
    }
  }

  function setRangeFill(input) {
    const min = +input.min, max = +input.max;
    input.style.setProperty('--fill', `${((input.value - min) / (max - min)) * 100}%`);
  }

  // Coalesce rapid slider input into one update per animation frame.
  function rafThrottle(fn) {
    let queued = false;
    return () => {
      if (queued) return;
      queued = true;
      requestAnimationFrame(() => {
        queued = false;
        fn();
      });
    };
  }
  const scheduleProcess = rafThrottle(process);
  const scheduleRender = rafThrottle(renderDisplay);

  // ------------------------------------------------- build pickers (UI) ----
  function buildPalettePicker() {
    const frag = document.createDocumentFragment();
    for (const p of PALETTES) {
      const card = document.createElement('button');
      card.type = 'button';
      card.className = 'pal-card';
      card.dataset.id = p.id;
      card.title = p.desc;

      const strip = document.createElement('span');
      strip.className = 'pal-strip';
      if (p.kind === 'preset') {
        [...p.colors]
          .sort((a, b) => lightness(a) - lightness(b))
          .forEach((c) => {
            const i = document.createElement('i');
            i.style.background = toHex(c.r, c.g, c.b);
            strip.append(i);
          });
      } else {
        strip.classList.add(`pal-strip-${p.kind}`);
      }

      const name = document.createElement('span');
      name.className = 'pal-name';
      name.textContent = p.name;

      const meta = document.createElement('span');
      meta.className = 'pal-meta';
      meta.textContent = p.kind === 'preset' ? `${p.colors.length} colors` : p.kind === 'auto' ? 'From your image' : 'No limit';

      card.append(strip, name, meta);
      frag.append(card);
    }
    els.palettePicker.append(frag);
  }

  function buildDitherOptions() {
    const frag = document.createDocumentFragment();
    for (const [id, d] of Object.entries(DITHERS)) {
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.className = 'chip';
      chip.dataset.id = id;
      chip.textContent = d.chip;
      chip.title = d.name;
      frag.append(chip);
    }
    els.ditherOptions.append(frag);
  }

  // ------------------------------------------------------ UI sync/bind ----
  function syncUI() {
    const s = state.settings;
    const preset = PALETTE_BY_ID[s.palette] || PALETTE_BY_ID.auto;
    const dither = DITHERS[s.dither] || DITHERS.none;
    const fullColor = preset.kind === 'full';

    els.gridSize.value = s.gridSize;
    els.colorCount.value = s.colorCount;
    els.ditherStrength.value = s.ditherStrength;
    els.saturation.value = s.saturation;
    els.contrast.value = s.contrast;
    els.showGrid.checked = s.showGrid;
    els.gridColor.value = s.gridColor;
    els.gridOpacity.value = s.gridOpacity;
    els.exportScale.value = s.exportScale;

    els.gridSizeValue.textContent = s.gridSize;
    els.colorCountValue.textContent = s.colorCount;
    els.ditherStrengthValue.textContent = `${s.ditherStrength}%`;
    els.saturationValue.textContent = `${s.saturation}%`;
    els.contrastValue.textContent = `${s.contrast}%`;
    els.gridOpacityValue.textContent = `${s.gridOpacity}%`;
    els.exportScaleValue.textContent = `${s.exportScale}px`;

    // Palette
    els.palettePicker.querySelectorAll('.pal-card').forEach((card) => {
      const on = card.dataset.id === preset.id;
      card.classList.toggle('is-active', on);
      card.setAttribute('aria-pressed', on);
    });
    els.paletteDesc.textContent = preset.desc;
    els.colorCountField.hidden = preset.kind !== 'auto';

    // Dithering (needs a limited palette to have any effect)
    els.ditherOptions.querySelectorAll('.chip').forEach((chip) => {
      const on = chip.dataset.id === s.dither;
      chip.classList.toggle('is-active', on);
      chip.setAttribute('aria-pressed', on);
    });
    els.ditherFields.classList.toggle('is-disabled', fullColor);
    els.ditherStrengthField.classList.toggle('is-disabled', fullColor || !dither.type);
    els.ditherDesc.textContent = fullColor ? 'Pick a limited palette to use dithering.' : dither.desc;

    // Grid
    els.gridLineFields.classList.toggle('is-disabled', !s.showGrid);
    els.gridPresets.querySelectorAll('.chip').forEach((chip) => {
      chip.classList.toggle('is-active', +chip.dataset.size === s.gridSize);
    });

    document.querySelectorAll('input[type="range"]').forEach(setRangeFill);
    updateExportHint();
  }

  // [element, settings key, parser, what to redo]
  const bindings = [
    [els.gridSize, 'gridSize', Number, 'process'],
    [els.colorCount, 'colorCount', Number, 'process'],
    [els.ditherStrength, 'ditherStrength', Number, 'process'],
    [els.saturation, 'saturation', Number, 'process'],
    [els.contrast, 'contrast', Number, 'process'],
    [els.showGrid, 'showGrid', null, 'render'],
    [els.gridColor, 'gridColor', String, 'render'],
    [els.gridOpacity, 'gridOpacity', Number, 'render'],
    [els.exportScale, 'exportScale', Number, 'none'],
  ];

  bindings.forEach(([el, key, parse, action]) => {
    el.addEventListener('input', () => {
      state.settings[key] = el.type === 'checkbox' ? el.checked : parse(el.value);
      syncUI();
      if (action === 'process') scheduleProcess();
      else if (action === 'render') scheduleRender();
    });
  });

  // Clicking a chip/card inside `container` sets `key` to its data attribute.
  function bindChoice(container, selector, key, attr, parse = String) {
    container.addEventListener('click', (e) => {
      const el = e.target.closest(selector);
      if (!el) return;
      state.settings[key] = parse(el.dataset[attr]);
      syncUI();
      scheduleProcess();
    });
  }
  bindChoice(els.gridPresets, '.chip', 'gridSize', 'size', Number);
  bindChoice(els.palettePicker, '.pal-card', 'palette', 'id');
  bindChoice(els.ditherOptions, '.chip', 'dither', 'id');

  els.resetBtn.addEventListener('click', () => {
    state.settings = { ...DEFAULTS };
    syncUI();
    scheduleProcess();
    toast('Settings reset');
  });

  // File picking
  const openPicker = () => els.fileInput.click();
  els.uploadBtn.addEventListener('click', openPicker);
  els.dropzone.addEventListener('click', openPicker);
  els.dropzone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      openPicker();
    }
  });
  els.fileInput.addEventListener('change', () => {
    loadFile(els.fileInput.files[0]);
    els.fileInput.value = '';
  });

  els.downloadBtn.addEventListener('click', download);

  // Drag & drop anywhere on the page
  let dragDepth = 0;
  const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes('Files');
  window.addEventListener('dragenter', (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    dragDepth++;
    document.body.classList.add('is-dragging');
  });
  window.addEventListener('dragover', (e) => {
    if (hasFiles(e)) e.preventDefault();
  });
  window.addEventListener('dragleave', () => {
    dragDepth = Math.max(0, dragDepth - 1);
    if (!dragDepth) document.body.classList.remove('is-dragging');
  });
  window.addEventListener('drop', (e) => {
    e.preventDefault();
    dragDepth = 0;
    document.body.classList.remove('is-dragging');
    const file = [...(e.dataTransfer?.files || [])].find((f) => f.type.startsWith('image/'));
    if (file) loadFile(file);
    else toast('Drop an image file (PNG, JPG, WebP, GIF…)');
  });

  // Paste from clipboard
  window.addEventListener('paste', (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith('image/'));
    if (item) loadFile(item.getAsFile());
  });

  // Compare with original (press & hold the button, or hold "C")
  const setComparing = (on) => {
    if (!state.result) return;
    els.canvasWrap.classList.toggle('is-comparing', on);
    els.compareBtn.classList.toggle('is-active', on);
  };
  els.compareBtn.addEventListener('pointerdown', () => setComparing(true));
  ['pointerup', 'pointerleave', 'pointercancel'].forEach((ev) =>
    els.compareBtn.addEventListener(ev, () => setComparing(false))
  );
  window.addEventListener('keydown', (e) => {
    const typing = e.target.isContentEditable || (e.target.tagName === 'INPUT' && e.target.type === 'text');
    if (e.key.toLowerCase() === 'c' && !e.repeat && !e.ctrlKey && !e.metaKey && !typing) {
      setComparing(true);
    }
  });
  window.addEventListener('keyup', (e) => {
    if (e.key.toLowerCase() === 'c') setComparing(false);
  });

  // Keep the preview fitted to the stage
  new ResizeObserver(scheduleRender).observe(els.stageBody);

  // ------------------------------------------------------------- start ----
  buildPalettePicker();
  buildDitherOptions();
  syncUI();
  loadUrl(SAMPLE_URL, 'sample', { silent: true });
})();
