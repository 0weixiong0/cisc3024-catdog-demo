// Faithful JS port of src/training/diffusion.py sample(): VE sigma schedule
// (rho=7), EDM preconditioning, 18 Heun steps, CFG when guidance != 1.
// The ONNX graph covers UNet.forward(x, t, labels) only.
// Initial noise uses a browser PRNG (mulberry32 + Box-Muller), NOT torch's
// MT19937: deterministic within the demo, same distribution as torch seeds.
export const SIGMA_MIN = 0.002, SIGMA_MAX = 80.0, RHO = 7.0, SIGMA_DATA = 0.557;

export function sigmaSchedule(steps) {
  const a = Math.pow(SIGMA_MAX, 1 / RHO), b = Math.pow(SIGMA_MIN, 1 / RHO);
  const s = [];
  for (let i = 0; i < steps; i++) s.push(Math.pow(a + (i / (steps - 1)) * (b - a), RHO));
  s.push(0);
  return s;
}

export function mulberry32(seed) {
  let t = seed >>> 0;
  return () => {
    t += 0x6D2B79F5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

export function initialNoise(b, seed) {
  const rnd = mulberry32(seed);
  const n = b * 3 * 32 * 32;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    const u1 = Math.max(rnd(), 1e-12), u2 = rnd();
    const r = Math.sqrt(-2 * Math.log(u1)), th = 2 * Math.PI * u2;
    out[i] = r * Math.cos(th);
    if (i + 1 < n) out[i + 1] = r * Math.sin(th);
  }
  return out;
}

function precondition(sigma) {
  const s2 = sigma * sigma + SIGMA_DATA * SIGMA_DATA;
  return {
    c_skip: SIGMA_DATA * SIGMA_DATA / s2,
    c_out: sigma * SIGMA_DATA / Math.sqrt(s2),
    c_in: 1 / Math.sqrt(s2),
    c_noise: Math.log(sigma) / 4
  };
}

export async function generate(ort, session, labels, seed, guidance, steps = 18, onProgress, noiseOverride) {
  const b = labels.length;
  const sched = sigmaSchedule(steps);
  const x = noiseOverride ? Float32Array.from(noiseOverride) : initialNoise(b, seed);
  for (let i = 0; i < x.length; i++) x[i] *= sched[0];
  const uncond = labels.map(() => 2);
  let nfe = 0;
  const total = steps * (guidance === 1 ? 1 : 2) * 2 - (guidance === 1 ? 1 : 2);

  async function denoise(xCur, sigma, lab) {
    const p = precondition(sigma);
    const xin = new Float32Array(xCur.length);
    for (let i = 0; i < xin.length; i++) xin[i] = xCur[i] * p.c_in;
    const res = await session.run({
      x: new ort.Tensor("float32", xin, [b, 3, 32, 32]),
      t: new ort.Tensor("float32", new Float32Array(b).fill(p.c_noise), [b]),
      labels: new ort.Tensor("int64", BigInt64Array.from(lab.map(BigInt)), [b])
    });
    const o = res.out.data;
    const d = new Float32Array(xCur.length);
    for (let i = 0; i < d.length; i++) d[i] = p.c_skip * xCur[i] + p.c_out * o[i];
    nfe++;
    if (onProgress) onProgress(nfe, total);
    return d;
  }
  async function d(xCur, sigma) {
    if (guidance === 1) return denoise(xCur, sigma, labels);
    const dc = await denoise(xCur, sigma, labels);
    const du = await denoise(xCur, sigma, uncond);
    const out = new Float32Array(dc.length);
    for (let i = 0; i < out.length; i++) out[i] = du[i] + guidance * (dc[i] - du[i]);
    return out;
  }

  for (let i = 0; i < steps; i++) {
    const s = sched[i], sNext = sched[i + 1];
    const dCur = await d(x, s);
    const xEuler = new Float32Array(x.length);
    for (let j = 0; j < x.length; j++) xEuler[j] = x[j] + (sNext - s) * (x[j] - dCur[j]) / s;
    if (sNext > 0) {
      const dNext = await d(xEuler, sNext);
      for (let j = 0; j < x.length; j++)
        x[j] = x[j] + (sNext - s) * ((x[j] - dCur[j]) / s + (xEuler[j] - dNext[j]) / sNext) / 2;
    } else {
      x.set(xEuler);
    }
  }
  const img = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) img[i] = Math.min(1, Math.max(0, (x[i] + 1) / 2));
  return img;
}