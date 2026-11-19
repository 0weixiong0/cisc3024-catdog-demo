import { generate } from "./sampler.js";

const $ = (s) => document.querySelector(s);
const sessions = {};
let epUsed = "";

async function getSession(model) {
  if (sessions[model]) return sessions[model];
  $("#lv-status").textContent = `downloading ${model}.onnx (38.7 MB, cached after first load)...`;
  const url = `live/weights/${model}.onnx`;
  ort.env.wasm.wasmPaths = new URL("vendor/", import.meta.url).href;
  ort.env.wasm.numThreads = 2;
  let sess;
  try {
    sess = await ort.InferenceSession.create(url, { executionProviders: ["webgpu", "wasm"] });
    epUsed = "webgpu";
  } catch (e) {
    sess = await ort.InferenceSession.create(url, { executionProviders: ["wasm"] });
    epUsed = "wasm";
  }
  sessions[model] = sess;
  return sess;
}

function paint(img, b) {
  const out = $("#lv-out");
  out.replaceChildren();
  for (let i = 0; i < b; i++) {
    const c = document.createElement("canvas");
    c.width = 32; c.height = 32;
    c.style.width = "128px"; c.style.height = "128px";
    c.style.imageRendering = "pixelated";
    c.style.borderRadius = "4px";
    const ctx = c.getContext("2d");
    const d = ctx.createImageData(32, 32);
    for (let p = 0; p < 32 * 32; p++) {
      d.data[p * 4] = Math.round(img[i * 3072 + p] * 255);
      d.data[p * 4 + 1] = Math.round(img[i * 3072 + 1024 + p] * 255);
      d.data[p * 4 + 2] = Math.round(img[i * 3072 + 2048 + p] * 255);
      d.data[p * 4 + 3] = 255;
    }
    ctx.putImageData(d, 0, 0);
    const wrap = document.createElement("div");
    wrap.appendChild(c);
    const cap = document.createElement("div");
    cap.className = "muted";
    cap.style.fontSize = "11px";
    cap.textContent = (window.__liveLabels[i] === 0 ? "cat" : "dog") + " #" + i;
    wrap.appendChild(cap);
    out.appendChild(wrap);
  }
}

async function run() {
  const model = $("#lv-model").value;
  const species = $("#lv-species").value;
  const seed = parseInt($("#lv-seed").value || "0", 10);
  const w = parseFloat($("#lv-w").value);
  const n = parseInt($("#lv-count").value, 10);
  const labels = species === "cat" ? Array(n).fill(0)
    : species === "dog" ? Array(n).fill(1)
      : Array.from({ length: n }, (_, i) => i % 2);
  window.__liveLabels = labels;
  $("#lv-go").disabled = true;
  const t0 = performance.now();
  try {
    const sess = await getSession(model);
    $("#lv-status").textContent = `running Heun 18 steps on ${epUsed}...`;
    const img = await generate(ort, sess, labels, seed, w, 18, (nfe, total) => {
      $("#lv-bar").style.width = (100 * nfe / total).toFixed(1) + "%";
    }, window.__noiseOverride || null);
    window.__lastLive = img;
    paint(img, n);
    $("#lv-status").textContent =
      `${n} image(s) in ${((performance.now() - t0) / 1000).toFixed(1)} s on ${epUsed} · seed ${seed} · w=${w}`;
  } catch (e) {
    $("#lv-status").textContent = "error: " + e;
  }
  $("#lv-go").disabled = false;
}

document.addEventListener("DOMContentLoaded", () => {
  $("#lv-go").addEventListener("click", run);
  $("#lv-w").addEventListener("input", () => { $("#lv-wval").textContent = $("#lv-w").value; });
});