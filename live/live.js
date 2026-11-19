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

function makeCanvases(outEl, labels) {
  outEl.replaceChildren();
  const ctxs = [];
  for (let i = 0; i < labels.length; i++) {
    const c = document.createElement("canvas");
    c.width = 32; c.height = 32;
    c.style.width = "128px"; c.style.height = "128px";
    c.style.imageRendering = "pixelated";
    c.style.borderRadius = "4px";
    const ctx = c.getContext("2d");
    ctxs.push(ctx);
    const wrap = document.createElement("div");
    wrap.appendChild(c);
    const cap = document.createElement("div");
    cap.className = "muted";
    cap.style.fontSize = "11px";
    cap.textContent = (labels[i] === 0 ? "cat" : "dog") + " #" + i;
    wrap.appendChild(cap);
    outEl.appendChild(wrap);
  }
  return ctxs;
}

function paintAll(ctxs, img) {
  const d = ctxs.map((ctx) => ctx.createImageData(32, 32));
  for (let i = 0; i < ctxs.length; i++) {
    for (let p = 0; p < 32 * 32; p++) {
      d[i].data[p * 4] = Math.round(img[i * 3072 + p] * 255);
      d[i].data[p * 4 + 1] = Math.round(img[i * 3072 + 1024 + p] * 255);
      d[i].data[p * 4 + 2] = Math.round(img[i * 3072 + 2048 + p] * 255);
      d[i].data[p * 4 + 3] = 255;
    }
    ctxs[i].putImageData(d[i], 0, 0);
  }
}

async function runPanel(model, labels, seed, w) {
  const status = $("#status-" + model);
  const bar = $("#bar-" + model);
  const ctxs = makeCanvases($("#out-" + model), labels);
  bar.style.width = "0%";
  const sess = await getSession(model);
  const t0 = performance.now();
  status.textContent = `running Heun 18 steps on ${epUsed}...`;
  const img = await generate(ort, sess, labels, seed, w, 18,
    (nfe, totalNfe) => { bar.style.width = (100 * nfe / totalNfe).toFixed(1) + "%"; },
    window.__noiseOverride || null,
    (i, steps, preview, sigma) => {
      paintAll(ctxs, preview);
      status.textContent = i < 0 ? "initial noise"
        : `step ${i + 1}/${steps} · sigma=${sigma < 0.01 ? sigma.toExponential(1) : sigma.toFixed(2)}`;
    });
  paintAll(ctxs, img);
  bar.style.width = "100%";
  status.textContent =
    `${labels.length} image(s) in ${((performance.now() - t0) / 1000).toFixed(1)} s on ${epUsed} · w=${w}`;
  return img;
}

async function run() {
  const species = $("#lv-species").value;
  const seed = parseInt($("#lv-seed").value || "0", 10);
  const n = parseInt($("#lv-count").value, 10);
  const labels = species === "cat" ? Array(n).fill(0)
    : species === "dog" ? Array(n).fill(1)
      : Array.from({ length: n }, (_, i) => i % 2);
  window.__liveLabels = labels;
  $("#lv-go").disabled = true;
  $("#lv-status").textContent = "generating main, then branch (same seed, same labels)...";
  try {
    const wMain = parseFloat($("#w-main").value);
    const wBranch = parseFloat($("#w-branch").value);
    const imgMain = await runPanel("main", labels, seed, wMain);
    const imgBranch = await runPanel("branch", labels, seed, wBranch);
    window.__lastLive = { main: imgMain, branch: imgBranch };
    $("#lv-status").textContent = `done · seed ${seed} · main w=${wMain} · branch w=${wBranch}`;
  } catch (e) {
    $("#lv-status").textContent = "error: " + e;
  }
  $("#lv-go").disabled = false;
}

document.addEventListener("DOMContentLoaded", () => {
  $("#lv-go").addEventListener("click", run);
  for (const m of ["main", "branch"]) {
    $("#w-" + m).addEventListener("input", () => { $("#wval-" + m).textContent = $("#w-" + m).value; });
  }
});