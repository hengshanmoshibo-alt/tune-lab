const samples = [
  "房间有点小，设施也不算新，不过位置很方便，卫生干净，服务态度很好，整体住下来还是挺满意的。",
  "包装一般，说明书写得不够清楚，但机器用起来很稳定，速度快，价格也合适，总体值得购买。",
];

const el = {
  text: document.querySelector("#text"),
  samples: document.querySelector("#samples"),
  status: document.querySelector("#status"),
  baseAcc: document.querySelector("#baseAcc"),
  loraAcc: document.querySelector("#loraAcc"),
  f1Gain: document.querySelector("#f1Gain"),
  runLora: document.querySelector("#runLora"),
  runBase: document.querySelector("#runBase"),
  runBoth: document.querySelector("#runBoth"),
  loraLabel: document.querySelector("#loraLabel"),
  loraRaw: document.querySelector("#loraRaw"),
  loraTime: document.querySelector("#loraTime"),
  baseLabel: document.querySelector("#baseLabel"),
  baseRaw: document.querySelector("#baseRaw"),
  baseTime: document.querySelector("#baseTime"),
};

function pct(value) {
  return `${(value * 100).toFixed(2)}%`;
}

function setBusy(isBusy) {
  el.runLora.disabled = isBusy;
  el.runBase.disabled = isBusy;
  el.runBoth.disabled = isBusy;
}

function renderResult(mode, data) {
  const prefix = mode === "lora" ? "lora" : "base";
  el[`${prefix}Label`].textContent = data.label || "无法判断";
  el[`${prefix}Raw`].textContent = data.raw_output || "无输出";
  el[`${prefix}Time`].textContent = `用时 ${data.latency_sec.toFixed(2)}s`;
}

async function predict(mode) {
  const text = el.text.value.trim();
  if (!text) return;
  const response = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, mode }),
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message);
  }
  const data = await response.json();
  renderResult(mode, data);
}

async function run(mode) {
  setBusy(true);
  el.status.textContent = "推理中";
  try {
    if (mode === "both") {
      await predict("lora");
      await predict("base");
    } else {
      await predict(mode);
    }
    el.status.textContent = "就绪";
  } catch (error) {
    el.status.textContent = "出错";
    console.error(error);
    alert("推理失败，请查看服务日志。");
  } finally {
    setBusy(false);
  }
}

async function loadHealth() {
  const response = await fetch("/api/health");
  const data = await response.json();
  el.status.textContent = data.cuda ? "GPU 就绪" : "CPU";
  const base = data.metrics?.base;
  const lora = data.metrics?.lora;
  if (base && lora) {
    el.baseAcc.textContent = pct(base.accuracy);
    el.loraAcc.textContent = pct(lora.accuracy);
    el.f1Gain.textContent = `+${((lora.macro_f1 - base.macro_f1) * 100).toFixed(2)}pp`;
  }
}

samples.forEach((sample) => {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = sample.slice(0, 18);
  button.title = sample;
  button.addEventListener("click", () => {
    el.text.value = sample;
  });
  el.samples.appendChild(button);
});

el.runLora.addEventListener("click", () => run("lora"));
el.runBase.addEventListener("click", () => run("base"));
el.runBoth.addEventListener("click", () => run("both"));

loadHealth().catch(() => {
  el.status.textContent = "未连接";
});
