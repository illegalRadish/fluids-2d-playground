import { FluidSim } from "./fluid.js";

const canvas = document.getElementById("fluid");
const err = document.getElementById("error");

let sim;
try {
  sim = new FluidSim(canvas);
} catch (e) {
  err.textContent = e.message || String(e);
  err.hidden = false;
  throw e;
}

let hue = 0;
let lastX = 0;
let lastY = 0;
let drawing = false;
/** Dye brightness multiplier (from settings panel). */
let colorIntensity = 4.5;

function normCoords(clientX, clientY) {
  const r = canvas.getBoundingClientRect();
  const x = (clientX - r.left) / r.width;
  const y = 1 - (clientY - r.top) / r.height;
  return { x, y };
}

function hslToRgb(h, s, l) {
  h = ((h % 360) + 360) % 360;
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = l - c / 2;
  let r = 0;
  let g = 0;
  let b = 0;
  if (h < 60) {
    r = c;
    g = x;
  } else if (h < 120) {
    r = x;
    g = c;
  } else if (h < 180) {
    g = c;
    b = x;
  } else if (h < 240) {
    g = x;
    b = c;
  } else if (h < 300) {
    r = x;
    b = c;
  } else {
    r = c;
    b = x;
  }
  return [r + m, g + m, b + m];
}

function splatAt(x, y, dx, dy) {
  hue = (hue + 3.4 + Math.abs(dx + dy) * 80) % 360;
  const sat = 0.92 + 0.08 * Math.sin(hue * 0.07);
  const light = 0.5 + 0.12 * Math.cos(hue * 0.11);
  const [r, g, b] = hslToRgb(hue, sat, light);
  const k = colorIntensity;
  sim.splat(x, y, dx, dy, [r * k, g * k, b * k]);
}

function onPointerDown(e) {
  drawing = true;
  canvas.setPointerCapture(e.pointerId);
  const { x, y } = normCoords(e.clientX, e.clientY);
  lastX = x;
  lastY = y;
}

function onPointerMove(e) {
  if (!drawing && e.buttons === 0) return;
  const { x, y } = normCoords(e.clientX, e.clientY);
  const dx = x - lastX;
  const dy = y - lastY;
  lastX = x;
  lastY = y;
  if (Math.abs(dx) + Math.abs(dy) > 1e-6) {
    splatAt(x, y, dx, dy);
  }
}

function onPointerUp(e) {
  drawing = false;
  if (canvas.hasPointerCapture?.(e.pointerId)) {
    canvas.releasePointerCapture(e.pointerId);
  }
}

canvas.addEventListener("pointerdown", onPointerDown);
canvas.addEventListener("pointermove", onPointerMove);
window.addEventListener("pointerup", onPointerUp);
canvas.addEventListener("pointerleave", onPointerUp);

function wireControl(id, apply, format) {
  const input = document.getElementById(id);
  const valEl = document.getElementById(`${id}-val`);
  if (!input || !valEl) return;
  const sync = () => {
    const v = apply.read(input);
    apply.write(v);
    valEl.textContent = format(v);
  };
  input.addEventListener("input", sync);
  sync();
}

function initSettingsPanel() {
  wireControl("simSize", {
    read: (el) => Number(el.value),
    write: (v) => sim.setSimSize(v),
  }, String);

  wireControl("pressureIterations", {
    read: (el) => Number(el.value),
    write: (v) => {
      sim.pressureIterations = v;
    },
  }, (v) => String(Math.round(v)));

  wireControl("vorticityCurl", {
    read: (el) => Number(el.value),
    write: (v) => {
      sim.vorticityCurl = v;
    },
  }, (v) => String(Math.round(v)));

  wireControl("velocityDissipation", {
    read: (el) => Number(el.value) / 1000,
    write: (v) => {
      sim.velocityDissipation = v;
    },
  }, (v) => v.toFixed(2));

  wireControl("dyeDissipation", {
    read: (el) => Number(el.value) / 1000,
    write: (v) => {
      sim.dyeDissipation = v;
    },
  }, (v) => v.toFixed(2));

  wireControl("splatForce", {
    read: (el) => Number(el.value),
    write: (v) => {
      sim.splatForce = v;
    },
  }, (v) => String(Math.round(v)));

  wireControl("strokeSize", {
    read: (el) => Number(el.value),
    write: (v) => {
      sim.strokeSize = v;
    },
  }, (v) => String(Math.round(v)));

  wireControl("colorIntensity", {
    read: (el) => Number(el.value) / 100,
    write: (v) => {
      colorIntensity = v;
    },
  }, (v) => v.toFixed(1));

  wireControl("bloomThreshold", {
    read: (el) => Number(el.value) / 100,
    write: (v) => {
      sim.bloomThreshold = v;
    },
  }, (v) => v.toFixed(2));

  wireControl("bloomSoftKnee", {
    read: (el) => Number(el.value) / 100,
    write: (v) => {
      sim.bloomSoftKnee = v;
    },
  }, (v) => v.toFixed(2));

  wireControl("bloomIntensity", {
    read: (el) => Number(el.value) / 100,
    write: (v) => {
      sim.bloomIntensity = v;
    },
  }, (v) => v.toFixed(2));

  const clearBtn = document.getElementById("clear-fluid");
  if (clearBtn) {
    clearBtn.addEventListener("click", () => sim.clear());
  }
}

initSettingsPanel();

function tick() {
  sim.frame();
  requestAnimationFrame(tick);
}

requestAnimationFrame(tick);
