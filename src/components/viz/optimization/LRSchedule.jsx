import { useMemo, useState } from "react";
import { fmt, clamp } from "../scales";
import { Slider, SegmentedControl, Stat } from "../ui";
import LineChart from "../LineChart";

/**
 * Two deterministic demos behind a SegmentedControl.
 *
 * 1) "LR schedule" — a linear-warmup-then-cosine-decay learning-rate schedule,
 *    plotted with the shared LineChart. Four sliders (warmup_steps, max_lr,
 *    total_steps, min_lr) reshape the curve live. The math matches the PyTorch
 *    LambdaLR formula in the chapter exactly.
 *
 * 2) "Optimizers" — three optimizers (SGD, Momentum, Adam) descending the same
 *    2D quadratic "bowl" with an anisotropic (ravine-shaped) curvature. Their
 *    stepped paths are computed analytically from a fixed start, so the demo is
 *    fully deterministic — no randomness, no animation clock required. You can
 *    see SGD oscillate across the ravine, Momentum overshoot then settle, and
 *    Adam march down a near-straight, normalized path.
 */

// ----- 1) learning-rate schedule ------------------------------------------

function lrAt(step, warmup, total, maxLr, minLr) {
  if (step < warmup) {
    // linear warmup from 0 -> maxLr
    return maxLr * (step / Math.max(1, warmup));
  }
  if (step > total) return minLr;
  // cosine decay from maxLr -> minLr over [warmup, total]
  const progress = (step - warmup) / Math.max(1, total - warmup);
  const cos = 0.5 * (1 + Math.cos(Math.PI * progress));
  return minLr + (maxLr - minLr) * cos;
}

function ScheduleDemo() {
  const [warmup, setWarmup] = useState(500);
  const [maxLr, setMaxLr] = useState(3.0); // displayed in units of 1e-4
  const [total, setTotal] = useState(5000);
  const [minLr, setMinLr] = useState(0.3); // units of 1e-4

  const warmupC = clamp(warmup, 0, total);
  const points = useMemo(() => {
    const pts = [];
    const N = 160;
    for (let i = 0; i <= N; i++) {
      const step = (i / N) * total;
      pts.push([step, lrAt(step, warmupC, total, maxLr, minLr)]);
    }
    return pts;
  }, [warmupC, total, maxLr, minLr]);

  const peakLr = maxLr;
  const finalLr = minLr;

  return (
    <div className="space-y-5">
      <LineChart
        height={300}
        xLabel="training step"
        yLabel="learning rate (×10⁻⁴)"
        series={[{ label: "lr(t)", color: "#22d3ee", points }]}
        annotations={[
          { x: warmupC, label: "warmup ends", color: "#f59e0b" },
        ]}
        fmtX={(v) => (v >= 1000 ? `${(v / 1000).toFixed(1)}k` : v.toFixed(0))}
        fmtY={(v) => v.toFixed(2)}
      />

      <div className="grid gap-x-6 gap-y-4 sm:grid-cols-2">
        <Slider
          label="warmup_steps"
          min={0}
          max={2000}
          step={50}
          value={warmup}
          onChange={setWarmup}
          format={(v) => v.toFixed(0)}
        />
        <Slider
          label="total_steps"
          min={1000}
          max={10000}
          step={250}
          value={total}
          onChange={setTotal}
          accent="cyan"
          format={(v) => `${(v / 1000).toFixed(2)}k`}
        />
        <Slider
          label="max_lr (×10⁻⁴)"
          min={0.5}
          max={10}
          step={0.1}
          value={maxLr}
          onChange={setMaxLr}
          accent="violet"
          format={(v) => `${v.toFixed(1)}e-4`}
        />
        <Slider
          label="min_lr (×10⁻⁴)"
          min={0}
          max={2}
          step={0.05}
          value={minLr}
          onChange={setMinLr}
          accent="violet"
          format={(v) => `${v.toFixed(2)}e-4`}
        />
      </div>

      <div className="grid grid-cols-3 gap-3">
        <Stat label="peak lr" value={`${fmt(peakLr, 1)}e-4`} accent="text-cyan-200" />
        <Stat label="warmup %" value={`${((warmupC / total) * 100).toFixed(1)}%`} accent="text-amber-200" />
        <Stat label="final lr" value={`${fmt(finalLr, 2)}e-4`} accent="text-violet-200" />
      </div>

      <p className="text-xs leading-relaxed text-slate-500">
        The rate ramps <em>linearly</em> from 0 to <span className="font-mono">max_lr</span> over the
        first <span className="font-mono">warmup_steps</span> — protecting the half-random initial
        weights from a giant first step — then follows a half <span className="font-mono">cosine</span>{" "}
        down to <span className="font-mono">min_lr</span>. This single curve is the schedule behind
        GPT-3, LLaMA, and almost every modern pretraining run.
      </p>
    </div>
  );
}

// ----- 2) optimizer trajectories on a 2D bowl -----------------------------

// Loss surface: f(x, y) = 0.5 * (a*x^2 + b*y^2), an elongated bowl ("ravine").
// Gradient: (a*x, b*y). a << b makes the y-direction steep and x-direction shallow.
const A = 0.06; // gentle curvature along x
const B = 1.4; // steep curvature along y (the ravine walls)
const START = [-4.4, 2.7];
const STEPS = 42;

function runSGD(lr) {
  const path = [START.slice()];
  let [x, y] = START;
  for (let i = 0; i < STEPS; i++) {
    const gx = A * x;
    const gy = B * y;
    x -= lr * gx;
    y -= lr * gy;
    path.push([x, y]);
  }
  return path;
}

function runMomentum(lr, beta) {
  const path = [START.slice()];
  let [x, y] = START;
  let vx = 0;
  let vy = 0;
  for (let i = 0; i < STEPS; i++) {
    const gx = A * x;
    const gy = B * y;
    vx = beta * vx + gx;
    vy = beta * vy + gy;
    x -= lr * vx;
    y -= lr * vy;
    path.push([x, y]);
  }
  return path;
}

function runAdam(lr, b1, b2) {
  const path = [START.slice()];
  let [x, y] = START;
  let mx = 0,
    my = 0,
    vx = 0,
    vy = 0;
  const eps = 1e-8;
  for (let i = 0; i < STEPS; i++) {
    const t = i + 1;
    const gx = A * x;
    const gy = B * y;
    mx = b1 * mx + (1 - b1) * gx;
    my = b1 * my + (1 - b1) * gy;
    vx = b2 * vx + (1 - b2) * gx * gx;
    vy = b2 * vy + (1 - b2) * gy * gy;
    const mhx = mx / (1 - b1 ** t);
    const mhy = my / (1 - b1 ** t);
    const vhx = vx / (1 - b2 ** t);
    const vhy = vy / (1 - b2 ** t);
    x -= (lr * mhx) / (Math.sqrt(vhx) + eps);
    y -= (lr * mhy) / (Math.sqrt(vhy) + eps);
    path.push([x, y]);
  }
  return path;
}

const OPT_COLORS = {
  sgd: "#fb7185",
  momentum: "#f59e0b",
  adam: "#34d399",
};

function OptimizerDemo() {
  // Tuned so each is stable and tells its story on this exact surface.
  const sgd = useMemo(() => runSGD(1.25), []);
  const momentum = useMemo(() => runMomentum(0.42, 0.82), []);
  const adam = useMemo(() => runAdam(0.45, 0.9, 0.999), []);

  // Drawing geometry.
  const W = 560;
  const H = 320;
  const pad = 18;
  const xDom = [-5, 5];
  const yDom = [-3.4, 3.4];
  const sx = (x) => pad + ((x - xDom[0]) / (xDom[1] - xDom[0])) * (W - 2 * pad);
  const sy = (y) => pad + ((yDom[1] - y) / (yDom[1] - yDom[0])) * (H - 2 * pad);

  const lossOf = (x, y) => 0.5 * (A * x * x + B * y * y);

  // A few iso-loss contour ellipses for the bowl.
  const contours = [0.6, 2.0, 4.5, 8.0].map((L) => {
    // ellipse: A x^2 / 2 + B y^2 / 2 = L  ->  x = sqrt(2L/A) cosθ, y = sqrt(2L/B) sinθ
    const rx = Math.sqrt((2 * L) / A);
    const ry = Math.sqrt((2 * L) / B);
    const pts = [];
    for (let k = 0; k <= 48; k++) {
      const th = (k / 48) * Math.PI * 2;
      pts.push(`${sx(rx * Math.cos(th)).toFixed(1)},${sy(ry * Math.sin(th)).toFixed(1)}`);
    }
    return pts.join(" ");
  });

  const pathStr = (p) =>
    p.map((pt, i) => `${i === 0 ? "M" : "L"} ${sx(pt[0]).toFixed(1)} ${sy(pt[1]).toFixed(1)}`).join(" ");

  const runs = [
    { key: "sgd", label: "SGD", path: sgd },
    { key: "momentum", label: "Momentum", path: momentum },
    { key: "adam", label: "Adam", path: adam },
  ];

  return (
    <div className="space-y-4">
      <div className="overflow-x-auto scrollbar-thin">
        <svg viewBox={`0 0 ${W} ${H}`} className="block w-full min-w-[420px]">
          {/* contour ellipses */}
          {contours.map((c, i) => (
            <polygon
              key={i}
              points={c}
              fill="none"
              stroke="#1c2438"
              strokeWidth={1}
            />
          ))}
          {/* minimum marker */}
          <circle cx={sx(0)} cy={sy(0)} r={4} fill="#5b7dff" />
          <text x={sx(0) + 8} y={sy(0) + 4} fontSize={11} fill="#8eabff">
            minimum
          </text>

          {/* trajectories */}
          {runs.map((r) => (
            <g key={r.key}>
              <path
                d={pathStr(r.path)}
                fill="none"
                stroke={OPT_COLORS[r.key]}
                strokeWidth={2}
                strokeLinejoin="round"
                strokeLinecap="round"
                opacity={0.95}
              />
              {/* step dots */}
              {r.path.map((pt, i) =>
                i % 3 === 0 ? (
                  <circle
                    key={i}
                    cx={sx(pt[0])}
                    cy={sy(pt[1])}
                    r={1.8}
                    fill={OPT_COLORS[r.key]}
                    opacity={0.7}
                  />
                ) : null
              )}
            </g>
          ))}

          {/* start marker */}
          <circle cx={sx(START[0])} cy={sy(START[1])} r={4.5} fill="#e9faff" stroke="#0b0e1a" strokeWidth={1.5} />
          <text x={sx(START[0]) + 8} y={sy(START[1]) - 6} fontSize={11} fill="#cbd5e1">
            start
          </text>
        </svg>
      </div>

      <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-xs">
        {runs.map((r) => (
          <span key={r.key} className="flex items-center gap-1.5 text-slate-300">
            <span className="inline-block h-2.5 w-5 rounded-sm" style={{ background: OPT_COLORS[r.key] }} />
            {r.label}
            <span className="font-mono text-slate-500">
              · final loss {fmt(lossOf(r.path[r.path.length - 1][0], r.path[r.path.length - 1][1]), 3)}
            </span>
          </span>
        ))}
      </div>

      <p className="text-xs leading-relaxed text-slate-500">
        The bowl is an elongated <em>ravine</em>: steep across (the <span className="font-mono">y</span>{" "}
        walls), shallow along (the <span className="font-mono">x</span> floor) — the classic case
        that wrecks plain gradient descent. <span style={{ color: OPT_COLORS.sgd }}>SGD</span> zig-zags
        across the walls and crawls along the floor.{" "}
        <span style={{ color: OPT_COLORS.momentum }}>Momentum</span> builds velocity down the valley
        but overshoots. <span style={{ color: OPT_COLORS.adam }}>Adam</span> rescales each axis by its
        own gradient history, so it takes near-equal, well-conditioned steps and walks almost straight
        to the minimum.
      </p>
    </div>
  );
}

// ----- wrapper -------------------------------------------------------------

export default function LRSchedule() {
  const [tab, setTab] = useState("schedule");
  return (
    <div className="space-y-5">
      <SegmentedControl
        value={tab}
        onChange={setTab}
        options={[
          { value: "schedule", label: "LR schedule" },
          { value: "optimizers", label: "Optimizers on a bowl" },
        ]}
      />
      {tab === "schedule" ? <ScheduleDemo /> : <OptimizerDemo />}
    </div>
  );
}
