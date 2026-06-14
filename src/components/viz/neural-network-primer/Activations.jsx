import { useMemo, useState } from "react";
import LineChart from "../LineChart";
import { Toggle } from "../ui";

/**
 * Activation-function explorer.
 *
 * Plots ReLU, GELU, sigmoid and tanh on one set of axes over x ∈ [-4, 4] by
 * sampling each function densely and handing the resulting point arrays to the
 * shared LineChart as separate series. Toggles let you focus on the two
 * families that matter most in deep nets (saturating vs. non-saturating).
 */

const X_MIN = -4;
const X_MAX = 4;
const SAMPLES = 161; // dense enough that the curves look smooth

const relu = (x) => Math.max(0, x);
const sigmoid = (x) => 1 / (1 + Math.exp(-x));
const tanh = (x) => Math.tanh(x);
// Tanh approximation of GELU — the form actually shipped in GPT-2/BERT kernels.
const gelu = (x) =>
  0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));

function sample(fn) {
  const pts = [];
  for (let i = 0; i < SAMPLES; i++) {
    const x = X_MIN + ((X_MAX - X_MIN) * i) / (SAMPLES - 1);
    pts.push([x, fn(x)]);
  }
  return pts;
}

const FUNCS = [
  { key: "relu", label: "ReLU", color: "#22d3ee", fn: relu },
  { key: "gelu", label: "GELU", color: "#5b7dff", fn: gelu },
  { key: "sigmoid", label: "sigmoid", color: "#f59e0b", fn: sigmoid },
  { key: "tanh", label: "tanh", color: "#fb7185", fn: tanh },
];

export default function Activations() {
  const [showSaturating, setShowSaturating] = useState(true);

  const series = useMemo(() => {
    const built = FUNCS.map((f) => ({
      label: f.label,
      color: f.color,
      points: sample(f.fn),
      saturating: f.key === "sigmoid" || f.key === "tanh",
    }));
    return built
      .filter((s) => showSaturating || !s.saturating)
      .map(({ saturating, ...rest }) => rest);
  }, [showSaturating]);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-5">
        <Toggle
          label="Show saturating units (sigmoid, tanh)"
          checked={showSaturating}
          onChange={setShowSaturating}
        />
      </div>

      <LineChart
        series={series}
        xLabel="input  x"
        yLabel="output  φ(x)"
        height={340}
        xTicks={8}
        yTicks={6}
        fmtX={(v) => v.toFixed(0)}
        fmtY={(v) => v.toFixed(1)}
        annotations={[{ x: 0, label: "x = 0", color: "#475569" }]}
      />

      <div className="grid gap-2 text-xs text-slate-400 sm:grid-cols-2">
        <p>
          <span className="font-mono text-cyan-300">ReLU</span> and{" "}
          <span className="font-mono text-brand-300">GELU</span> keep a strong slope for
          positive inputs, so gradients survive through deep stacks — they dominate modern
          transformers.
        </p>
        <p>
          <span className="font-mono text-amber-300">sigmoid</span> and{" "}
          <span className="font-mono text-rose-300">tanh</span> flatten (saturate) at both ends:
          their gradient goes to ≈0 far from the origin, which starves learning in deep networks.
        </p>
      </div>
    </div>
  );
}
