import { useMemo, useState } from "react";
import { Slider, Stat, Legend } from "../ui";
import { heat, diverge, mulberry32, fmt } from "../scales";

/**
 * LoRAViz — visualizes the LoRA decomposition  W = W0 + ΔW,  ΔW = (α/r)·B·A.
 *
 *   • W0 is a frozen d×d weight matrix (rendered as a static, deterministic
 *     heatmap so it never changes — the pretrained knowledge is locked).
 *   • The trainable update is the product of two skinny matrices:
 *       B ∈ ℝ^{d×r}  (a tall strip, d rows × r columns)
 *       A ∈ ℝ^{r×d}  (a wide strip, r rows × d columns)
 *     Their product B·A is a full d×d matrix, but it is constrained to rank r.
 *
 * A single slider for the rank r (1..d) grows/shrinks the two strips and
 * recomputes the live parameter accounting:
 *       full  ΔW :  d·d              trainable params (if you fine-tuned it densely)
 *       LoRA  ΔW :  r·(d + d) = 2·d·r trainable params
 *       ratio    :  2r / d           (fraction trainable)
 *
 * Everything is deterministic (seeded RNG) so the figure renders identically
 * on every load. d is fixed small (the visual stand-in); a callout maps it to
 * the real d = 4096 case in the prose.
 */

const D = 12; // on-screen matrix side (a small stand-in for a real d=4096)
const CELL = 17; // px per heatmap cell
const GAP = 2;

// Deterministic base weight matrix W0 (frozen). Seeded — never regenerated.
const W0 = (() => {
  const rnd = mulberry32(20251234);
  const m = [];
  for (let i = 0; i < D; i++) {
    const row = [];
    for (let j = 0; j < D; j++) row.push(rnd() * 2 - 1); // in [-1, 1]
    m.push(row);
  }
  return m;
})();

// Deterministic full-rank factors. We only ever USE the first r columns of B
// and first r rows of A, so changing r is a clean truncation (no re-seeding).
const B_FULL = (() => {
  const rnd = mulberry32(777);
  const m = [];
  for (let i = 0; i < D; i++) {
    const row = [];
    for (let k = 0; k < D; k++) row.push(rnd() * 2 - 1);
    m.push(row);
  }
  return m;
})();
const A_FULL = (() => {
  const rnd = mulberry32(555);
  const m = [];
  for (let k = 0; k < D; k++) {
    const row = [];
    for (let j = 0; j < D; j++) row.push(rnd() * 2 - 1);
    m.push(row);
  }
  return m;
})();

// ΔW = (α/r) · B[:, :r] · A[:r, :]  — a d×d matrix of rank ≤ r.
function deltaW(r, alpha) {
  const scale = alpha / r;
  const out = [];
  for (let i = 0; i < D; i++) {
    const row = new Array(D).fill(0);
    for (let j = 0; j < D; j++) {
      let s = 0;
      for (let k = 0; k < r; k++) s += B_FULL[i][k] * A_FULL[k][j];
      row[j] = (s / Math.sqrt(D)) * scale; // /√D keeps the magnitude readable
    }
    out.push(row);
  }
  return out;
}

/** A small heatmap grid drawn as inline divs. domain centers color mapping. */
function Grid({ matrix, rows, cols, cell = CELL, color, title, dim = false }) {
  return (
    <div className="inline-flex flex-col items-center gap-1">
      <div
        className="grid"
        style={{
          gridTemplateColumns: `repeat(${cols}, ${cell}px)`,
          gridTemplateRows: `repeat(${rows}, ${cell}px)`,
          gap: GAP,
        }}
      >
        {matrix.flatMap((row, i) =>
          row.slice(0, cols).map((v, j) => (
            <div
              key={`${i}-${j}`}
              className="rounded-[3px] transition-all duration-300"
              style={{
                width: cell,
                height: cell,
                background: color(v),
                opacity: dim ? 0.5 : 1,
              }}
              title={fmt(v, 3)}
            />
          ))
        )}
      </div>
      {title && (
        <div className="font-mono text-[11px] text-slate-400">{title}</div>
      )}
    </div>
  );
}

export default function LoRAViz() {
  const [r, setR] = useState(2);
  const alpha = 8; // fixed LoRA scaling numerator α (so effective scale = α/r)

  const dW = useMemo(() => deltaW(r, alpha), [r]);

  // Parameter accounting (uses the on-screen d = D; prose maps to d = 4096).
  const full = D * D; // dense ΔW
  const lora = 2 * D * r; // B (d×r) + A (r×d)
  const frac = lora / full; // fraction trainable
  const reduction = (1 - frac) * 100;

  // Color: W0 frozen → cool indigo/heat ramp; ΔW signed → diverging rose/emerald.
  const colorW0 = (v) => heat((v + 1) / 2, 0.85);
  const maxAbs = useMemo(() => {
    let m = 1e-6;
    for (const row of dW) for (const v of row) m = Math.max(m, Math.abs(v));
    return m;
  }, [dW]);
  const colorDelta = (v) => diverge(v / maxAbs);
  const colorB = (v) => diverge((v / Math.sqrt(D)) * 1.4);
  const colorA = (v) => diverge((v / Math.sqrt(D)) * 1.4);

  return (
    <div className="space-y-6">
      {/* Control */}
      <div className="max-w-sm">
        <Slider
          label="LoRA rank r"
          value={r}
          min={1}
          max={D}
          step={1}
          onChange={setR}
          accent="violet"
          format={(v) => `${v} / ${D}`}
        />
        <p className="mt-2 text-xs text-slate-500">
          Slide r up to make the adapter more expressive (and bigger); slide it
          down toward the sweet spot. In real models <span className="font-mono">d</span> is
          thousands, yet <span className="font-mono">r</span> ∈ 4–64 is plenty.
        </p>
      </div>

      {/* The equation, drawn as matrices */}
      <div className="overflow-x-auto scrollbar-thin">
        <div className="flex min-w-[640px] items-center justify-center gap-4 py-2">
          {/* Frozen W0 */}
          <div className="flex flex-col items-center gap-1">
            <Grid
              matrix={W0}
              rows={D}
              cols={D}
              color={colorW0}
              title={`W₀  (${D}×${D}, frozen)`}
              dim
            />
            <span className="rounded bg-slate-500/15 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-slate-400">
              ❄ frozen · {full.toLocaleString()} params
            </span>
          </div>

          <span className="text-2xl text-slate-500">+</span>

          {/* The trainable low-rank update, factored */}
          <div className="flex flex-col items-center gap-1.5">
            <span className="text-[11px] uppercase tracking-wider text-violet-300">
              trainable update ΔW = (α/r)·B·A
            </span>
            <div className="flex items-center gap-3">
              {/* B: tall d×r strip */}
              <Grid
                matrix={B_FULL}
                rows={D}
                cols={r}
                color={colorB}
                title={`B  (${D}×${r})`}
              />
              <span className="text-lg text-slate-500">·</span>
              {/* A: wide r×d strip */}
              <Grid
                matrix={A_FULL}
                rows={r}
                cols={D}
                color={colorA}
                title={`A  (${r}×${D})`}
              />
              <span className="text-lg text-slate-500">=</span>
              {/* Product ΔW: full d×d but rank ≤ r */}
              <Grid
                matrix={dW}
                rows={D}
                cols={D}
                color={colorDelta}
                title={`ΔW  (${D}×${D}, rank ≤ ${r})`}
              />
            </div>
            <span className="rounded bg-violet-500/15 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-violet-300">
              trains only B + A · {lora.toLocaleString()} params
            </span>
          </div>
        </div>
      </div>

      <Legend
        items={[
          { color: heat(0.6), label: "W₀ entries (frozen)" },
          { color: "#34d399", label: "ΔW > 0" },
          { color: "#fb7185", label: "ΔW < 0" },
        ]}
      />

      {/* Live parameter accounting */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat
          label="Dense ΔW params"
          value={`${full.toLocaleString()}`}
          accent="text-slate-300"
        />
        <Stat
          label="LoRA params  2·d·r"
          value={`${lora.toLocaleString()}`}
          accent="text-violet-200"
        />
        <Stat
          label="Fraction trainable"
          value={`${(frac * 100).toFixed(1)}%`}
          accent="text-cyan-200"
        />
        <Stat
          label="Params saved"
          value={`${reduction.toFixed(1)}%`}
          accent="text-emerald-200"
        />
      </div>

      {/* The same accounting at real scale (d = 4096), recomputed live for r */}
      <div className="rounded-lg border border-white/10 bg-ink-900/40 px-4 py-3 text-sm text-slate-400">
        <span className="text-slate-300">At a realistic </span>
        <span className="font-mono text-slate-200">d = k = 4096</span>
        <span className="text-slate-300"> with this same rank </span>
        <span className="font-mono text-violet-200">r = {r}</span>:{" "}
        a dense update is{" "}
        <span className="font-mono text-slate-200">
          {(4096 * 4096).toLocaleString()}
        </span>{" "}
        params, but LoRA needs only{" "}
        <span className="font-mono text-violet-200">
          {(2 * 4096 * r).toLocaleString()}
        </span>{" "}
        —{" "}
        <span className="font-mono text-emerald-200">
          {((1 - (2 * 4096 * r) / (4096 * 4096)) * 100).toFixed(2)}%
        </span>{" "}
        fewer. Two skinny matrices replace one huge one.
      </div>
    </div>
  );
}
