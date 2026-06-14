import { useMemo, useState } from "react";
import { heat, mulberry32, fmt, clamp } from "../scales";
import { Slider, Toggle, Legend } from "../ui";

/**
 * A tiny multilayer perceptron you can poke.
 *
 *   2 inputs  →  3 hidden units (ReLU)  →  1 output
 *
 * Weights are fixed (generated once from a seeded mulberry32 so they are
 * deterministic across renders — never Math.random in render). Two sliders set
 * the inputs; the forward pass is recomputed live. Edges are colored by weight
 * sign (cyan = positive, rose = negative) and thicker for larger magnitude.
 * Each node's fill intensity tracks its activation via heat(). Flip the toggle
 * to overlay the backward pass: dashed gradient arrows flow from the loss back
 * toward the inputs (illustrative, animated with animate-flow-dash).
 */

// ── Fixed parameters, drawn ONCE from a seeded RNG (deterministic) ──────────
const rng = mulberry32(20240614);
const randW = () => +(rng() * 2 - 1).toFixed(2); // weights in [-1, 1]
const randB = () => +(rng() * 0.6 - 0.3).toFixed(2); // small biases

// W1: hidden[j] = sum_i x[i] * W1[j][i] + b1[j]   (3 hidden, 2 inputs)
const W1 = [
  [randW(), randW()],
  [randW(), randW()],
  [randW(), randW()],
];
const B1 = [randB(), randB(), randB()];
// W2: out = sum_j h[j] * W2[j] + b2          (1 output, 3 hidden)
const W2 = [randW(), randW(), randW()];
const B2 = randB();
const TARGET = 1.0; // toy regression target, for the MSE/loss readout

const relu = (z) => Math.max(0, z);

function forward(x0, x1) {
  const x = [x0, x1];
  const zHidden = W1.map((w, j) => w[0] * x[0] + w[1] * x[1] + B1[j]);
  const hidden = zHidden.map(relu);
  const zOut = hidden.reduce((s, h, j) => s + h * W2[j], B2);
  const out = zOut; // linear output head (regression)
  const loss = 0.5 * (out - TARGET) ** 2; // ½(ŷ − y)²
  return { x, zHidden, hidden, zOut, out, loss };
}

// Geometry
const W = 560;
const H = 300;
const COL_X = [70, 250, 430, 510]; // input / hidden / output / loss columns
const NODE_R = 22;

function nodeY(count, i) {
  const span = 200;
  const top = H / 2 - span / 2;
  return count === 1 ? H / 2 : top + (span / (count - 1)) * i;
}

function edgeColor(w) {
  // cyan for positive weights, rose for negative (per the palette)
  return w >= 0 ? "#22d3ee" : "#fb7185";
}
function edgeWidth(w) {
  return 0.6 + Math.min(5, Math.abs(w) * 4.5);
}

export default function MLPForward() {
  const [x0, setX0] = useState(0.8);
  const [x1, setX1] = useState(-0.5);
  const [showBackward, setShowBackward] = useState(false);

  const f = useMemo(() => forward(x0, x1), [x0, x1]);

  // intensity in [0,1] for a node fill, from a signed activation value
  const intensity = (v) => clamp(0.12 + Math.abs(v) * 0.55, 0, 1);

  const inY = [nodeY(2, 0), nodeY(2, 1)];
  const hidY = [nodeY(3, 0), nodeY(3, 1), nodeY(3, 2)];
  const outY = nodeY(1, 0);
  const lossY = nodeY(1, 0);

  return (
    <div className="space-y-5">
      {/* Controls */}
      <div className="flex flex-wrap items-end gap-6">
        <div className="min-w-[200px] flex-1">
          <Slider
            label="input x₁"
            min={-1.5}
            max={1.5}
            step={0.01}
            value={x0}
            onChange={setX0}
            format={(v) => fmt(v, 2)}
          />
        </div>
        <div className="min-w-[200px] flex-1">
          <Slider
            label="input x₂"
            min={-1.5}
            max={1.5}
            step={0.01}
            value={x1}
            onChange={setX1}
            accent="cyan"
            format={(v) => fmt(v, 2)}
          />
        </div>
        <Toggle
          label="Show backward pass (gradients)"
          checked={showBackward}
          onChange={setShowBackward}
        />
      </div>

      {/* Network diagram */}
      <div className="overflow-x-auto scrollbar-thin">
        <svg viewBox={`0 0 ${W} ${H}`} className="mx-auto block min-w-[520px]">
          {/* column captions */}
          <text x={COL_X[0]} y={18} textAnchor="middle" fontSize="11" fill="#64748b">
            inputs
          </text>
          <text x={COL_X[1]} y={18} textAnchor="middle" fontSize="11" fill="#64748b">
            hidden · ReLU
          </text>
          <text x={COL_X[2]} y={18} textAnchor="middle" fontSize="11" fill="#64748b">
            output ŷ
          </text>
          <text x={COL_X[3]} y={18} textAnchor="middle" fontSize="11" fill="#64748b">
            loss
          </text>

          {/* ── edges: input → hidden ── */}
          {W1.map((w, j) =>
            w.map((wij, i) => (
              <line
                key={`ih-${j}-${i}`}
                x1={COL_X[0] + NODE_R}
                y1={inY[i]}
                x2={COL_X[1] - NODE_R}
                y2={hidY[j]}
                stroke={edgeColor(wij)}
                strokeWidth={edgeWidth(wij)}
                opacity={0.55}
              />
            ))
          )}
          {/* ── edges: hidden → output ── */}
          {W2.map((wj, j) => (
            <line
              key={`ho-${j}`}
              x1={COL_X[1] + NODE_R}
              y1={hidY[j]}
              x2={COL_X[2] - NODE_R}
              y2={outY}
              stroke={edgeColor(wj)}
              strokeWidth={edgeWidth(wj)}
              opacity={0.55}
            />
          ))}
          {/* ── edge: output → loss ── */}
          <line
            x1={COL_X[2] + NODE_R}
            y1={outY}
            x2={COL_X[3] - 14}
            y2={lossY}
            stroke="#475569"
            strokeWidth={2}
            opacity={0.6}
          />

          {/* ── backward-pass gradient arrows (illustrative) ── */}
          {showBackward && (
            <g>
              <defs>
                <marker
                  id="nnp-grad-arrow"
                  markerWidth="7"
                  markerHeight="7"
                  refX="5"
                  refY="3"
                  orient="auto"
                >
                  <path d="M0,0 L6,3 L0,6 Z" fill="#a855f7" />
                </marker>
              </defs>
              {/* loss → output */}
              <line
                x1={COL_X[3] - 14}
                y1={lossY + 9}
                x2={COL_X[2] + NODE_R}
                y2={outY + 9}
                stroke="#a855f7"
                strokeWidth={2}
                strokeDasharray="5 4"
                className="animate-flow-dash"
                markerEnd="url(#nnp-grad-arrow)"
              />
              {/* output → hidden */}
              {hidY.map((hy, j) => (
                <line
                  key={`bo-${j}`}
                  x1={COL_X[2] - NODE_R}
                  y1={outY + 9}
                  x2={COL_X[1] + NODE_R}
                  y2={hy + 9}
                  stroke="#a855f7"
                  strokeWidth={1.6}
                  strokeDasharray="5 4"
                  className="animate-flow-dash"
                  markerEnd="url(#nnp-grad-arrow)"
                  opacity={0.8}
                />
              ))}
              {/* hidden → input */}
              {inY.map((iy, i) =>
                hidY.map((hy, j) => (
                  <line
                    key={`bi-${i}-${j}`}
                    x1={COL_X[1] - NODE_R}
                    y1={hy + 9}
                    x2={COL_X[0] + NODE_R}
                    y2={iy + 9}
                    stroke="#a855f7"
                    strokeWidth={1.2}
                    strokeDasharray="5 4"
                    className="animate-flow-dash"
                    markerEnd="url(#nnp-grad-arrow)"
                    opacity={0.5}
                  />
                ))
              )}
            </g>
          )}

          {/* ── input nodes ── */}
          {f.x.map((v, i) => (
            <g key={`in-${i}`}>
              <circle
                cx={COL_X[0]}
                cy={inY[i]}
                r={NODE_R}
                fill={heat(intensity(v))}
                stroke="#334155"
              />
              <text
                x={COL_X[0]}
                y={inY[i] + 4}
                textAnchor="middle"
                fontSize="12"
                fontFamily="JetBrains Mono, monospace"
                fill={intensity(v) > 0.5 ? "#0b0e1a" : "#e2e8f0"}
              >
                {fmt(v, 2)}
              </text>
              <text x={COL_X[0] - NODE_R - 8} y={inY[i] + 4} textAnchor="end" fontSize="11" fill="#64748b">
                x{i === 0 ? "₁" : "₂"}
              </text>
            </g>
          ))}

          {/* ── hidden nodes ── */}
          {f.hidden.map((h, j) => (
            <g key={`hid-${j}`}>
              <circle
                cx={COL_X[1]}
                cy={hidY[j]}
                r={NODE_R}
                fill={heat(intensity(h))}
                stroke="#334155"
              />
              <text
                x={COL_X[1]}
                y={hidY[j] + 4}
                textAnchor="middle"
                fontSize="12"
                fontFamily="JetBrains Mono, monospace"
                fill={intensity(h) > 0.5 ? "#0b0e1a" : "#e2e8f0"}
              >
                {fmt(h, 2)}
              </text>
              {/* pre-activation z shown faintly above each hidden unit */}
              <text x={COL_X[1]} y={hidY[j] - NODE_R - 5} textAnchor="middle" fontSize="9" fill="#64748b">
                z={fmt(f.zHidden[j], 2)}
              </text>
            </g>
          ))}

          {/* ── output node ── */}
          <g>
            <circle
              cx={COL_X[2]}
              cy={outY}
              r={NODE_R}
              fill={heat(intensity(f.out))}
              stroke="#334155"
            />
            <text
              x={COL_X[2]}
              y={outY + 4}
              textAnchor="middle"
              fontSize="12"
              fontFamily="JetBrains Mono, monospace"
              fill={intensity(f.out) > 0.5 ? "#0b0e1a" : "#e2e8f0"}
            >
              {fmt(f.out, 2)}
            </text>
          </g>

          {/* ── loss node ── */}
          <g>
            <rect
              x={COL_X[3] - 8}
              y={lossY - 16}
              width={56}
              height={32}
              rx={8}
              fill="#1c2438"
              stroke="#a855f7"
              opacity={0.9}
            />
            <text x={COL_X[3] + 20} y={lossY - 2} textAnchor="middle" fontSize="9" fill="#94a3b8">
              ½(ŷ−y)²
            </text>
            <text
              x={COL_X[3] + 20}
              y={lossY + 11}
              textAnchor="middle"
              fontSize="11"
              fontFamily="JetBrains Mono, monospace"
              fill="#d8b4fe"
            >
              {fmt(f.loss, 3)}
            </text>
          </g>
        </svg>
      </div>

      {/* Legend + live readout */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <Legend
          items={[
            { label: "positive weight", color: "#22d3ee" },
            { label: "negative weight", color: "#fb7185" },
            ...(showBackward ? [{ label: "gradient flow", color: "#a855f7" }] : []),
          ]}
        />
        <div className="font-mono text-sm">
          <span className="text-slate-500">ŷ = </span>
          <span className="text-brand-200">{fmt(f.out, 3)}</span>
          <span className="mx-2 text-slate-600">·</span>
          <span className="text-slate-500">target y = </span>
          <span className="text-slate-300">{TARGET.toFixed(1)}</span>
          <span className="mx-2 text-slate-600">·</span>
          <span className="text-slate-500">loss = </span>
          <span className="text-violet-300">{fmt(f.loss, 3)}</span>
        </div>
      </div>

      <p className="text-xs text-slate-500">
        Drag the input sliders and watch every value recompute. A hidden unit goes dark when its
        pre-activation <span className="font-mono">z</span> is negative — ReLU has clamped it to 0,
        so it contributes nothing downstream. Toggle the backward pass to see which direction the
        gradient signal travels during learning.
      </p>
    </div>
  );
}
