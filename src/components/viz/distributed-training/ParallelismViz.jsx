import { useMemo, useState } from "react";
import { Slider, SegmentedControl, Stat } from "../ui";
import { fmt } from "../scales";

/**
 * Two deterministic demos for distributed training.
 *
 * 1) ParallelismDiagram — a SegmentedControl across {Data, Tensor, Pipeline,
 *    FSDP}. For each mode we draw 4 GPU boxes and show, with a small grid of
 *    "layer × shard" cells, exactly which slice of the model and which slice of
 *    the data each GPU holds, plus the collective communication that mode
 *    requires. No animation, fully static per selection.
 *
 * 2) MemoryBar — a stacked bar of the per-GPU training-memory budget (params,
 *    gradients, optimizer state, activations) driven by a parameter-count
 *    slider, using the "16 bytes/param for mixed-precision AdamW" rule.
 */

const MODES = ["Data", "Tensor", "Pipeline", "FSDP"];
const N_GPU = 4;
const N_LAYERS = 4; // pretend the model has 4 transformer layers

const GPU_COLORS = ["#22d3ee", "#a855f7", "#f59e0b", "#34d399"];

// Per-mode descriptive copy.
const INFO = {
  Data: {
    headline: "Data Parallel (DDP)",
    model: "Every GPU holds a full copy of the model.",
    data: "The global batch is split — each GPU sees a different data shard.",
    comm: "all-reduce of gradients after backward (every step).",
    commColor: "#fb7185",
  },
  Tensor: {
    headline: "Tensor Parallel (TP)",
    model: "Each GPU holds a vertical slice of every layer's weight matrices.",
    data: "All GPUs process the SAME tokens — they cooperate on each matmul.",
    comm: "all-reduce inside every layer (twice per transformer block).",
    commColor: "#22d3ee",
  },
  Pipeline: {
    headline: "Pipeline Parallel (PP)",
    model: "Each GPU owns a contiguous block of layers (a 'stage').",
    data: "Micro-batches flow stage→stage; idle time = the 'bubble'.",
    comm: "point-to-point send/recv of activations between adjacent stages.",
    commColor: "#f59e0b",
  },
  FSDP: {
    headline: "Fully Sharded Data Parallel (ZeRO-3)",
    model: "Each GPU holds only a 1/N shard of params, grads & optimizer state.",
    data: "Like DDP, each GPU sees a different data shard…",
    comm: "all-gather params just-in-time per layer, reduce-scatter grads.",
    commColor: "#34d399",
  },
};

function ParallelismDiagram() {
  const [mode, setMode] = useState("Data");
  const info = INFO[mode];

  // For each GPU g and (layer L, shard column c), decide if this GPU "holds"
  // that piece of weights. The 4×4 grid inside each GPU box represents the
  // model as [layers (rows)] × [width-shards (cols)].
  const holds = (g, layer, col) => {
    switch (mode) {
      case "Data":
        // full model copy: everything present on every GPU
        return true;
      case "Tensor":
        // each GPU holds one vertical width-slice across ALL layers
        return col === g;
      case "Pipeline":
        // each GPU holds all width of one contiguous layer
        return layer === g;
      case "FSDP":
        // each GPU holds a 1/N flat shard — approximate as "one column",
        // gathered just-in-time for the current layer (we shade the rest faint)
        return col === g;
      default:
        return true;
    }
  };

  // Layout
  const boxW = 138;
  const boxH = 150;
  const gapX = 24;
  const totalW = N_GPU * boxW + (N_GPU - 1) * gapX;
  const cellPad = 14;
  const gridW = boxW - cellPad * 2;
  const cw = gridW / N_LAYERS; // cell width
  const ch = 18; // cell height
  const gridTop = 46;

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <SegmentedControl
          label="Parallelism strategy"
          options={MODES}
          value={mode}
          onChange={setMode}
        />
        <div className="text-xs text-slate-500">
          4 GPUs · a toy model of {N_LAYERS} layers, each split into {N_LAYERS} width-shards.
        </div>
      </div>

      {/* legend for the grid */}
      <div className="flex flex-wrap gap-x-5 gap-y-1 text-[11px] text-slate-400">
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-3 w-3 rounded-sm" style={{ background: "#3a55f5" }} />
          weights resident on this GPU
        </span>
        <span className="flex items-center gap-1.5">
          <span
            className="inline-block h-3 w-3 rounded-sm border border-dashed"
            style={{ background: "transparent", borderColor: "#27314a" }}
          />
          {mode === "FSDP" ? "absent — gathered just-in-time" : "not on this GPU"}
        </span>
      </div>

      <div className="overflow-x-auto scrollbar-thin">
        <svg width={totalW} height={boxH + 58} className="block min-w-[600px]">
          {Array.from({ length: N_GPU }, (_, g) => {
            const x0 = g * (boxW + gapX);
            const accent = GPU_COLORS[g];
            // For PP, label which contiguous layers this GPU owns.
            return (
              <g key={g}>
                {/* GPU frame */}
                <rect
                  x={x0}
                  y={28}
                  width={boxW}
                  height={boxH}
                  rx={10}
                  fill="#101728"
                  stroke={accent}
                  strokeWidth={1.5}
                  opacity={0.95}
                />
                <rect x={x0} y={28} width={boxW} height={4} rx={2} fill={accent} />
                <text
                  x={x0 + boxW / 2}
                  y={20}
                  textAnchor="middle"
                  fontSize="12"
                  fontWeight="600"
                  fill={accent}
                  fontFamily="JetBrains Mono, monospace"
                >
                  GPU {g}
                </text>

                {/* the layer×shard grid */}
                {Array.from({ length: N_LAYERS }, (_, layer) =>
                  Array.from({ length: N_LAYERS }, (_, col) => {
                    const on = holds(g, layer, col);
                    const cx = x0 + cellPad + col * cw;
                    const cy = 28 + gridTop + layer * (ch + 4);
                    return (
                      <rect
                        key={`${layer}-${col}`}
                        x={cx}
                        y={cy}
                        width={cw - 4}
                        height={ch}
                        rx={3}
                        fill={on ? "#3a55f5" : "transparent"}
                        stroke={on ? "#6f8bff" : "#27314a"}
                        strokeWidth={1}
                        strokeDasharray={on ? undefined : "2 2"}
                        opacity={on ? 0.92 : 0.5}
                      />
                    );
                  })
                )}

                {/* row label: "layer i" on the leftmost GPU only */}
                {g === 0 &&
                  Array.from({ length: N_LAYERS }, (_, layer) => (
                    <text
                      key={`rl${layer}`}
                      x={x0 - 6}
                      y={28 + gridTop + layer * (ch + 4) + ch - 5}
                      textAnchor="end"
                      fontSize="8.5"
                      fill="#475569"
                      fontFamily="JetBrains Mono, monospace"
                    >
                      L{layer}
                    </text>
                  ))}

                {/* data-shard tag under the grid */}
                <text
                  x={x0 + boxW / 2}
                  y={28 + boxH - 22}
                  textAnchor="middle"
                  fontSize="9.5"
                  fill="#94a3b8"
                  fontFamily="JetBrains Mono, monospace"
                >
                  {mode === "Tensor"
                    ? "batch: ALL (shared)"
                    : mode === "Pipeline"
                    ? "micro-batches →"
                    : `data shard ${g}`}
                </text>
                <text
                  x={x0 + boxW / 2}
                  y={28 + boxH - 9}
                  textAnchor="middle"
                  fontSize="8.5"
                  fill="#475569"
                  fontFamily="JetBrains Mono, monospace"
                >
                  {mode === "Pipeline"
                    ? `stage ${g}: layer ${g}`
                    : mode === "Tensor"
                    ? `width-slice ${g}`
                    : mode === "FSDP"
                    ? `param shard ${g}/${N_GPU}`
                    : "full model"}
                </text>

                {/* communication arrows between GPUs */}
                {g < N_GPU - 1 &&
                  (mode === "Pipeline" ? (
                    // forward send/recv between adjacent stages
                    <g>
                      <line
                        x1={x0 + boxW}
                        y1={28 + boxH / 2}
                        x2={x0 + boxW + gapX}
                        y2={28 + boxH / 2}
                        stroke={info.commColor}
                        strokeWidth={2}
                        markerEnd="url(#ppArrow)"
                      />
                    </g>
                  ) : (
                    // collective: a ring/line connecting all GPUs
                    <line
                      x1={x0 + boxW}
                      y1={28 + boxH + 14}
                      x2={x0 + boxW + gapX}
                      y2={28 + boxH + 14}
                      stroke={info.commColor}
                      strokeWidth={1.5}
                      strokeDasharray="4 3"
                      opacity={0.8}
                    />
                  ))}
              </g>
            );
          })}

          {/* collective band label for non-pipeline modes */}
          {mode !== "Pipeline" && (
            <text
              x={totalW / 2}
              y={28 + boxH + 38}
              textAnchor="middle"
              fontSize="10.5"
              fill={info.commColor}
              fontFamily="JetBrains Mono, monospace"
            >
              {mode === "Data"
                ? "⇄ all-reduce gradients across all GPUs"
                : mode === "Tensor"
                ? "⇄ all-reduce activations within each layer"
                : "⇄ all-gather params (fwd) · reduce-scatter grads (bwd)"}
            </text>
          )}

          <defs>
            <marker
              id="ppArrow"
              markerWidth="8"
              markerHeight="8"
              refX="6"
              refY="3"
              orient="auto"
            >
              <path d="M0,0 L6,3 L0,6 Z" fill={INFO.Pipeline.commColor} />
            </marker>
          </defs>
        </svg>
      </div>

      {/* the bubble illustration, only for Pipeline */}
      {mode === "Pipeline" && <PipelineBubble />}

      {/* descriptive panel */}
      <div className="rounded-xl border border-white/10 bg-ink-900/50 p-4 text-sm">
        <div className="mb-2 font-medium text-slate-200">{info.headline}</div>
        <ul className="space-y-1 text-slate-400">
          <li>
            <span className="text-slate-500">Model:</span> {info.model}
          </li>
          <li>
            <span className="text-slate-500">Data:</span> {info.data}
          </li>
          <li>
            <span className="text-slate-500">Communication:</span>{" "}
            <span style={{ color: info.commColor }}>{info.comm}</span>
          </li>
        </ul>
      </div>
    </div>
  );
}

/**
 * A tiny pipeline schedule (GPipe-style) showing micro-batches flowing through
 * 4 stages, with the warmup/cooldown "bubble" (idle cells) shaded. Deterministic.
 */
function PipelineBubble() {
  const stages = N_GPU;
  const micro = 4; // micro-batches
  // In a naive GPipe forward schedule, stage s starts micro-batch m at timestep
  // t = s + m (then a symmetric backward). Total forward timesteps = stages+micro-1.
  const T = stages + micro - 1;
  const cw = 30;
  const ch = 22;
  const left = 64;
  const top = 24;

  return (
    <div className="rounded-xl border border-white/10 bg-ink-900/50 p-4">
      <div className="mb-2 text-xs uppercase tracking-wider text-slate-500">
        The pipeline bubble · forward pass schedule
      </div>
      <div className="overflow-x-auto scrollbar-thin">
        <svg width={left + T * cw + 20} height={top + stages * (ch + 4) + 26}>
          {/* timestep header */}
          {Array.from({ length: T }, (_, t) => (
            <text
              key={`t${t}`}
              x={left + t * cw + cw / 2}
              y={top - 6}
              textAnchor="middle"
              fontSize="9"
              fill="#475569"
              fontFamily="JetBrains Mono, monospace"
            >
              t{t}
            </text>
          ))}
          {Array.from({ length: stages }, (_, s) => (
            <g key={s}>
              <text
                x={left - 8}
                y={top + s * (ch + 4) + ch - 6}
                textAnchor="end"
                fontSize="9.5"
                fill="#94a3b8"
                fontFamily="JetBrains Mono, monospace"
              >
                stage {s}
              </text>
              {Array.from({ length: T }, (_, t) => {
                // micro-batch m is on stage s at time t when t = s + m, i.e. m = t - s
                const m = t - s;
                const busy = m >= 0 && m < micro;
                const x = left + t * cw;
                const y = top + s * (ch + 4);
                return (
                  <rect
                    key={t}
                    x={x}
                    y={y}
                    width={cw - 4}
                    height={ch}
                    rx={3}
                    fill={busy ? GPU_COLORS[m % GPU_COLORS.length] : "#0d1422"}
                    stroke={busy ? "none" : "#1c2438"}
                    strokeDasharray={busy ? undefined : "2 2"}
                    opacity={busy ? 0.85 : 1}
                  >
                    {busy && <title>{`stage ${s} · micro-batch ${m}`}</title>}
                  </rect>
                );
              })}
            </g>
          ))}
          <text
            x={left}
            y={top + stages * (ch + 4) + 16}
            fontSize="9.5"
            fill="#64748b"
            fontFamily="JetBrains Mono, monospace"
          >
            colored = computing · dashed = idle (the bubble). More micro-batches ⇒ smaller bubble
            fraction.
          </text>
        </svg>
      </div>
    </div>
  );
}

// ---- 2) per-GPU memory budget bar -----------------------------------------

const SEG = [
  { key: "params", label: "params (bf16)", bytes: 2, color: "#22d3ee" },
  { key: "grads", label: "grads (bf16)", bytes: 2, color: "#a855f7" },
  { key: "master", label: "fp32 master copy", bytes: 4, color: "#5b7dff" },
  { key: "m", label: "Adam m (fp32)", bytes: 4, color: "#f59e0b" },
  { key: "v", label: "Adam v (fp32)", bytes: 4, color: "#fb7185" },
];

function gib(bytes) {
  return bytes / 1024 ** 3;
}

function MemoryBar() {
  // params in billions
  const [pB, setPB] = useState(7);
  // activations as a chosen fixed overhead in GiB (depends on batch/seq, so we
  // expose it as a separate knob rather than pretend it's a per-param constant).
  const [actGiB, setActGiB] = useState(12);
  const [sharded, setSharded] = useState(false);
  const shards = 8;

  const params = pB * 1e9;
  const segBytes = SEG.map((s) => ({ ...s, total: params * s.bytes }));
  const modelStateBytes = segBytes.reduce((a, s) => a + s.total, 0); // 16 bytes/param
  const denom = sharded ? shards : 1;

  // Per-GPU values: model-state segments shard under ZeRO/FSDP; activations don't.
  const perGpuSeg = segBytes.map((s) => ({ ...s, perGpu: gib(s.total) / denom }));
  const modelGiB = gib(modelStateBytes) / denom;
  const totalGiB = modelGiB + actGiB;

  const maxScale = useMemo(() => {
    // headroom up to the unsharded worst case so the bar doesn't jump around
    return gib(80e9 * 16) + 20; // ~ 80B params @16B/param + activations slack
  }, []);

  const barW = 620;
  const pxPerGiB = barW / maxScale;

  return (
    <div className="space-y-5">
      <div className="grid gap-x-6 gap-y-4 sm:grid-cols-2">
        <Slider
          label="parameters (billions)"
          min={0.5}
          max={70}
          step={0.5}
          value={pB}
          onChange={setPB}
          format={(v) => `${v}B`}
        />
        <Slider
          label="activations (GiB, per GPU)"
          min={1}
          max={40}
          step={1}
          value={actGiB}
          onChange={setActGiB}
          accent="cyan"
          format={(v) => `${v} GiB`}
        />
      </div>

      <label className="flex cursor-pointer items-center gap-2 text-sm text-slate-300">
        <input
          type="checkbox"
          checked={sharded}
          onChange={(e) => setSharded(e.target.checked)}
          className="h-4 w-4 accent-emerald-400"
        />
        Shard model state across {shards} GPUs (ZeRO-3 / FSDP)
      </label>

      {/* the stacked bar */}
      <div className="rounded-xl border border-white/10 bg-ink-900/50 p-4">
        <div className="overflow-x-auto scrollbar-thin">
          <svg width={barW + 4} height={92} className="block min-w-[560px]">
            {(() => {
              let x = 0;
              const rects = perGpuSeg.map((s) => {
                const w = s.perGpu * pxPerGiB;
                const r = (
                  <rect
                    key={s.key}
                    x={x}
                    y={24}
                    width={Math.max(0, w - 1)}
                    height={30}
                    fill={s.color}
                    opacity={0.85}
                  >
                    <title>{`${s.label}: ${fmt(s.perGpu, 1)} GiB`}</title>
                  </rect>
                );
                x += w;
                return r;
              });
              const actW = actGiB * pxPerGiB;
              rects.push(
                <rect
                  key="act"
                  x={x}
                  y={24}
                  width={Math.max(0, actW - 1)}
                  height={30}
                  fill="#34d399"
                  opacity={0.55}
                >
                  <title>{`activations: ${fmt(actGiB, 1)} GiB`}</title>
                </rect>
              );
              x += actW;
              // 80 GiB reference line (an A100/H100 GPU)
              const ref80 = 80 * pxPerGiB;
              rects.push(
                <g key="ref">
                  <line x1={ref80} y1={14} x2={ref80} y2={64} stroke="#e2e8f0" strokeDasharray="3 3" opacity={0.6} />
                  <text x={ref80 + 4} y={12} fontSize="9.5" fill="#94a3b8" fontFamily="JetBrains Mono, monospace">
                    80 GiB (one H100)
                  </text>
                </g>
              );
              return rects;
            })()}

            <text x={0} y={74} fontSize="11" fill="#cbd5e1" fontFamily="JetBrains Mono, monospace">
              total per GPU ≈ {fmt(totalGiB, 1)} GiB
              {totalGiB > 80 && (
                <tspan fill="#fb7185"> — exceeds one 80 GiB GPU!</tspan>
              )}
            </text>
          </svg>
        </div>

        {/* legend */}
        <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-slate-400">
          {SEG.map((s) => (
            <span key={s.key} className="flex items-center gap-1.5">
              <span className="inline-block h-3 w-3 rounded-sm" style={{ background: s.color }} />
              {s.label}
            </span>
          ))}
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-3 w-3 rounded-sm" style={{ background: "#34d399", opacity: 0.55 }} />
            activations
          </span>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <Stat label="bytes / param" value="16" accent="text-cyan-200" />
        <Stat label="model state" value={`${fmt(modelGiB, 1)} GiB`} accent="text-brand-200" />
        <Stat label="+ activations" value={`${fmt(actGiB, 1)} GiB`} accent="text-emerald-200" />
        <Stat label="= per GPU" value={`${fmt(totalGiB, 1)} GiB`} accent="text-amber-200" />
      </div>

      <p className="text-xs leading-relaxed text-slate-500">
        The mixed-precision AdamW rule of thumb: <span className="font-mono text-slate-300">2 + 2 + 4
        + 4 + 4 = 16</span> bytes per parameter (bf16 weights + bf16 grads + fp32 master weights +
        fp32 first &amp; second moments). A 7B model therefore needs ~112 GiB <em>just</em> for model
        state — already over a single 80 GiB GPU before a single activation is stored. Sharding (the
        toggle) is what makes large models trainable at all.
      </p>
    </div>
  );
}

export default function ParallelismViz({ variant = "diagram" }) {
  return variant === "memory" ? <MemoryBar /> : <ParallelismDiagram />;
}
