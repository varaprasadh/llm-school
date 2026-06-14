import { useMemo, useState } from "react";
import { Slider, Button } from "../ui";
import { heat } from "../scales";

/**
 * TransformerStack — an interactive vertical diagram of a decoder-only
 * transformer, drawn as bespoke SVG (matching the AttentionExplorer house style).
 *
 * Bottom -> top:
 *   input token chips -> token + positional embeddings -> N identical Blocks
 *   (each: LN -> Attention -> +residual -> LN -> FFN -> +residual)
 *   -> final LayerNorm -> LM head -> logits / softmax.
 *
 * - A <Slider> sets N; the stack visibly grows (blocks are abbreviated when N is
 *   large so the figure stays readable).
 * - Clicking any component highlights it and opens a side panel describing what
 *   it does and the tensor shape flowing through it.
 * - A "signal" pulse can be stepped up the residual stream. Fully deterministic:
 *   the active stage is plain integer state, no Math.random in render.
 */

const TOKENS = ["The", "river", "bank", "was"];

// d_model used purely for the displayed shapes / param hints.
const D = 4;
const VOCAB = 12;

// Each "stage" the signal can occupy as it flows bottom -> top. Block internals
// are represented by a single stage per block (the residual-stream value after
// that block); clicking a block reveals its sub-steps in the panel.
function buildStages(n) {
  const stages = [
    { id: "tokens", kind: "tokens" },
    { id: "embed", kind: "embed" },
  ];
  for (let i = 0; i < n; i++) stages.push({ id: `block-${i}`, kind: "block", index: i });
  stages.push({ id: "lnf", kind: "lnf" });
  stages.push({ id: "head", kind: "head" });
  stages.push({ id: "logits", kind: "logits" });
  return stages;
}

// Human-readable info for the side panel, keyed by stage kind.
const INFO = {
  tokens: {
    name: "Token IDs",
    color: "#64748b",
    shape: "(B, T)",
    lines: [
      "The prompt after tokenization: a grid of integers, one ID per token.",
      "These index into the embedding table — they carry no meaning yet, just identity.",
    ],
    chapter: "Chapter 5",
    slug: "tokenization",
  },
  embed: {
    name: "Token + Positional Embeddings",
    color: "#34d399",
    shape: "(B, T, d)",
    lines: [
      "Each ID is looked up in the embedding table to become a d-dim vector,",
      "then a positional vector is added so the model can tell order apart.",
    ],
    chapter: "Chapter 6 & 10",
    slug: "embeddings",
  },
  block: {
    name: "Transformer Block",
    color: "#a855f7",
    shape: "(B, T, d)",
    lines: [
      "Pre-norm block: x -> LayerNorm -> Multi-Head Attention -> add back to x,",
      "then -> LayerNorm -> Feed-Forward MLP -> add back to x. Shape never changes.",
    ],
    chapter: "Chapters 8-11",
    slug: "self-attention",
  },
  lnf: {
    name: "Final LayerNorm",
    color: "#22d3ee",
    shape: "(B, T, d)",
    lines: [
      "One last normalization of the residual stream before the output head,",
      "so the logits see a clean, unit-scale representation.",
    ],
    chapter: "Chapter 11",
    slug: "transformer-block",
  },
  head: {
    name: "LM Head (unembed)",
    color: "#f59e0b",
    shape: "(B, T, V)",
    lines: [
      "A single linear projection from d dimensions to the vocabulary size V.",
      "It scores every possible next token at every position.",
    ],
    chapter: "Chapter 6",
    slug: "embeddings",
  },
  logits: {
    name: "Logits -> softmax",
    color: "#fb7185",
    shape: "(B, T, V)",
    lines: [
      "Raw scores become a probability distribution over the vocabulary via softmax.",
      "Sample from the last position to pick the next token, then repeat.",
    ],
    chapter: "Chapter 3",
    slug: "language-modeling",
  },
};

// The sub-steps drawn inside every block box.
const SUBSTEPS = ["LN", "Attn", "⊕", "LN", "FFN", "⊕"];

export default function TransformerStack() {
  const [n, setN] = useState(6);
  const [selected, setSelected] = useState("block-0");
  const [pulse, setPulse] = useState(-1); // index into stages; -1 = no pulse yet

  const stages = useMemo(() => buildStages(n), [n]);

  // Keep selection valid when N shrinks past the selected block.
  const selStage = stages.find((s) => s.id === selected) || stages[0];
  const info = INFO[selStage.kind];

  // ---- Layout -------------------------------------------------------------
  // We draw bottom -> top, but SVG y grows downward, so we build a list of rows
  // top-first and assign increasing y. Each row knows which stage(s) it shows.
  // When N is large we collapse the middle blocks into a "⋮ ×k" row so the
  // figure never grows unbounded.
  const COLLAPSE_AT = 7; // show all blocks up to this many, else abbreviate
  const showAllBlocks = n <= COLLAPSE_AT;

  const W = 360;
  const rowH = 56;
  const blockH = 76;
  const gap = 14;
  const cx = W / 2;
  const boxW = 240;

  // Build ordered rows from top (logits) to bottom (tokens).
  const rows = [];
  rows.push({ stageId: "logits", kind: "logits", h: rowH });
  rows.push({ stageId: "head", kind: "head", h: rowH });
  rows.push({ stageId: "lnf", kind: "lnf", h: rowH });

  if (showAllBlocks) {
    for (let i = n - 1; i >= 0; i--) {
      rows.push({ stageId: `block-${i}`, kind: "block", index: i, h: blockH });
    }
  } else {
    // top block, an ellipsis for the hidden middle, then the bottom block.
    rows.push({ stageId: `block-${n - 1}`, kind: "block", index: n - 1, h: blockH });
    rows.push({ kind: "ellipsis", hidden: n - 2, h: 38 });
    rows.push({ stageId: "block-0", kind: "block", index: 0, h: blockH });
  }

  rows.push({ stageId: "embed", kind: "embed", h: rowH });
  rows.push({ stageId: "tokens", kind: "tokens", h: rowH + 6 });

  // Assign y positions (top-first).
  let y = 16;
  const placed = rows.map((r) => {
    const top = y;
    y += r.h + gap;
    return { ...r, top, mid: top + r.h / 2 };
  });
  const totalH = y + 6;

  // Map stageId -> mid-y so the pulse and residual line can find coordinates.
  const midOf = {};
  placed.forEach((r) => {
    if (r.stageId) midOf[r.stageId] = r.mid;
  });

  const residualTop = midOf["embed"] ?? 0;
  const residualBottom = midOf["lnf"] ?? totalH;

  // Pulse coordinate: snap to the nearest visible stage row.
  const pulseStage = pulse >= 0 && pulse < stages.length ? stages[pulse] : null;
  const pulseY =
    pulseStage && midOf[pulseStage.id] != null ? midOf[pulseStage.id] : null;

  const stepPulse = () => setPulse((p) => (p + 1) % stages.length);
  const resetPulse = () => setPulse(-1);

  const isSel = (id) => id === selected;

  return (
    <div className="flex flex-col gap-5 lg:flex-row">
      {/* ---- Controls + SVG diagram ---- */}
      <div className="lg:w-[60%]">
        <div className="mb-4 flex flex-wrap items-end gap-4">
          <div className="w-56">
            <Slider
              label="Layers  N (n_layer)"
              min={1}
              max={12}
              value={n}
              onChange={(v) => setN(Math.round(v))}
              accent="violet"
              format={(v) => `${v}`}
            />
          </div>
          <Button onClick={stepPulse} active={pulse >= 0}>
            Step signal ▲
          </Button>
          <Button onClick={resetPulse}>↻ Reset</Button>
          <span className="text-xs text-slate-500">Click any component to inspect it.</span>
        </div>

        <div className="overflow-x-auto scrollbar-thin">
          <svg
            width={W}
            height={totalH}
            className="mx-auto block"
            style={{ maxWidth: "100%" }}
          >
            {/* Residual stream: a single vertical highway behind the blocks. */}
            <line
              x1={cx}
              y1={residualTop}
              x2={cx}
              y2={residualBottom}
              stroke="#27314a"
              strokeWidth={10}
              strokeLinecap="round"
            />
            <line
              x1={cx}
              y1={residualTop}
              x2={cx}
              y2={residualBottom}
              stroke="#3a55f5"
              strokeWidth={2}
              strokeDasharray="2 5"
              opacity={0.55}
            />

            {/* Flow connectors between every adjacent row (thin guides). */}
            {placed.slice(0, -1).map((r, i) => {
              const next = placed[i + 1];
              return (
                <line
                  key={`c-${i}`}
                  x1={cx}
                  y1={r.top + r.h}
                  x2={cx}
                  y2={next.top}
                  stroke="#1e2740"
                  strokeWidth={2}
                />
              );
            })}

            {/* The travelling signal pulse. Colored by how far up the stack it
                has reached, using the shared heat ramp (ink -> cyan -> white). */}
            {pulseY != null && (
              <g>
                <circle
                  cx={cx}
                  cy={pulseY}
                  r={11}
                  fill={heat(0.4 + 0.6 * (pulse / (stages.length - 1)))}
                  opacity={0.25}
                />
                <circle
                  cx={cx}
                  cy={pulseY}
                  r={5.5}
                  fill={heat(0.5 + 0.5 * (pulse / (stages.length - 1)))}
                />
              </g>
            )}

            {/* Rows. */}
            {placed.map((r, i) => {
              if (r.kind === "ellipsis") {
                return (
                  <g key={`row-${i}`}>
                    <text
                      x={cx}
                      y={r.mid + 5}
                      textAnchor="middle"
                      fontSize="20"
                      fill="#64748b"
                    >
                      ⋮
                    </text>
                    <text
                      x={cx + 80}
                      y={r.mid + 4}
                      textAnchor="middle"
                      fontSize="10"
                      fontFamily="JetBrains Mono, monospace"
                      fill="#475569"
                    >
                      ×{r.hidden} more
                    </text>
                  </g>
                );
              }

              if (r.kind === "tokens") {
                const sel = isSel("tokens");
                const chipW = 46;
                const totalChips = TOKENS.length * (chipW + 6) - 6;
                const startX = cx - totalChips / 2;
                return (
                  <g
                    key={`row-${i}`}
                    onClick={() => setSelected("tokens")}
                    style={{ cursor: "pointer" }}
                  >
                    {TOKENS.map((t, k) => (
                      <g key={k}>
                        <rect
                          x={startX + k * (chipW + 6)}
                          y={r.top + 6}
                          width={chipW}
                          height={30}
                          rx={6}
                          fill={sel ? "#243056" : "#161c2e"}
                          stroke={sel ? "#8eabff" : "#334155"}
                        />
                        <text
                          x={startX + k * (chipW + 6) + chipW / 2}
                          y={r.top + 25}
                          textAnchor="middle"
                          fontSize="11"
                          fontFamily="JetBrains Mono, monospace"
                          fill="#cbd5e1"
                        >
                          {t}
                        </text>
                      </g>
                    ))}
                    <text
                      x={cx}
                      y={r.top + r.h - 2}
                      textAnchor="middle"
                      fontSize="9"
                      fill="#475569"
                    >
                      token IDs · (B, T)
                    </text>
                  </g>
                );
              }

              // Generic boxed row (embed / lnf / head / logits) and block boxes.
              const meta = INFO[r.kind];
              const sel = isSel(r.stageId);
              const isBlock = r.kind === "block";
              const label = isBlock
                ? `Block ${r.index + 1}`
                : r.kind === "embed"
                ? "Embeddings  +  Positions"
                : r.kind === "lnf"
                ? "Final LayerNorm"
                : r.kind === "head"
                ? "LM Head  (Linear d → V)"
                : "Logits  →  softmax";

              return (
                <g
                  key={`row-${i}`}
                  onClick={() => setSelected(r.stageId)}
                  style={{ cursor: "pointer" }}
                >
                  <rect
                    x={cx - boxW / 2}
                    y={r.top}
                    width={boxW}
                    height={r.h}
                    rx={9}
                    fill={sel ? "#1c2438" : "#141a2e"}
                    stroke={sel ? meta.color : "rgba(255,255,255,0.10)"}
                    strokeWidth={sel ? 2 : 1}
                  />
                  {/* color tab on the left edge */}
                  <rect
                    x={cx - boxW / 2}
                    y={r.top}
                    width={5}
                    height={r.h}
                    rx={2}
                    fill={meta.color}
                    opacity={sel ? 1 : 0.7}
                  />
                  <text
                    x={cx}
                    y={isBlock ? r.top + 20 : r.mid + 4}
                    textAnchor="middle"
                    fontSize="12.5"
                    fontWeight="600"
                    fill={sel ? "#f1f5f9" : "#cbd5e1"}
                  >
                    {label}
                  </text>

                  {/* For block rows, draw the internal sub-step pills. */}
                  {isBlock && (
                    <g>
                      {SUBSTEPS.map((s, k) => {
                        const pw = 30;
                        const pgap = 5;
                        const totalW = SUBSTEPS.length * (pw + pgap) - pgap;
                        const sx = cx - totalW / 2 + k * (pw + pgap);
                        const isAdd = s === "⊕";
                        return (
                          <g key={k}>
                            <rect
                              x={sx}
                              y={r.top + 34}
                              width={pw}
                              height={26}
                              rx={5}
                              fill={isAdd ? "transparent" : "#0e1322"}
                              stroke={isAdd ? "#3a55f5" : "#27314a"}
                              strokeDasharray={isAdd ? "3 2" : undefined}
                            />
                            <text
                              x={sx + pw / 2}
                              y={r.top + 51}
                              textAnchor="middle"
                              fontSize="10"
                              fontFamily="JetBrains Mono, monospace"
                              fill={isAdd ? "#8eabff" : "#94a3b8"}
                            >
                              {s}
                            </text>
                          </g>
                        );
                      })}
                    </g>
                  )}

                  {/* shape badge on the right for non-block rows */}
                  {!isBlock && (
                    <text
                      x={cx + boxW / 2 - 10}
                      y={r.mid + 4}
                      textAnchor="end"
                      fontSize="9.5"
                      fontFamily="JetBrains Mono, monospace"
                      fill="#475569"
                    >
                      {meta.shape}
                    </text>
                  )}
                </g>
              );
            })}
          </svg>
        </div>
      </div>

      {/* ---- Side panel ---- */}
      <div className="lg:w-[40%]">
        <div className="sticky top-4 rounded-xl border border-white/10 bg-ink-900/60 p-4">
          <div className="mb-1 flex items-center gap-2">
            <span
              className="inline-block h-3 w-3 rounded-sm"
              style={{ background: info.color }}
            />
            <span className="text-[10px] uppercase tracking-wider text-slate-500">
              Selected component
            </span>
          </div>
          <div className="mb-2 text-lg font-semibold text-slate-100">{info.name}</div>

          <div className="mb-3 inline-flex items-center gap-2 rounded-lg border border-white/10 bg-ink-800/70 px-3 py-1.5">
            <span className="text-[10px] uppercase tracking-wider text-slate-500">shape</span>
            <span className="font-mono text-sm text-cyan-200">{info.shape}</span>
          </div>

          <ul className="space-y-1.5 text-sm leading-relaxed text-slate-300">
            {info.lines.map((l, i) => (
              <li key={i} className="flex gap-2">
                <span className="mt-1 inline-block h-1 w-1 shrink-0 rounded-full bg-slate-600" />
                <span>{l}</span>
              </li>
            ))}
          </ul>

          {selStage.kind === "block" && (
            <div className="mt-3 rounded-lg border border-violet-500/30 bg-violet-500/[0.06] p-3">
              <div className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-violet-200">
                Inside the block
              </div>
              <div className="font-mono text-xs leading-relaxed text-slate-300">
                x ← x + Attn(LN(x))
                <br />
                x ← x + FFN(LN(x))
              </div>
            </div>
          )}

          <div className="mt-3 text-xs text-slate-500">
            Deep dive:{" "}
            <a href={`/chapter/${info.slug}`} className="prose-link">
              {info.chapter}
            </a>
          </div>

          <div className="mt-4 grid grid-cols-3 gap-2 text-center">
            <PanelStat label="layers" value={n} />
            <PanelStat label="d_model" value={D} />
            <PanelStat label="vocab" value={VOCAB} />
          </div>
          <div className="mt-2 text-[11px] leading-relaxed text-slate-600">
            Toy dimensions for the diagram. The residual stream stays{" "}
            <span className="font-mono text-slate-400">(B, T, d)</span> from the embeddings
            all the way to the final norm — every block reads it and writes back into it.
          </div>
        </div>
      </div>
    </div>
  );
}

function PanelStat({ label, value }) {
  return (
    <div className="rounded-lg border border-white/10 bg-ink-800/60 px-2 py-1.5">
      <div className="text-[9px] uppercase tracking-wider text-slate-500">{label}</div>
      <div className="font-mono text-sm text-slate-200">{value}</div>
    </div>
  );
}
