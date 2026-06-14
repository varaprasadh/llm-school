import { useMemo, useState } from "react";
import { Slider } from "../ui";
import { heat } from "../scales";

/**
 * ModuleTree — an interactive, nested-box view of the GPT module hierarchy
 * (the object tree you'd get from `print(model)` in PyTorch), matching the
 * AttentionExplorer / TransformerStack house style.
 *
 * The tree mirrors the code built in this chapter, module for module:
 *
 *   GPT
 *   ├─ transformer
 *   │  ├─ wte   Embedding(vocab, d)
 *   │  ├─ wpe   Embedding(block, d)
 *   │  ├─ drop  Dropout
 *   │  ├─ h  ModuleList[ Block × n_layer ]
 *   │  │     └─ Block
 *   │  │        ├─ ln_1  LayerNorm
 *   │  │        ├─ attn  CausalSelfAttention { c_attn, c_proj, ... }
 *   │  │        ├─ ln_2  LayerNorm
 *   │  │        └─ mlp   MLP { c_fc, c_proj, ... }
 *   │  └─ ln_f  LayerNorm
 *   └─ lm_head  Linear(d, vocab)  ← weight-tied to wte (0 extra params)
 *
 * Two <Slider>s tweak n_layer and n_embd; every node's parameter count and the
 * running TOTAL update live. Hover or click any node to inspect its type, the
 * tensor shape it outputs, and its parameter count. Fully deterministic — the
 * only state is the two integers and the hovered/selected node id.
 */

// Fixed dims for the toy config (only n_layer & n_embd are user-tunable, to
// keep the param arithmetic legible). These match the sanity-check config in
// the chapter text.
const VOCAB = 65; // tiny char-level vocab (e.g. Shakespeare)
const BLOCK = 64; // context length
const N_HEAD = 4; // heads (must divide n_embd; we round n_embd to a multiple)

const fmtParams = (p) => {
  if (p >= 1e6) return `${(p / 1e6).toFixed(2)}M`;
  if (p >= 1e3) return `${(p / 1e3).toFixed(1)}K`;
  return `${p}`;
};
const fmtFull = (p) => p.toLocaleString("en-US");

// Param counts for each leaf, given the config. nanoGPT-faithful:
//   Linear(in,out) with bias  -> in*out + out
//   LayerNorm(d) with affine   -> 2*d
//   Embedding(rows, d)         -> rows*d
//   lm_head weight is TIED to wte, so it contributes 0 new parameters.
function leafParams(kind, d) {
  switch (kind) {
    case "wte":
      return VOCAB * d;
    case "wpe":
      return BLOCK * d;
    case "ln":
      return 2 * d;
    case "c_attn":
      return d * (3 * d) + 3 * d; // Linear d -> 3d (Q,K,V fused)
    case "c_proj_attn":
      return d * d + d; // Linear d -> d
    case "c_fc":
      return d * (4 * d) + 4 * d; // Linear d -> 4d
    case "c_proj_mlp":
      return 4 * d * d + d; // Linear 4d -> d
    case "lm_head":
      return 0; // weight-tied to wte
    case "drop":
      return 0;
    default:
      return 0;
  }
}

// Output shape a node produces, as a display string. <M>-style but plain text
// because this lives inside SVG / pills.
function shapeOf(kind) {
  switch (kind) {
    case "wte":
    case "wpe":
    case "drop":
    case "ln":
    case "c_proj_attn":
    case "c_proj_mlp":
    case "block":
    case "h":
    case "transformer":
      return "(B, T, d)";
    case "c_attn":
      return "(B, T, 3d)";
    case "c_fc":
      return "(B, T, 4d)";
    case "lm_head":
    case "gpt":
      return "(B, T, V)";
    default:
      return "(B, T, d)";
  }
}

const DESC = {
  gpt: "The top-level container. Holds the transformer trunk and the language-model head; ties their weights and runs the forward pass.",
  transformer:
    "An nn.ModuleDict bundling the embeddings, dropout, the stack of blocks, and the final norm — everything except the output head.",
  wte: "Token embedding table. Row i is the learned d-dim vector for token id i. lm_head shares this exact weight matrix.",
  wpe: "Positional embedding table. Row p is the vector added at position p, so the model can tell token order apart.",
  drop: "Dropout applied to the summed embeddings. Zero parameters — it only zeros activations at train time.",
  h: "nn.ModuleList of n_layer identical Blocks, applied in sequence. This is where almost all the parameters live.",
  block:
    "One pre-norm transformer block: x = x + attn(ln_1(x)); x = x + mlp(ln_2(x)). Input and output shapes are identical.",
  ln: "LayerNorm over the last (feature) dimension, with a learned scale (weight) and shift (bias): 2·d parameters.",
  attn: "Causal multi-head self-attention. Fuses Q,K,V into one projection, masks the future, then projects back.",
  c_attn: "Fused QKV projection: a single Linear from d to 3d, later split into the query, key and value tensors.",
  c_proj_attn: "Output projection of the attention block: Linear d → d, mixing the concatenated head outputs.",
  mlp: "Position-wise feed-forward network: expand to 4d, apply GELU, project back to d. The bulk of per-block params.",
  c_fc: "Up-projection of the MLP: Linear d → 4d, widening the representation before the nonlinearity.",
  c_proj_mlp: "Down-projection of the MLP: Linear 4d → d, returning to the residual-stream width.",
  lm_head:
    "The unembedding: Linear d → V (no bias). Its weight is TIED to wte, so it adds zero new parameters but still produces logits over the vocab.",
};

const COL = {
  gpt: "#5b7dff",
  transformer: "#a855f7",
  wte: "#34d399",
  wpe: "#34d399",
  drop: "#64748b",
  h: "#a855f7",
  block: "#a855f7",
  ln: "#22d3ee",
  attn: "#f59e0b",
  c_attn: "#f59e0b",
  c_proj_attn: "#f59e0b",
  mlp: "#fb7185",
  c_fc: "#fb7185",
  c_proj_mlp: "#fb7185",
  lm_head: "#5b7dff",
};

// Build the per-block leaf list with computed params, given d.
function blockLeaves(d) {
  return [
    { id: "ln_1", kind: "ln", label: "ln_1", type: "LayerNorm(d)", params: leafParams("ln", d) },
    {
      id: "attn",
      kind: "attn",
      label: "attn",
      type: "CausalSelfAttention",
      // a parent grouping its own leaves:
      children: [
        { id: "c_attn", kind: "c_attn", label: "c_attn", type: "Linear(d, 3d)", params: leafParams("c_attn", d) },
        { id: "c_proj", kind: "c_proj_attn", label: "c_proj", type: "Linear(d, d)", params: leafParams("c_proj_attn", d) },
      ],
    },
    { id: "ln_2", kind: "ln", label: "ln_2", type: "LayerNorm(d)", params: leafParams("ln", d) },
    {
      id: "mlp",
      kind: "mlp",
      label: "mlp",
      type: "MLP",
      children: [
        { id: "c_fc", kind: "c_fc", label: "c_fc", type: "Linear(d, 4d)", params: leafParams("c_fc", d) },
        { id: "c_proj", kind: "c_proj_mlp", label: "c_proj", type: "Linear(4d, d)", params: leafParams("c_proj_mlp", d) },
      ],
    },
  ];
}

// Sum params of a node (and its descendants).
function sumParams(node) {
  if (node.params != null) return node.params;
  if (!node.children) return 0;
  return node.children.reduce((s, c) => s + sumParams(c), 0);
}

export default function ModuleTree() {
  const [nLayer, setNLayer] = useState(4);
  // n_embd is rounded to a multiple of N_HEAD so heads divide evenly.
  const [nEmbdRaw, setNEmbdRaw] = useState(128);
  const d = Math.max(N_HEAD, Math.round(nEmbdRaw / N_HEAD) * N_HEAD);

  const [hover, setHover] = useState(null);
  const [selected, setSelected] = useState("gpt");

  // ---- Build the tree (memoized on the two knobs) ------------------------
  const { tree, total, byId } = useMemo(() => {
    const oneBlock = (i) => ({
      id: `block-${i}`,
      kind: "block",
      label: `h[${i}]`,
      type: "Block",
      children: blockLeaves(d).map((c) => ({ ...c, id: `block-${i}.${c.id}` })),
    });

    const blocks = Array.from({ length: nLayer }, (_, i) => oneBlock(i));

    const transformer = {
      id: "transformer",
      kind: "transformer",
      label: "transformer",
      type: "ModuleDict",
      children: [
        { id: "wte", kind: "wte", label: "wte", type: `Embedding(${VOCAB}, d)`, params: leafParams("wte", d) },
        { id: "wpe", kind: "wpe", label: "wpe", type: `Embedding(${BLOCK}, d)`, params: leafParams("wpe", d) },
        { id: "drop", kind: "drop", label: "drop", type: "Dropout", params: 0 },
        {
          id: "h",
          kind: "h",
          label: "h",
          type: `ModuleList[Block × ${nLayer}]`,
          children: blocks,
        },
        { id: "ln_f", kind: "ln", label: "ln_f", type: "LayerNorm(d)", params: leafParams("ln", d) },
      ],
    };

    const root = {
      id: "gpt",
      kind: "gpt",
      label: "GPT",
      type: "nn.Module",
      children: [
        transformer,
        { id: "lm_head", kind: "lm_head", label: "lm_head", type: "Linear(d, V) · tied", params: 0 },
      ],
    };

    // Index every node by id for the side panel & precompute subtree sums.
    const index = {};
    const walk = (n) => {
      index[n.id] = n;
      n._params = sumParams(n);
      if (n.children) n.children.forEach(walk);
    };
    walk(root);

    return { tree: root, total: root._params, byId: index };
  }, [d, nLayer]);

  const activeId = hover || selected;
  const active = byId[activeId] || tree;

  // Max leaf params (for the heat-coloured bars) — use c_fc which is the largest
  // single leaf, so bars are comparable across the tree.
  const maxLeaf = leafParams("c_fc", d);

  return (
    <div className="flex flex-col gap-5 lg:flex-row">
      {/* ---- Controls + nested-box tree ---- */}
      <div className="lg:w-[62%]">
        <div className="mb-4 flex flex-wrap items-end gap-5">
          <div className="w-52">
            <Slider
              label="n_layer"
              min={1}
              max={8}
              value={nLayer}
              onChange={(v) => setNLayer(Math.round(v))}
              accent="violet"
              format={(v) => `${v}`}
            />
          </div>
          <div className="w-52">
            <Slider
              label="n_embd (d)"
              min={N_HEAD * 4}
              max={384}
              step={N_HEAD}
              value={d}
              onChange={(v) => setNEmbdRaw(v)}
              accent="cyan"
              format={(v) => `${v}`}
            />
          </div>
          <div className="rounded-lg border border-brand-400/40 bg-brand-500/10 px-3 py-2">
            <div className="text-[10px] uppercase tracking-wider text-slate-400">Total parameters</div>
            <div className="font-mono text-xl font-semibold text-brand-100">{fmtParams(total)}</div>
            <div className="font-mono text-[10px] text-slate-500">{fmtFull(total)}</div>
          </div>
        </div>

        <div className="text-xs text-slate-500 mb-2">
          Hover a node to inspect it; the bar length and colour scale with its parameter share.
        </div>

        <div className="rounded-xl border border-white/10 bg-ink-900/40 p-3">
          <TreeNode
            node={tree}
            depth={0}
            total={total}
            maxLeaf={maxLeaf}
            activeId={activeId}
            onHover={setHover}
            onSelect={setSelected}
          />
        </div>
      </div>

      {/* ---- Side panel ---- */}
      <div className="lg:w-[38%]">
        <div className="sticky top-4 rounded-xl border border-white/10 bg-ink-900/60 p-4">
          <div className="mb-1 flex items-center gap-2">
            <span className="inline-block h-3 w-3 rounded-sm" style={{ background: COL[active.kind] }} />
            <span className="text-[10px] uppercase tracking-wider text-slate-500">Selected module</span>
          </div>
          <div className="mb-1 font-mono text-lg font-semibold text-slate-100">{active.label}</div>
          <div className="mb-3 text-xs font-mono text-slate-400">{active.type}</div>

          <div className="mb-3 flex flex-wrap gap-2">
            <span className="inline-flex items-center gap-2 rounded-lg border border-white/10 bg-ink-800/70 px-3 py-1.5">
              <span className="text-[10px] uppercase tracking-wider text-slate-500">output</span>
              <span className="font-mono text-sm text-cyan-200">{shapeOf(active.kind)}</span>
            </span>
            <span className="inline-flex items-center gap-2 rounded-lg border border-white/10 bg-ink-800/70 px-3 py-1.5">
              <span className="text-[10px] uppercase tracking-wider text-slate-500">params</span>
              <span className="font-mono text-sm text-amber-200">{fmtFull(active._params ?? 0)}</span>
            </span>
          </div>

          <p className="text-sm leading-relaxed text-slate-300">{DESC[active.kind]}</p>

          {active._params > 0 && (
            <div className="mt-3">
              <div className="mb-1 flex items-center justify-between text-[11px] text-slate-500">
                <span>share of total</span>
                <span className="font-mono">{((active._params / total) * 100).toFixed(1)}%</span>
              </div>
              <div className="h-2 w-full overflow-hidden rounded-full bg-ink-800">
                <div
                  className="h-full rounded-full"
                  style={{
                    width: `${Math.max(2, (active._params / total) * 100)}%`,
                    background: COL[active.kind],
                  }}
                />
              </div>
            </div>
          )}

          <div className="mt-4 grid grid-cols-3 gap-2 text-center">
            <PanelStat label="n_layer" value={nLayer} />
            <PanelStat label="n_embd" value={d} />
            <PanelStat label="n_head" value={N_HEAD} />
            <PanelStat label="vocab" value={VOCAB} />
            <PanelStat label="block" value={BLOCK} />
            <PanelStat label="d / head" value={d / N_HEAD} />
          </div>

          <div className="mt-3 rounded-lg border border-emerald-500/30 bg-emerald-500/[0.06] p-3 text-[11px] leading-relaxed text-slate-400">
            <span className="font-semibold text-emerald-200">lm_head is weight-tied</span> to{" "}
            <span className="font-mono text-slate-300">wte</span>, so although it produces{" "}
            <span className="font-mono text-slate-300">(B, T, V)</span> logits it adds{" "}
            <span className="font-mono text-slate-300">0</span> parameters to the total.
          </div>
        </div>
      </div>
    </div>
  );
}

// A single, indentable row in the tree. Parents render a faint container box
// around their children; leaves render a labelled param bar.
function TreeNode({ node, depth, total, maxLeaf, activeId, onHover, onSelect }) {
  const isActive = node.id === activeId;
  const color = COL[node.kind] || "#64748b";
  const isLeaf = !node.children;
  const params = node._params ?? 0;
  const barFrac = maxLeaf > 0 ? Math.min(1, params / maxLeaf) : 0;

  const handlers = {
    onMouseEnter: () => onHover(node.id),
    onMouseLeave: () => onHover(null),
    onClick: (e) => {
      e.stopPropagation();
      onSelect(node.id);
    },
  };

  return (
    <div className="select-none" style={{ marginLeft: depth === 0 ? 0 : 12 }}>
      <div
        {...handlers}
        className={`group flex cursor-pointer items-center gap-2 rounded-md px-2 py-1 transition-colors ${
          isActive ? "bg-white/[0.06]" : "hover:bg-white/[0.03]"
        }`}
        style={{ outline: isActive ? `1px solid ${color}` : "none" }}
      >
        <span className="inline-block h-2.5 w-2.5 shrink-0 rounded-sm" style={{ background: color }} />
        <span className="font-mono text-[13px] text-slate-200">{node.label}</span>
        <span className="font-mono text-[11px] text-slate-500">{node.type}</span>

        {/* param bar + count, right-aligned */}
        <span className="ml-auto flex items-center gap-2">
          {isLeaf && params > 0 && (
            <span className="hidden h-1.5 w-20 overflow-hidden rounded-full bg-ink-800 sm:inline-block">
              <span
                className="block h-full rounded-full"
                style={{ width: `${Math.max(4, barFrac * 100)}%`, background: heat(0.3 + barFrac * 0.7) }}
              />
            </span>
          )}
          <span
            className={`font-mono text-[11px] tabular-nums ${
              params > 0 ? "text-slate-400" : "text-slate-600"
            }`}
          >
            {params > 0 ? fmtParams(params) : "0"}
          </span>
        </span>
      </div>

      {node.children && (
        <div
          className="mt-0.5 border-l pl-1"
          style={{ borderColor: isActive ? color : "rgba(255,255,255,0.07)" }}
        >
          {node.children.map((c) => (
            <TreeNode
              key={c.id}
              node={c}
              depth={depth + 1}
              total={total}
              maxLeaf={maxLeaf}
              activeId={activeId}
              onHover={onHover}
              onSelect={onSelect}
            />
          ))}
        </div>
      )}
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
