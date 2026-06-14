import { useState } from "react";
import { Button } from "../ui";
import { heat } from "../scales";

/**
 * PipelineRecap — the capstone "you learned all of this" walkthrough.
 *
 * A horizontal (wrapping) sequence of the ~10 production stages of building an
 * LLM. Click a stage to reveal a card describing, for that stage:
 *   • input  → output artifact   (what you start with → what you produce)
 *   • the key DECISION you make there
 *   • the one tool / technique that defines it
 *   • the chapter(s) it maps to (rendered as text — no router dependency)
 *
 * A cumulative "What you have now" panel builds up as you advance: every stage
 * at or before the selected one contributes one concrete artifact, so by the
 * final stage you can see the whole chain you assembled — from raw HTML to a
 * token streaming back to a user.
 *
 * Fully deterministic: the only state is the selected index. Colors come from
 * the shared `heat` ramp so the sequence reads as a gradient of progress.
 */

// The ten stages. `chip` is the artifact added to the cumulative inventory when
// this stage completes — phrased as a noun you now "have".
const STAGES = [
  {
    key: "data",
    label: "Data",
    icon: "🌐",
    input: "The open web (Common Crawl, code, books, papers)",
    output: "A clean, deduplicated token corpus",
    decision: "What to keep and what to throw away — quality filtering, dedup, decontamination.",
    tool: "MinHash + LSH dedup, fastText/quality classifiers",
    chapter: "Ch. 4 · Data Collection, Cleaning & Curation",
    chip: "≈20B+ clean tokens",
  },
  {
    key: "tokenize",
    label: "Tokenize",
    icon: "🔡",
    input: "Raw UTF-8 text",
    output: "A vocabulary + sequences of integer IDs",
    decision: "Vocabulary size and the merge rules (or just reuse an existing tokenizer).",
    tool: "Byte-Pair Encoding (BPE) / SentencePiece",
    chapter: "Ch. 5 · Tokenization & Byte-Pair Encoding",
    chip: "A 32k–50k BPE vocabulary",
  },
  {
    key: "arch",
    label: "Architecture",
    icon: "🏛️",
    input: "A hyperparameter budget (depth, width, heads, context)",
    output: "An initialized GPT module — random weights",
    decision: "n_layer, n_head, n_embd, block_size — and the positional scheme (RoPE).",
    tool: "Decoder-only Transformer (attention + MLP + residual + norm)",
    chapter: "Ch. 6–11 · Embeddings → Attention → Blocks",
    chip: "A ~1B-param GPT (untrained)",
  },
  {
    key: "pretrain",
    label: "Pretrain",
    icon: "🧠",
    input: "Corpus + initialized model",
    output: "A fluent base model (the expensive checkpoint)",
    decision: "Compute-optimal token/param split (Chinchilla), LR schedule, batch size.",
    tool: "AdamW + cosine LR, next-token cross-entropy, distributed training",
    chapter: "Ch. 12–17 · Loss, Optimizers, Training Loop, Scaling",
    chip: "A base model (next-token predictor)",
  },
  {
    key: "eval",
    label: "Evaluate",
    icon: "📊",
    input: "A checkpoint to interrogate",
    output: "A scorecard: capability, safety, regressions",
    decision: "Which benchmarks to trust, and how to avoid fooling yourself (contamination).",
    tool: "Perplexity, MMLU, HumanEval, LLM-as-judge",
    chapter: "Ch. 21 · Evaluation & Benchmarks",
    chip: "A benchmark scorecard",
  },
  {
    key: "sft",
    label: "SFT",
    icon: "🎯",
    input: "Base model + curated instruction→response pairs",
    output: "An instruction-following model",
    decision: "The chat template, and masking the loss so only responses are learned.",
    tool: "Supervised fine-tuning (often with LoRA/QLoRA)",
    chapter: "Ch. 18 · Supervised Fine-Tuning · Ch. 20 · LoRA",
    chip: "An instruction-tuned model",
  },
  {
    key: "align",
    label: "Align",
    icon: "⚖️",
    input: "SFT model + human preference comparisons",
    output: "A helpful, honest, harmless assistant",
    decision: "Preference objective (PPO vs. the simpler DPO) and the reward signal.",
    tool: "RLHF / Direct Preference Optimization (DPO)",
    chapter: "Ch. 19 · RLHF, Reward Models & DPO",
    chip: "An aligned assistant",
  },
  {
    key: "optimize",
    label: "Optimize",
    icon: "⚡",
    input: "An aligned model (full precision, slow)",
    output: "A small, fast, cheap-to-serve model",
    decision: "How much quality to trade for latency — quantization bits, what to cache.",
    tool: "KV cache, quantization, FlashAttention, distillation",
    chapter: "Ch. 22 · Inference Optimization",
    chip: "A quantized, low-latency model",
  },
  {
    key: "deploy",
    label: "Deploy",
    icon: "🚀",
    input: "An optimized model + an inference server",
    output: "A production API streaming tokens to users",
    decision: "Batching strategy and autoscaling to hit your latency/cost SLOs.",
    tool: "vLLM, continuous batching, paged attention",
    chapter: "Ch. 23 · Deployment & Serving at Scale",
    chip: "A live tokens API",
  },
  {
    key: "monitor",
    label: "Monitor",
    icon: "🛡️",
    input: "Live traffic + the deployed model",
    output: "A guarded, observed, continually improved service",
    decision: "What to log, which guardrails to enforce, when to retrain on drift.",
    tool: "Observability, red-teaming, content filters, jailbreak defense",
    chapter: "Ch. 24 · Monitoring, Safety & Guardrails",
    chip: "Guardrails + telemetry",
  },
];

export default function PipelineRecap() {
  const [sel, setSel] = useState(0);
  const stage = STAGES[sel];
  // Color the selected stage by its position along the pipeline.
  const tOf = (i) => 0.18 + (i / (STAGES.length - 1)) * 0.74;
  const accent = heat(tOf(sel));

  return (
    <div className="space-y-4">
      {/* ── The stage rail ─────────────────────────────────────────────── */}
      <div className="flex flex-wrap items-stretch gap-1.5">
        {STAGES.map((s, i) => {
          const done = i <= sel;
          const isSel = i === sel;
          const c = heat(tOf(i));
          return (
            <div key={s.key} className="flex items-center">
              <button
                onClick={() => setSel(i)}
                aria-pressed={isSel}
                className="group flex min-w-[78px] flex-col items-center gap-1.5 rounded-xl border px-2 py-2.5 transition-all"
                style={{
                  borderColor: isSel ? c : done ? "rgba(255,255,255,0.18)" : "rgba(255,255,255,0.07)",
                  background: isSel ? `${c}26` : done ? "rgba(255,255,255,0.04)" : "rgba(20,26,46,0.5)",
                  opacity: done ? 1 : 0.6,
                }}
              >
                <span
                  className="flex h-9 w-9 items-center justify-center rounded-lg text-lg transition-transform group-hover:scale-110"
                  style={{ background: done ? `${c}22` : "rgba(255,255,255,0.04)" }}
                >
                  {s.icon}
                </span>
                <span
                  className="text-[11px] font-semibold sm:text-xs"
                  style={{ color: isSel ? c : done ? "#cbd5e1" : "#64748b" }}
                >
                  {s.label}
                </span>
              </button>
              {i < STAGES.length - 1 && (
                <svg width="16" height="20" viewBox="0 0 16 20" className="shrink-0" aria-hidden>
                  <line
                    x1="0"
                    y1="10"
                    x2="14"
                    y2="10"
                    stroke={i < sel ? heat(tOf(i + 1)) : "#33415580"}
                    strokeWidth="2"
                    strokeDasharray="3 3"
                  />
                  <path
                    d="M10 6 L15 10 L10 14"
                    fill="none"
                    stroke={i < sel ? heat(tOf(i + 1)) : "#475569"}
                    strokeWidth="1.6"
                  />
                </svg>
              )}
            </div>
          );
        })}
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <Button onClick={() => setSel((v) => Math.max(0, v - 1))} disabled={sel === 0}>
          ◀ Prev
        </Button>
        <Button onClick={() => setSel((v) => Math.min(STAGES.length - 1, v + 1))} disabled={sel === STAGES.length - 1}>
          Next ▶
        </Button>
        <span className="ml-1 text-xs text-slate-500">
          Stage {sel + 1} of {STAGES.length} — click any stage to inspect it.
        </span>
      </div>

      {/* ── Detail + cumulative inventory ──────────────────────────────── */}
      <div className="grid gap-4 lg:grid-cols-[1.5fr_1fr]">
        {/* Stage detail card */}
        <div
          className="rounded-xl border bg-ink-900/50 p-4"
          style={{ borderColor: `${accent}66` }}
        >
          <div className="mb-3 flex items-center gap-2">
            <span
              className="flex h-8 w-8 items-center justify-center rounded-lg text-lg"
              style={{ background: `${accent}22` }}
            >
              {stage.icon}
            </span>
            <span className="text-base font-semibold text-slate-100">{stage.label}</span>
            <span className="ml-auto font-mono text-[11px] text-slate-500">{stage.chapter}</span>
          </div>

          {/* input -> output artifact */}
          <div className="flex flex-col gap-2 sm:flex-row sm:items-stretch">
            <div className="flex-1 rounded-lg border border-white/10 bg-ink-900/60 p-2.5">
              <div className="text-[10px] uppercase tracking-wider text-slate-500">Input</div>
              <div className="mt-0.5 text-[13px] leading-snug text-slate-300">{stage.input}</div>
            </div>
            <div className="flex items-center justify-center text-slate-500 sm:px-1">
              <span className="hidden sm:inline" style={{ color: accent }}>
                ➜
              </span>
              <span className="sm:hidden" style={{ color: accent }}>
                ↓
              </span>
            </div>
            <div
              className="flex-1 rounded-lg border p-2.5"
              style={{ borderColor: `${accent}55`, background: `${accent}12` }}
            >
              <div className="text-[10px] uppercase tracking-wider text-slate-400">Output artifact</div>
              <div className="mt-0.5 text-[13px] font-medium leading-snug text-slate-100">{stage.output}</div>
            </div>
          </div>

          {/* decision */}
          <div className="mt-3">
            <div className="text-[10px] uppercase tracking-wider text-slate-500">The key decision</div>
            <p className="mt-0.5 text-[13px] leading-relaxed text-slate-300/90">{stage.decision}</p>
          </div>

          {/* the one technique */}
          <div className="mt-3 flex items-start gap-2 rounded-lg border border-white/10 bg-black/20 p-2.5">
            <span className="mt-0.5 text-[10px] uppercase tracking-wider text-slate-500">Tool</span>
            <span className="font-mono text-[12px] leading-snug text-cyan-200">{stage.tool}</span>
          </div>
        </div>

        {/* Cumulative "what you have now" */}
        <div className="rounded-xl border border-white/10 bg-ink-900/60 p-4">
          <div className="mb-2 flex items-baseline justify-between">
            <span className="text-[10px] uppercase tracking-wider text-slate-500">What you have now</span>
            <span className="font-mono text-[11px]" style={{ color: accent }}>
              {sel + 1}/{STAGES.length}
            </span>
          </div>
          <ol className="space-y-1.5">
            {STAGES.map((s, i) => {
              const has = i <= sel;
              const c = heat(tOf(i));
              return (
                <li
                  key={s.key}
                  className="flex items-center gap-2 text-[12.5px] transition-opacity"
                  style={{ opacity: has ? 1 : 0.28 }}
                >
                  <span
                    className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full text-[9px] font-bold"
                    style={{
                      background: has ? `${c}33` : "transparent",
                      color: has ? c : "#475569",
                      border: has ? `1px solid ${c}` : "1px solid rgba(255,255,255,0.1)",
                    }}
                  >
                    {has ? "✓" : i + 1}
                  </span>
                  <span className={has ? "text-slate-200" : "text-slate-600"}>{s.chip}</span>
                </li>
              );
            })}
          </ol>
          <p className="mt-3 border-t border-white/5 pt-2.5 text-[11px] leading-relaxed text-slate-500">
            {sel === STAGES.length - 1
              ? "That is the whole chain — raw web pages on the left, a guarded token-streaming assistant on the right. You now understand every link."
              : "Advance the stages: each one hands the next a concrete artifact. The chain only works end to end."}
          </p>
        </div>
      </div>
    </div>
  );
}
