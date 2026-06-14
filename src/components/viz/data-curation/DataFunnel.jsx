import { useMemo, useState } from "react";
import { fmt } from "../scales";
import { Button, Toggle } from "../ui";

/**
 * DataFunnel — an interactive vertical "shrinking corpus" funnel.
 *
 * Each pipeline stage is a horizontal bar whose width is proportional to the
 * fraction of the original raw web corpus that survives that stage. Clicking or
 * hovering a stage reveals what that stage removes, the percentage of the
 * ORIGINAL corpus dropped at that step, and an illustrative example of a removed
 * snippet (boilerplate, gibberish, a near-duplicate, etc.).
 *
 * Everything is deterministic — the survival fractions and example snippets are
 * fixed constants, so the figure renders identically on every load. Toggle the
 * inset to reveal a tiny MinHash illustration showing how two near-duplicate
 * strings share many min-hashes (and so are caught by near-dedup).
 */

// Fixed survival fractions (% of the ORIGINAL raw crawl that remains AFTER each
// stage). Chosen to be dramatic but in the right ballpark for real pipelines.
const STAGES = [
  {
    key: "raw",
    label: "Raw web crawl",
    remain: 100,
    color: "#5b7dff",
    removes: "Nothing yet — this is the firehose: raw WARC/HTML straight from Common Crawl.",
    sample: "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>Home</title>…",
    sampleNote: "One page of raw HTML, markup and all.",
  },
  {
    key: "extract",
    label: "Text extracted",
    remain: 60,
    color: "#22d3ee",
    removes:
      "Strips HTML tags, scripts, CSS, nav bars, ads and cookie banners — keeping only the main article text. A huge amount of a typical page is boilerplate.",
    sample: "Accept all cookies · Subscribe to our newsletter · © 2024 · Terms · Privacy · Menu",
    sampleNote: "Removed: navigation / boilerplate that appears on millions of pages.",
  },
  {
    key: "lang",
    label: "Language-filtered (English)",
    remain: 43,
    color: "#34d399",
    removes:
      "Runs a language classifier (e.g. fastText / CLD3) and keeps only documents above a confidence threshold for the target language — here, English.",
    sample: "Willkommen auf unserer Seite. Bitte melden Sie sich an, um fortzufahren.",
    sampleNote: "Removed: a German page (kept for a multilingual model, dropped here).",
  },
  {
    key: "quality",
    label: "Quality-filtered",
    remain: 28,
    color: "#a855f7",
    removes:
      "Drops low-quality text via heuristics (too short, too many symbols, no stopwords, repetitive) and a model-based classifier that scores 'looks like curated prose'.",
    sample: "BUY!!! CHEAP $$$ >>> click >>> here >>> !!! ▓▓▓ Lorem 99 99 99 ###",
    sampleNote: "Removed: spam / gibberish — high symbol ratio, almost no real words.",
  },
  {
    key: "dedup",
    label: "Deduplicated",
    remain: 18,
    color: "#f59e0b",
    removes:
      "Removes exact and near-duplicate documents (mirrors, reposts, SEO copies) using MinHash + LSH, so the model isn't trained on the same text dozens of times.",
    sample: "The quick brown fox jumps over the lazy dog. (…seen 14,000× across the web)",
    sampleNote: "Removed: a near-duplicate — 14k copies collapse to one.",
  },
  {
    key: "final",
    label: "Final training tokens",
    remain: 15,
    color: "#fb7185",
    removes:
      "What's left after PII/toxicity removal and benchmark decontamination: a clean, deduplicated, high-quality corpus. This is what the model actually sees.",
    sample: "In 1687, Newton published the Principia, setting out his three laws of motion…",
    sampleNote: "Kept: clean, encyclopedic prose — the kind of text we want to learn from.",
  },
];

// Geometry for the funnel SVG.
const VB_W = 560;
const ROW_H = 58;
const GAP = 10;
const MAX_BAR = 480; // px for the 100% bar
const LEFT = (VB_W - MAX_BAR) / 2;

function barWidth(remain) {
  return (remain / 100) * MAX_BAR;
}

// ── Tiny MinHash illustration (deterministic) ──────────────────────────────
// Two near-duplicate sentences differ by one word. We shingle into 3-grams of
// words, hash each shingle under k tiny hash functions, and the MinHash signature
// is the per-function minimum. Matching signature slots estimate Jaccard.
const DOC_A = "the cat sat on the mat";
const DOC_B = "the cat sat on a mat";

function wordShingles(text, k = 3) {
  const w = text.split(/\s+/).filter(Boolean);
  const out = [];
  for (let i = 0; i + k <= w.length; i++) out.push(w.slice(i, i + k).join(" "));
  return out;
}

// Tiny deterministic string hash -> 32-bit int, salted per hash-function index.
function hashStr(s, salt) {
  let h = (2166136261 ^ salt) >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h >>> 0;
}

function minhashSignature(shingles, numHashes) {
  const sig = [];
  for (let f = 0; f < numHashes; f++) {
    let m = Infinity;
    for (const sh of shingles) m = Math.min(m, hashStr(sh, (f + 1) * 2654435761));
    sig.push(m >>> 0);
  }
  return sig;
}

function MinHashInset() {
  const NUM = 6;
  const { sigA, sigB, matches, jaccard } = useMemo(() => {
    const shA = wordShingles(DOC_A);
    const shB = wordShingles(DOC_B);
    const sigA = minhashSignature(shA, NUM);
    const sigB = minhashSignature(shB, NUM);
    const matches = sigA.map((x, i) => x === sigB[i]);
    // true Jaccard over the shingle sets for reference.
    const setA = new Set(shA);
    const setB = new Set(shB);
    const inter = [...setA].filter((x) => setB.has(x)).length;
    const uni = new Set([...shA, ...shB]).size;
    return { sigA, sigB, matches, jaccard: inter / uni };
  }, []);

  const est = matches.filter(Boolean).length / NUM;

  return (
    <div className="rounded-xl border border-white/10 bg-ink-900/50 p-4">
      <div className="mb-3 text-xs uppercase tracking-wider text-slate-500">
        Inside dedup · MinHash sketch
      </div>
      <div className="space-y-1.5 font-mono text-[12px] text-slate-300">
        <div>
          A: <span className="text-cyan-200">“{DOC_A}”</span>
        </div>
        <div>
          B: <span className="text-emerald-200">“{DOC_B}”</span>{" "}
          <span className="text-slate-500">(one word changed)</span>
        </div>
      </div>
      <div className="mt-3 text-[11px] text-slate-500">
        3-word shingles → hash under {NUM} functions → keep each function's minimum (the
        signature). Matching slots estimate similarity:
      </div>
      <div className="mt-2 grid grid-cols-6 gap-1.5">
        {sigA.map((v, i) => (
          <div
            key={i}
            className={`rounded-md border px-1 py-1.5 text-center font-mono text-[10px] ${
              matches[i]
                ? "border-emerald-400/50 bg-emerald-400/15 text-emerald-100"
                : "border-rose-400/40 bg-rose-400/10 text-rose-200"
            }`}
            title={`h${i + 1}: minA=${v}  minB=${sigB[i]}`}
          >
            <div className="text-slate-500">h{i + 1}</div>
            <div>{matches[i] ? "match" : "≠"}</div>
          </div>
        ))}
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-[12px]">
        <span className="text-slate-400">
          Estimated similarity:{" "}
          <span className="font-mono text-emerald-200">{fmt(est, 2)}</span>
        </span>
        <span className="text-slate-400">
          True Jaccard:{" "}
          <span className="font-mono text-cyan-200">{fmt(jaccard, 2)}</span>
        </span>
      </div>
      <p className="mt-2 text-[11px] leading-relaxed text-slate-500">
        High overlap ⇒ the pair lands in the same LSH band ⇒ flagged as a near-duplicate. Only one
        copy survives.
      </p>
    </div>
  );
}

export default function DataFunnel() {
  const [active, setActive] = useState("quality");
  const [hover, setHover] = useState(null);
  const [showMinHash, setShowMinHash] = useState(false);

  const focused = hover ?? active;
  const stage = STAGES.find((s) => s.key === focused) || STAGES[0];
  const idx = STAGES.findIndex((s) => s.key === stage.key);
  const dropped = idx === 0 ? 0 : STAGES[idx - 1].remain - stage.remain;

  const svgH = STAGES.length * (ROW_H + GAP);

  return (
    <div className="space-y-4">
      <div className="grid gap-5 lg:grid-cols-[1.35fr_1fr]">
        {/* ── The funnel ─────────────────────────────────────────────── */}
        <div>
          <svg
            viewBox={`0 0 ${VB_W} ${svgH}`}
            className="w-full"
            role="img"
            aria-label="Data cleaning funnel"
          >
            {STAGES.map((s, i) => {
              const w = barWidth(s.remain);
              const x = LEFT + (MAX_BAR - w) / 2; // center each bar
              const y = i * (ROW_H + GAP);
              const isFocus = s.key === focused;
              return (
                <g
                  key={s.key}
                  onMouseEnter={() => setHover(s.key)}
                  onMouseLeave={() => setHover(null)}
                  onClick={() => setActive(s.key)}
                  style={{ cursor: "pointer" }}
                >
                  {/* connector ghost showing what was lost vs the previous bar */}
                  {i > 0 && (
                    <rect
                      x={LEFT + (MAX_BAR - barWidth(STAGES[i - 1].remain)) / 2}
                      y={y}
                      width={barWidth(STAGES[i - 1].remain)}
                      height={ROW_H}
                      fill="#ffffff"
                      opacity={0.03}
                      rx={8}
                    />
                  )}
                  <rect
                    x={x}
                    y={y}
                    width={w}
                    height={ROW_H}
                    rx={8}
                    fill={s.color}
                    opacity={isFocus ? 0.95 : 0.55}
                    stroke={isFocus ? "#fff" : "rgba(255,255,255,0.12)"}
                    strokeWidth={isFocus ? 1.5 : 1}
                  />
                  <text
                    x={VB_W / 2}
                    y={y + ROW_H / 2 - 3}
                    textAnchor="middle"
                    className="fill-white"
                    style={{ fontSize: 13, fontWeight: 600 }}
                  >
                    {s.label}
                  </text>
                  <text
                    x={VB_W / 2}
                    y={y + ROW_H / 2 + 14}
                    textAnchor="middle"
                    className="fill-white/80"
                    style={{ fontSize: 11, fontFamily: "monospace" }}
                  >
                    ≈{s.remain}% remains
                  </text>
                </g>
              );
            })}
          </svg>
          <p className="mt-1 text-center text-[11px] text-slate-500">
            Bar width ∝ fraction of the original crawl surviving. Hover or click a stage.
          </p>
        </div>

        {/* ── Detail panel ───────────────────────────────────────────── */}
        <div className="flex flex-col gap-3">
          <div
            className="rounded-xl border bg-ink-900/50 p-4"
            style={{ borderColor: `${stage.color}66` }}
          >
            <div className="flex items-baseline justify-between gap-2">
              <span className="text-sm font-semibold text-slate-100">{stage.label}</span>
              <span
                className="rounded-md px-2 py-0.5 font-mono text-xs"
                style={{ background: `${stage.color}22`, color: stage.color }}
              >
                {stage.remain}% kept
              </span>
            </div>
            {dropped > 0 && (
              <div className="mt-1 font-mono text-[11px] text-rose-300">
                −{dropped}% of the original dropped at this stage
              </div>
            )}
            <p className="mt-2 text-[13px] leading-relaxed text-slate-300/90">{stage.removes}</p>
          </div>

          <div className="rounded-xl border border-white/10 bg-ink-900/60 p-3">
            <div className="mb-1.5 text-[10px] uppercase tracking-wider text-slate-500">
              {stage.key === "raw" || stage.key === "final"
                ? "Example document"
                : "Example removed snippet"}
            </div>
            <code className="block whitespace-pre-wrap break-words rounded-md bg-black/30 p-2 font-mono text-[11.5px] leading-relaxed text-slate-300">
              {stage.sample}
            </code>
            <div className="mt-1.5 text-[11px] text-slate-500">{stage.sampleNote}</div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <Toggle
              label="Show MinHash sketch"
              checked={showMinHash}
              onChange={setShowMinHash}
            />
            <Button onClick={() => setActive("dedup")} active={active === "dedup"}>
              Jump to dedup
            </Button>
          </div>
        </div>
      </div>

      {showMinHash && <MinHashInset />}
    </div>
  );
}
