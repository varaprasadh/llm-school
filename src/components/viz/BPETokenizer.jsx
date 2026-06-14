import { useMemo, useState } from "react";
import { SegmentedControl, Slider, Button } from "./ui";

/**
 * Interactive Byte-Pair Encoding.
 *
 *  - "Learn merges": runs the BPE training algorithm on a tiny corpus, one
 *    merge at a time. You watch the most-frequent adjacent pair get fused into
 *    a new symbol, and the vocabulary grow from characters toward words.
 *  - "Tokenize text": applies the learned merges to encode arbitrary input.
 */

const EOW = "·"; // end-of-word marker (rendered faded)

// A small corpus with deliberate repetition so merges are meaningful.
const CORPUS = {
  low: 5,
  lowest: 2,
  newer: 6,
  wider: 3,
  new: 2,
  wide: 4,
  newest: 4,
  older: 3,
};

function trainBPE(wordFreqs, maxMerges) {
  let corpus = Object.entries(wordFreqs).map(([w, f]) => ({
    word: w,
    symbols: [...w, EOW],
    freq: f,
  }));
  const merges = [];
  const steps = [];
  for (let k = 0; k < maxMerges; k++) {
    const counts = {};
    for (const { symbols, freq } of corpus) {
      for (let i = 0; i < symbols.length - 1; i++) {
        const key = symbols[i] + "" + symbols[i + 1];
        counts[key] = (counts[key] || 0) + freq;
      }
    }
    const ranked = Object.entries(counts)
      .map(([key, c]) => ({ pair: key.split(""), count: c }))
      .sort((a, b) => b.count - a.count);
    if (!ranked.length || ranked[0].count < 2) break;
    const { pair, count } = ranked[0];
    const [a, b] = pair;
    const before = corpus.map((w) => ({ ...w, symbols: [...w.symbols] }));
    corpus = corpus.map(({ word, symbols, freq }) => {
      const out = [];
      for (let i = 0; i < symbols.length; i++) {
        if (i < symbols.length - 1 && symbols[i] === a && symbols[i + 1] === b) {
          out.push(a + b);
          i++;
        } else out.push(symbols[i]);
      }
      return { word, symbols: out, freq };
    });
    merges.push([a, b]);
    steps.push({ pair: [a, b], count, top: ranked.slice(0, 5), before, after: corpus });
  }
  return { merges, steps };
}

function encodeWord(word, merges) {
  let symbols = [...word, EOW];
  for (const [a, b] of merges) {
    const out = [];
    for (let i = 0; i < symbols.length; i++) {
      if (i < symbols.length - 1 && symbols[i] === a && symbols[i + 1] === b) {
        out.push(a + b);
        i++;
      } else out.push(symbols[i]);
    }
    symbols = out;
  }
  return symbols;
}

const CHIP_COLORS = [
  "#5b7dff", "#22d3ee", "#34d399", "#a855f7", "#f59e0b", "#fb7185",
];

function Sym({ s, dim, highlight }) {
  const isEow = s.endsWith(EOW);
  return (
    <span
      className={`inline-flex items-center rounded-md border px-1.5 py-0.5 font-mono text-sm transition-all ${
        highlight
          ? "border-amber-400 bg-amber-400/20 text-amber-100"
          : "border-white/10 bg-ink-700/60 text-slate-200"
      }`}
      style={{ opacity: dim ? 0.4 : 1 }}
    >
      {isEow ? (
        <>
          {s.slice(0, -1)}
          <span className="text-slate-600">{EOW}</span>
        </>
      ) : (
        s
      )}
    </span>
  );
}

export default function BPETokenizer() {
  const [mode, setMode] = useState("learn");
  const { merges, steps } = useMemo(() => trainBPE(CORPUS, 14), []);
  const [step, setStep] = useState(3);
  const [text, setText] = useState("the newest wider building");

  const activeMerges = merges.slice(0, step);

  // Vocabulary = base chars + EOW + learned merges (ordered).
  const vocab = useMemo(() => {
    const base = new Set();
    Object.keys(CORPUS).forEach((w) => [...w].forEach((c) => base.add(c)));
    base.add(EOW);
    const v = [...base].sort();
    activeMerges.forEach(([a, b]) => v.push(a + b));
    return v;
  }, [activeMerges]);
  const idOf = (s) => vocab.indexOf(s);

  return (
    <div className="space-y-5">
      <SegmentedControl
        options={[
          { value: "learn", label: "1 · Learn merges" },
          { value: "encode", label: "2 · Tokenize text" },
        ]}
        value={mode}
        onChange={setMode}
      />

      {mode === "learn" ? (
        <>
          <div className="flex flex-wrap items-center gap-4">
            <div className="min-w-[240px] flex-1">
              <Slider
                label="Merges learned"
                min={0}
                max={steps.length}
                value={step}
                onChange={(v) => setStep(Math.round(v))}
                format={(v) => `${Math.round(v)} / ${steps.length}`}
              />
            </div>
            <div className="flex gap-2">
              <Button onClick={() => setStep((s) => Math.max(0, s - 1))}>◀ Back</Button>
              <Button onClick={() => setStep((s) => Math.min(steps.length, s + 1))} active>
                Merge ▶
              </Button>
            </div>
          </div>

          {step > 0 && steps[step - 1] && (
            <div className="rounded-xl border border-amber-400/30 bg-amber-400/[0.06] p-3 text-sm">
              <span className="text-slate-300">Merge #{step}: fuse the most frequent pair </span>
              <span className="font-mono text-amber-200">
                ({steps[step - 1].pair[0].replace(EOW, "·")}, {steps[step - 1].pair[1].replace(EOW, "·")})
              </span>
              <span className="text-slate-300">
                {" "}→ <span className="font-mono text-amber-100">{steps[step - 1].pair.join("").replace(EOW, "·")}</span>{" "}
                (seen {steps[step - 1].count}× across the corpus).
              </span>
            </div>
          )}

          {/* Corpus state */}
          <div className="rounded-xl border border-white/10 bg-ink-900/50 p-4">
            <div className="mb-3 text-xs uppercase tracking-wider text-slate-500">
              Corpus — each word as a sequence of symbols
            </div>
            <div className="grid gap-2 sm:grid-cols-2">
              {(step > 0 ? steps[step - 1].after : initialCorpus()).map((w) => {
                const nextPair = step < steps.length ? steps[step].pair : null;
                return (
                  <div key={w.word} className="flex items-center gap-2">
                    <span className="w-7 shrink-0 text-right font-mono text-[11px] text-slate-600">
                      {w.freq}×
                    </span>
                    <div className="flex flex-wrap items-center gap-1">
                      {w.symbols.map((s, i) => {
                        const willMerge =
                          nextPair &&
                          s === nextPair[0] &&
                          w.symbols[i + 1] === nextPair[1];
                        const willMerge2 =
                          nextPair &&
                          i > 0 &&
                          w.symbols[i - 1] === nextPair[0] &&
                          s === nextPair[1];
                        return <Sym key={i} s={s} highlight={willMerge || willMerge2} />;
                      })}
                    </div>
                  </div>
                );
              })}
            </div>
            {step < steps.length && (
              <div className="mt-3 text-[11px] text-slate-500">
                <span className="rounded border border-amber-400/40 bg-amber-400/10 px-1 text-amber-200">
                  highlighted
                </span>{" "}
                = the pair that will be merged next.
              </div>
            )}
          </div>

          {/* Merge list */}
          {activeMerges.length > 0 && (
            <div className="rounded-xl border border-white/10 bg-ink-900/50 p-4">
              <div className="mb-2 text-xs uppercase tracking-wider text-slate-500">
                Learned merge rules (applied in this order when encoding)
              </div>
              <div className="flex flex-wrap gap-1.5 font-mono text-xs">
                {activeMerges.map(([a, b], i) => (
                  <span key={i} className="rounded bg-ink-700/60 px-2 py-1 text-slate-300">
                    <span className="text-slate-500">{i + 1}.</span> {a.replace(EOW, "·")}+
                    {b.replace(EOW, "·")} → {(a + b).replace(EOW, "·")}
                  </span>
                ))}
              </div>
            </div>
          )}
        </>
      ) : (
        <>
          <label className="block text-sm">
            <span className="mb-1.5 block text-slate-400">Type text to tokenize</span>
            <input
              value={text}
              onChange={(e) => setText(e.target.value)}
              className="w-full rounded-lg border border-white/10 bg-ink-900/70 px-3 py-2 font-mono text-slate-100 focus:border-brand-500/60 focus:outline-none"
            />
          </label>
          <div className="text-xs text-slate-500">
            Using {activeMerges.length} merges learned on the toy corpus (adjust in “Learn merges”).
            Unknown characters fall back to single-character tokens — exactly how real tokenizers
            guarantee any string is encodable.
          </div>

          <div className="rounded-xl border border-white/10 bg-ink-900/50 p-4">
            {text.trim().split(/\s+/).filter(Boolean).map((word, wi) => {
              const toks = encodeWord(word.toLowerCase(), activeMerges);
              return (
                <div key={wi} className="mb-3 flex flex-wrap items-center gap-2">
                  <span className="w-24 shrink-0 font-mono text-sm text-slate-500">“{word}”</span>
                  <span className="text-slate-600">→</span>
                  {toks.map((t, ti) => {
                    const id = idOf(t);
                    return (
                      <span
                        key={ti}
                        className="inline-flex flex-col items-center rounded-md px-2 py-1"
                        style={{ background: `${CHIP_COLORS[(id + 6) % CHIP_COLORS.length]}22` }}
                      >
                        <span className="font-mono text-sm text-slate-100">
                          {t.replace(EOW, EOW)}
                        </span>
                        <span className="font-mono text-[10px] text-slate-500">
                          id {id >= 0 ? id : "?"}
                        </span>
                      </span>
                    );
                  })}
                </div>
              );
            })}
            <div className="mt-2 flex gap-4 border-t border-white/5 pt-3 text-xs text-slate-500">
              <span>
                characters:{" "}
                <span className="font-mono text-slate-300">
                  {text.replace(/\s/g, "").length}
                </span>
              </span>
              <span>
                tokens:{" "}
                <span className="font-mono text-brand-300">
                  {text
                    .trim()
                    .split(/\s+/)
                    .filter(Boolean)
                    .reduce((s, w) => s + encodeWord(w.toLowerCase(), activeMerges).length, 0)}
                </span>
              </span>
              <span>
                vocab size: <span className="font-mono text-slate-300">{vocab.length}</span>
              </span>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function initialCorpus() {
  return Object.entries(CORPUS).map(([w, f]) => ({
    word: w,
    symbols: [...w, EOW],
    freq: f,
  }));
}
