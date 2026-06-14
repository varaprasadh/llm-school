import { useMemo, useState } from "react";
import { softmax, fmt, heat } from "./scales";
import { Slider, Button } from "./ui";

/**
 * A toy autoregressive "language model": a hand-authored transition table over
 * a tiny vocabulary. It demonstrates the core loop of generation —
 * predict a distribution → sample → append → repeat — and how *temperature*
 * reshapes that distribution. Logits (not probabilities) are stored so the
 * temperature slider behaves exactly like the real thing.
 */

// context (last token, lowercased) -> [ [word, logit], ... ]
const TABLE = {
  "<start>": [["The", 3.1], ["A", 2.4], ["My", 1.6], ["In", 1.2]],
  the: [["cat", 2.6], ["dog", 2.3], ["robot", 1.9], ["model", 2.1], ["sun", 1.4]],
  a: [["small", 2.2], ["large", 2.0], ["language", 2.7], ["curious", 1.3]],
  my: [["cat", 2.1], ["model", 2.4], ["friend", 1.7]],
  cat: [["sat", 2.8], ["slept", 2.0], ["jumped", 1.9], ["purred", 1.4]],
  dog: [["ran", 2.6], ["barked", 2.3], ["slept", 1.8]],
  robot: [["learned", 2.5], ["computed", 2.2], ["beeped", 1.6]],
  model: [["learned", 2.7], ["predicted", 2.5], ["generated", 2.3], ["trained", 1.8]],
  language: [["model", 3.0], ["is", 1.4]],
  sat: [["on", 2.9], ["quietly", 1.8], ["down", 2.1]],
  slept: [["on", 2.4], ["soundly", 2.0], ["all", 1.6]],
  ran: [["across", 2.3], ["fast", 1.9], ["away", 2.0]],
  learned: [["to", 2.8], ["from", 2.4], ["quickly", 1.7]],
  predicted: [["the", 2.9], ["each", 1.8]],
  generated: [["a", 2.4], ["the", 2.5], ["new", 2.0]],
  on: [["the", 3.0], ["a", 2.1], ["its", 1.6]],
  to: [["predict", 2.6], ["learn", 2.4], ["speak", 2.0], ["read", 1.8]],
  from: [["data", 2.8], ["text", 2.4], ["examples", 2.2]],
  predict: [["the", 2.7], ["each", 1.9], ["text", 1.6]],
  across: [["the", 2.8], ["a", 1.7]],
  small: [["cat", 2.0], ["model", 2.3], ["dog", 1.8]],
  large: [["language", 2.9], ["model", 2.4], ["dataset", 2.0]],
  curious: [["cat", 2.4], ["robot", 2.0], ["child", 1.9]],
  new: [["token", 2.6], ["word", 2.3], ["idea", 1.8]],
  data: [[".", 2.4], ["every", 1.6], ["and", 1.9]],
  text: [[".", 2.3], ["data", 1.8], ["corpus", 2.0]],
};

const DEFAULT_NEXT = [["the", 1.8], ["a", 1.6], ["and", 1.4], [".", 2.0], ["model", 1.3]];

function candidatesFor(lastWord) {
  const key = (lastWord || "<start>").toLowerCase().replace(/[^\w<>]/g, "");
  return TABLE[key] || DEFAULT_NEXT;
}

export default function NextTokenDemo() {
  const [tokens, setTokens] = useState(["The"]);
  const [temp, setTemp] = useState(0.8);
  const [lastPicked, setLastPicked] = useState(null);
  // Deterministic-ish PRNG counter so "sample" varies without Date/Math.random in render.
  const [rngTick, setRngTick] = useState(1);

  const last = tokens[tokens.length - 1];
  const cands = useMemo(() => candidatesFor(last), [last]);
  const logits = cands.map((c) => c[1]);
  const probs = useMemo(() => softmax(logits, temp), [logits, temp]);
  const ranked = cands
    .map((c, i) => ({ word: c[0], logit: c[1], p: probs[i] }))
    .sort((a, b) => b.p - a.p);

  const sample = () => {
    // weighted sample using a simple LCG seeded by current length + tick
    let seed = (tokens.length * 2654435761 + rngTick * 40503) % 2147483647;
    seed = (seed * 16807) % 2147483647;
    const r = (seed % 100000) / 100000;
    let acc = 0;
    let chosen = ranked[0].word;
    for (const item of ranked) {
      acc += item.p;
      if (r <= acc) {
        chosen = item.word;
        break;
      }
    }
    setTokens((t) => [...t, chosen]);
    setLastPicked(chosen);
    setRngTick((x) => x + 1);
  };

  const greedy = () => {
    const top = ranked[0].word;
    setTokens((t) => [...t, top]);
    setLastPicked(top);
  };

  const reset = () => {
    setTokens(["The"]);
    setLastPicked(null);
  };

  return (
    <div className="space-y-5">
      {/* Generated sequence so far */}
      <div className="rounded-xl border border-white/10 bg-ink-900/50 p-4">
        <div className="mb-2 text-xs uppercase tracking-wider text-slate-500">
          Generated sequence
        </div>
        <div className="flex flex-wrap items-center gap-1.5 font-mono text-lg">
          {tokens.map((t, i) => (
            <span
              key={i}
              className={`rounded-md px-2 py-1 ${
                i === tokens.length - 1 && lastPicked
                  ? "bg-brand-500/30 text-brand-100"
                  : "bg-ink-700/60 text-slate-200"
              }`}
            >
              {t}
            </span>
          ))}
          <span className="ml-1 animate-pulse-soft text-brand-400">▌</span>
        </div>
      </div>

      {/* Distribution over next token */}
      <div>
        <div className="mb-2 flex items-center justify-between text-xs uppercase tracking-wider text-slate-500">
          <span>
            P(next token | “…{last}”)
          </span>
          <span>temperature τ = {fmt(temp, 2)}</span>
        </div>
        <div className="space-y-1.5">
          {ranked.map((r) => (
            <div key={r.word} className="flex items-center gap-3">
              <span className="w-20 shrink-0 text-right font-mono text-sm text-slate-300">
                {r.word}
              </span>
              <div className="relative h-6 flex-1 overflow-hidden rounded-md bg-ink-800/70">
                <div
                  className="flex h-full items-center justify-end rounded-md px-2 transition-all duration-300"
                  style={{
                    width: `${Math.max(2, r.p * 100)}%`,
                    background: heat(0.25 + r.p * 0.7),
                  }}
                >
                  <span className="font-mono text-[11px] font-semibold text-ink-950">
                    {(r.p * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-4">
        <div className="min-w-[220px] flex-1">
          <Slider
            label="Temperature"
            min={0.1}
            max={2}
            step={0.05}
            value={temp}
            onChange={setTemp}
            format={(v) => v.toFixed(2)}
          />
        </div>
        <div className="flex gap-2">
          <Button onClick={greedy}>Greedy (argmax)</Button>
          <Button onClick={sample} active>
            🎲 Sample
          </Button>
          <Button onClick={reset}>↻ Reset</Button>
        </div>
      </div>

      <p className="text-xs leading-relaxed text-slate-500">
        Low temperature (→ 0) sharpens the distribution toward the single most likely token
        (greedy, repetitive). High temperature (→ 2) flattens it, increasing diversity and risk.
        Real models do exactly this, just with a 50,000-word vocabulary and billions of parameters
        computing the logits.
      </p>
    </div>
  );
}
