import { useMemo, useState } from "react";
import LineChart from "../LineChart";
import { Slider, Legend } from "../ui";
import { fmt } from "../scales";

/**
 * Preference / Bradley-Terry / DPO visualizer.
 *
 * Left: two candidate responses to one prompt — A is the human-chosen ("good")
 * answer, B is the rejected ("worse") one. A reward model assigns each a scalar
 * score; we expose the *gap* Δ = r_A − r_B with a slider.
 *
 * Middle: the Bradley-Terry model says the probability a human prefers A is
 *   P(A ≻ B) = σ(r_A − r_B) = σ(Δ).
 * We plot that sigmoid over Δ and drop a marker at the current gap.
 *
 * Right: a tiny "what DPO does" panel. DPO's implicit reward is
 *   r(x,y) = β · log( π(y|x) / π_ref(y|x) ).
 * Optimizing the DPO loss pushes the chosen response's log-prob UP and the
 * rejected one's DOWN relative to the reference, widening the implicit-reward
 * margin. We animate that with a "training progress" slider.
 *
 * Fully deterministic — no randomness anywhere.
 */

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

// Reference log-probs (per response) — fixed starting point before DPO.
const REF_LOGP_CHOSEN = -8.0;
const REF_LOGP_REJECTED = -7.2; // ref slightly *prefers* the worse answer — the problem DPO fixes
const BETA = 0.2; // KL / temperature coefficient (typical DPO range 0.1–0.5)

const PROMPT = "Explain why the sky is blue, in one sentence.";
const RESP_A =
  "Sunlight is scattered by air molecules, and shorter blue wavelengths scatter most, so the sky looks blue.";
const RESP_B = "The sky is blue because it is blue. Everyone knows that.";

export default function PreferenceViz() {
  // r_A is anchored; the slider moves the gap by moving r_B.
  const [gap, setGap] = useState(2.0); // Δ = r_A − r_B
  const [progress, setProgress] = useState(0); // DPO training progress in [0,1]

  const rA = 3.0;
  const rB = rA - gap;
  const pPreferA = sigmoid(gap);

  // Bradley-Terry sigmoid curve over a range of gaps, for the chart.
  const curve = useMemo(() => {
    const pts = [];
    for (let d = -6; d <= 6; d += 0.25) pts.push([d, sigmoid(d)]);
    return pts;
  }, []);

  // DPO: as training progresses, chosen log-prob rises, rejected falls.
  // (Magnitudes chosen to be illustrative, not from a specific run.)
  const logpChosen = REF_LOGP_CHOSEN + progress * 4.5; // -8.0 -> -3.5
  const logpRejected = REF_LOGP_REJECTED - progress * 3.5; // -7.2 -> -10.7

  // Implicit DPO rewards r = β·log(π/π_ref) = β·(logp_policy − logp_ref).
  const implicitRchosen = BETA * (logpChosen - REF_LOGP_CHOSEN);
  const implicitRrejected = BETA * (logpRejected - REF_LOGP_REJECTED);
  const margin = implicitRchosen - implicitRrejected; // β·((..)−(..)) inside the DPO loss

  // The DPO loss for this single pair: −log σ(margin).
  const dpoLoss = -Math.log(sigmoid(margin) + 1e-9);

  // Bars for the log-prob panel. Map log-probs (negative) to widths.
  const logpToPct = (lp) => Math.max(2, ((lp + 13) / 13) * 100); // -13..0 -> 0..100%

  return (
    <div className="space-y-6">
      {/* The two candidate responses + reward scores */}
      <div className="rounded-xl border border-white/10 bg-ink-900/50 p-4">
        <div className="mb-3 text-xs uppercase tracking-wider text-slate-500">
          Prompt
        </div>
        <div className="mb-4 rounded-lg bg-ink-850/60 px-3 py-2 text-sm text-slate-300">
          {PROMPT}
        </div>
        <div className="grid gap-3 md:grid-cols-2">
          <Candidate
            tag="A · chosen"
            tagColor="#34d399"
            text={RESP_A}
            score={rA}
            scoreColor="#34d399"
            winner={pPreferA >= 0.5}
          />
          <Candidate
            tag="B · rejected"
            tagColor="#fb7185"
            text={RESP_B}
            score={rB}
            scoreColor="#fb7185"
            winner={pPreferA < 0.5}
          />
        </div>
      </div>

      {/* Bradley-Terry */}
      <div className="rounded-xl border border-white/10 bg-ink-900/50 p-4">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <div className="text-sm text-slate-300">
            Bradley–Terry: a human prefers A with probability{" "}
            <span className="font-mono text-brand-200">σ(r_A − r_B)</span>
          </div>
          <div className="font-mono text-sm text-slate-400">
            σ(<span className="text-emerald-300">{fmt(rA, 1)}</span> −{" "}
            <span className="text-rose-300">{fmt(rB, 1)}</span>) = σ({fmt(gap, 1)}) ={" "}
            <span className="text-brand-200">{(pPreferA * 100).toFixed(0)}%</span>
          </div>
        </div>

        <div className="grid gap-5 lg:grid-cols-[1fr_280px] lg:items-center">
          <LineChart
            series={[
              {
                label: "P(A ≻ B) = σ(Δ)",
                color: "#5b7dff",
                points: curve,
              },
            ]}
            xLabel="reward gap  Δ = r_A − r_B"
            yLabel="P(human prefers A)"
            height={260}
            yTicks={4}
            fmtY={(v) => v.toFixed(1)}
            annotations={[
              { x: gap, label: `Δ=${fmt(gap, 1)}`, color: "#22d3ee" },
            ]}
          />

          <div className="space-y-4">
            <Slider
              label="reward gap  Δ = r_A − r_B"
              value={gap}
              min={-4}
              max={6}
              step={0.1}
              accent="cyan"
              onChange={setGap}
              format={(v) => fmt(v, 1)}
            />
            <div className="rounded-lg border border-white/10 bg-ink-850/50 px-3 py-2.5 text-xs leading-relaxed text-slate-400">
              The reward model is <em>trained</em> so that this probability
              matches the humans: maximize{" "}
              <span className="font-mono text-slate-300">log σ(r_A − r_B)</span>{" "}
              over every labeled pair. A bigger gap ⇒ a more confident, more
              separated reward.
            </div>
            {gap < 0 && (
              <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 px-3 py-2 text-xs text-amber-200">
                Δ &lt; 0: the reward model now ranks the <em>rejected</em> answer
                higher than the chosen one — it disagrees with the human label.
              </div>
            )}
          </div>
        </div>
      </div>

      {/* DPO margin panel */}
      <div className="rounded-xl border border-white/10 bg-ink-900/50 p-4">
        <div className="mb-1 text-sm text-slate-300">
          What DPO optimizes — directly, no reward model, no RL
        </div>
        <p className="mb-4 text-xs leading-relaxed text-slate-500">
          DPO’s <em>implicit</em> reward is{" "}
          <span className="font-mono text-slate-300">
            r = β·log(π/π_ref)
          </span>
          . Minimizing the DPO loss raises the chosen response’s log-prob and
          lowers the rejected one’s, widening the margin{" "}
          <span className="font-mono text-slate-300">
            β·((log π_chosen − log π_ref,chosen) − (log π_rejected − log π_ref,rejected))
          </span>
          .
        </p>

        <Slider
          label="DPO training progress"
          value={progress}
          min={0}
          max={1}
          step={0.01}
          accent="violet"
          onChange={setProgress}
          format={(v) => `${(v * 100).toFixed(0)}%`}
        />

        <div className="mt-4 grid gap-2.5">
          <LogpBar
            label="log π(chosen)"
            value={logpChosen}
            ref0={REF_LOGP_CHOSEN}
            color="#34d399"
            pct={logpToPct(logpChosen)}
            refPct={logpToPct(REF_LOGP_CHOSEN)}
          />
          <LogpBar
            label="log π(rejected)"
            value={logpRejected}
            ref0={REF_LOGP_REJECTED}
            color="#fb7185"
            pct={logpToPct(logpRejected)}
            refPct={logpToPct(REF_LOGP_REJECTED)}
          />
        </div>

        <Legend
          items={[
            { label: "current policy log-prob", color: "#64748b" },
            { label: "reference (π_ref) start", color: "#a855f7" },
          ]}
        />

        <div className="mt-4 grid gap-3 sm:grid-cols-3">
          <Stat label="implicit reward margin" value={fmt(margin, 2)} accent="text-brand-200" />
          <Stat
            label="P(chosen ≻ rejected)"
            value={`${(sigmoid(margin) * 100).toFixed(0)}%`}
            accent="text-emerald-300"
          />
          <Stat label="DPO loss  −log σ(margin)" value={fmt(dpoLoss, 3)} accent="text-rose-300" />
        </div>
        <p className="mt-3 text-xs text-slate-500">
          As you drag training forward, the margin grows, the model’s preference
          probability climbs toward 100%, and the loss falls toward 0 — all from a
          plain binary-classification objective on preference pairs.
        </p>
      </div>
    </div>
  );
}

function Candidate({ tag, tagColor, text, score, scoreColor, winner }) {
  return (
    <div
      className="rounded-lg border bg-ink-850/40 p-3"
      style={{
        borderColor: winner ? `${scoreColor}88` : "rgba(255,255,255,0.08)",
      }}
    >
      <div className="mb-2 flex items-center justify-between">
        <span
          className="rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider"
          style={{ background: `${tagColor}22`, color: tagColor }}
        >
          {tag}
        </span>
        <span className="font-mono text-sm" style={{ color: scoreColor }}>
          r = {fmt(score, 1)}
        </span>
      </div>
      <p className="text-sm leading-relaxed text-slate-300">{text}</p>
    </div>
  );
}

function LogpBar({ label, value, ref0, color, pct, refPct }) {
  return (
    <div className="flex items-center gap-3 text-sm">
      <span className="w-32 shrink-0 text-right font-mono text-xs text-slate-400">
        {label}
      </span>
      <div className="relative h-6 flex-1 overflow-hidden rounded bg-ink-800/70">
        <div
          className="h-full rounded transition-all duration-200"
          style={{ width: `${pct}%`, background: `${color}55` }}
        />
        {/* reference marker */}
        <div
          className="absolute top-0 h-full w-0.5"
          style={{ left: `${refPct}%`, background: "#a855f7" }}
        />
      </div>
      <span className="w-16 shrink-0 text-right font-mono text-xs" style={{ color }}>
        {fmt(value, 2)}
      </span>
    </div>
  );
}

function Stat({ label, value, accent }) {
  return (
    <div className="rounded-lg border border-white/10 bg-ink-900/50 px-3 py-2">
      <div className="text-[10px] uppercase tracking-wider text-slate-500">{label}</div>
      <div className={`font-mono text-lg ${accent}`}>{value}</div>
    </div>
  );
}
