import { useMemo, useState } from "react";
import { Slider, Stat, Button } from "../ui";
import { heat, fmt } from "../scales";

/**
 * Interactive gradient-accumulation visualizer.
 *
 * The user picks a micro-batch size, the number of accumulation steps, and a
 * GPU count. We then step through a single "optimizer cycle": micro-batches
 * stream in one at a time, each contributes a gradient that is *summed* into a
 * running buffer (we never zero it between micro-steps), and only after the
 * Nth micro-batch does the optimizer actually fire and the buffer reset.
 *
 * Everything is deterministic — the per-micro-batch gradient "magnitudes" are a
 * fixed, hand-authored sequence so the picture is stable across renders. The
 * point is the bookkeeping (when does .step() happen, what is the effective
 * batch), not real numbers.
 */

// A fixed, plausible-looking sequence of per-micro-batch gradient magnitudes
// (||g||) so the accumulating bars look organic but never change between renders.
const GRAD_SEQ = [0.82, 0.61, 0.74, 0.55, 0.9, 0.48, 0.67, 0.71, 0.58, 0.8, 0.63, 0.52];

export default function GradAccum() {
  const [microBsz, setMicroBsz] = useState(8);
  const [accum, setAccum] = useState(4);
  const [gpus, setGpus] = useState(1);
  // how many micro-batches of the *current* cycle have been processed (1..accum).
  // step === 0 means "cycle not started"; step === accum means "ready to .step()".
  const [step, setStep] = useState(0);
  const [didStep, setDidStep] = useState(false);

  const effective = microBsz * accum * gpus;

  // Reset the walkthrough whenever the structure changes.
  const reset = () => {
    setStep(0);
    setDidStep(false);
  };

  const next = () => {
    if (didStep) {
      // start a fresh cycle
      setStep(1);
      setDidStep(false);
      return;
    }
    if (step < accum) {
      setStep(step + 1);
    } else {
      // we are at the boundary -> fire the optimizer
      setDidStep(true);
    }
  };

  // The slice of GRAD_SEQ used for the current cycle.
  const grads = useMemo(
    () => Array.from({ length: accum }, (_, i) => GRAD_SEQ[i % GRAD_SEQ.length]),
    [accum]
  );
  const accumulated = grads.slice(0, step).reduce((a, b) => a + b, 0);
  const totalIfFull = grads.reduce((a, b) => a + b, 0);
  // The averaged gradient the optimizer actually consumes (we normalize by accum,
  // which is the standard loss = loss / accum trick).
  const avgGrad = totalIfFull / accum;

  const atBoundary = step >= accum;
  const phase = didStep
    ? "optimizer.step() fired — buffer zeroed"
    : step === 0
    ? "press Next to stream the first micro-batch"
    : atBoundary
    ? "buffer full — next press fires optimizer.step()"
    : `accumulating: micro-batch ${step} of ${accum} (no step yet)`;

  // ---- geometry for the micro-batch flow ----
  const W = 660;
  const slotW = Math.min(120, (W - 40) / accum);
  const slotGap = 10;

  return (
    <div className="space-y-6">
      {/* controls */}
      <div className="grid gap-x-6 gap-y-4 sm:grid-cols-3">
        <Slider
          label="micro_bsz (per GPU)"
          min={1}
          max={32}
          step={1}
          value={microBsz}
          onChange={(v) => {
            setMicroBsz(v);
          }}
        />
        <Slider
          label="accumulation steps"
          min={1}
          max={8}
          step={1}
          value={accum}
          onChange={(v) => {
            setAccum(v);
            reset();
          }}
          accent="cyan"
        />
        <Slider
          label="# GPUs (data-parallel)"
          min={1}
          max={8}
          step={1}
          value={gpus}
          onChange={setGpus}
          accent="violet"
        />
      </div>

      {/* effective batch readout */}
      <div className="flex flex-wrap items-center gap-3">
        <Stat label="micro_bsz" value={String(microBsz)} accent="text-brand-200" />
        <span className="font-mono text-slate-600">×</span>
        <Stat label="accum" value={String(accum)} accent="text-cyan-200" />
        <span className="font-mono text-slate-600">×</span>
        <Stat label="# GPUs" value={String(gpus)} accent="text-violet-200" />
        <span className="font-mono text-slate-600">=</span>
        <Stat label="effective batch" value={String(effective)} accent="text-amber-200" />
      </div>

      {/* the micro-batch flow */}
      <div className="rounded-xl border border-white/10 bg-ink-900/50 p-4">
        <div className="mb-3 flex items-center justify-between">
          <span className="text-sm text-slate-300">
            One optimizer cycle = <span className="font-mono text-cyan-200">{accum}</span> forward/
            backward passes, then <span className="font-mono text-amber-200">one</span> step.
          </span>
        </div>

        <div className="overflow-x-auto scrollbar-thin">
          <svg width={W} height={170} className="block min-w-[560px]">
            {/* the running gradient buffer (top bar) */}
            <text x={4} y={14} fontSize="11" fill="#94a3b8" fontFamily="JetBrains Mono, monospace">
              grad buffer Σ‖g‖
            </text>
            <rect x={130} y={4} width={W - 150} height={16} rx={4} fill="#1c2438" />
            <rect
              x={130}
              y={4}
              width={
                didStep ? 0 : ((W - 150) * accumulated) / Math.max(totalIfFull, 1e-6)
              }
              height={16}
              rx={4}
              fill={didStep ? "#34d399" : heat(0.4 + 0.5 * (accumulated / Math.max(totalIfFull, 1e-6)))}
              className="transition-all duration-300"
            />
            <text
              x={W - 18}
              y={16}
              textAnchor="end"
              fontSize="10"
              fill="#cbd5e1"
              fontFamily="JetBrains Mono, monospace"
            >
              {didStep ? "0.00 (zeroed)" : fmt(accumulated, 2)}
            </text>

            {/* micro-batch slots */}
            {grads.map((g, i) => {
              const x = 130 + i * (slotW + slotGap);
              const processed = !didStep && i < step;
              const isCurrent = !didStep && i === step - 1;
              const barH = 56 * g;
              return (
                <g key={i}>
                  {/* the data shard box */}
                  <rect
                    x={x}
                    y={48}
                    width={slotW}
                    height={40}
                    rx={6}
                    fill={processed ? "#16233b" : "#10182a"}
                    stroke={isCurrent ? "#8eabff" : processed ? "#2a4a6a" : "#27314a"}
                    strokeWidth={isCurrent ? 2 : 1}
                  />
                  <text
                    x={x + slotW / 2}
                    y={66}
                    textAnchor="middle"
                    fontSize="10.5"
                    fill={processed ? "#cbd5e1" : "#64748b"}
                    fontFamily="JetBrains Mono, monospace"
                  >
                    micro {i + 1}
                  </text>
                  <text
                    x={x + slotW / 2}
                    y={80}
                    textAnchor="middle"
                    fontSize="9"
                    fill="#475569"
                    fontFamily="JetBrains Mono, monospace"
                  >
                    {microBsz} samples
                  </text>

                  {/* its gradient contribution bar, growing up toward the buffer */}
                  <rect
                    x={x + slotW / 2 - 10}
                    y={48 - barH}
                    width={20}
                    height={processed ? barH : 0}
                    rx={3}
                    fill={heat(0.35 + 0.6 * g)}
                    opacity={processed ? 0.9 : 0}
                    className="transition-all duration-300"
                  />
                  {processed && (
                    <text
                      x={x + slotW / 2}
                      y={48 - barH - 4}
                      textAnchor="middle"
                      fontSize="9"
                      fill="#94a3b8"
                      fontFamily="JetBrains Mono, monospace"
                    >
                      ‖g‖={fmt(g, 2)}
                    </text>
                  )}

                  {/* "+" connectors between contributions */}
                  {i > 0 && (
                    <text
                      x={x - slotGap / 2}
                      y={70}
                      textAnchor="middle"
                      fontSize="13"
                      fill="#475569"
                      fontFamily="JetBrains Mono, monospace"
                    >
                      +
                    </text>
                  )}
                </g>
              );
            })}

            {/* the optimizer.step() badge */}
            <g transform={`translate(${130}, 112)`}>
              <rect
                x={0}
                y={0}
                width={W - 150}
                height={40}
                rx={8}
                fill={didStep ? "#1f3b2f" : atBoundary ? "#3a2a12" : "#10182a"}
                stroke={didStep ? "#34d399" : atBoundary ? "#f59e0b" : "#27314a"}
                strokeWidth={didStep || atBoundary ? 2 : 1}
                className="transition-all duration-300"
              />
              <text
                x={(W - 150) / 2}
                y={18}
                textAnchor="middle"
                fontSize="12"
                fontWeight="600"
                fill={didStep ? "#86efac" : atBoundary ? "#fcd34d" : "#64748b"}
                fontFamily="JetBrains Mono, monospace"
              >
                {didStep
                  ? "✓ optimizer.step()  +  optimizer.zero_grad()"
                  : "optimizer.step()  (waits for the buffer to fill)"}
              </text>
              <text
                x={(W - 150) / 2}
                y={33}
                textAnchor="middle"
                fontSize="9.5"
                fill="#64748b"
                fontFamily="JetBrains Mono, monospace"
              >
                consumes averaged grad ≈ {fmt(avgGrad, 2)} (Σ‖g‖ / accum)
              </text>
            </g>
          </svg>
        </div>

        <div className="mt-3 flex flex-wrap items-center gap-3">
          <Button onClick={next}>Next ▶</Button>
          <Button onClick={reset}>↻ Reset</Button>
          <span className="text-xs text-slate-400">{phase}</span>
        </div>
      </div>

      <p className="text-xs leading-relaxed text-slate-500">
        Notice what <em>doesn’t</em> happen: the buffer is never cleared between micro-batches —
        gradients <em>sum</em>. The weights only move once per cycle, on the
        <span className="font-mono text-amber-200"> step</span>. To the optimizer, that single update
        is indistinguishable from one taken on a batch of{" "}
        <span className="font-mono text-amber-200">{effective}</span> samples — but you only ever held{" "}
        <span className="font-mono text-brand-200">{microBsz}</span> samples in memory at a time.
      </p>
    </div>
  );
}
