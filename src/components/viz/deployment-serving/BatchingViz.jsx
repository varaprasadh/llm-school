import { useMemo, useState } from "react";
import { SegmentedControl, Button, Stat } from "../ui";

/**
 * BatchingViz — a time-grid of GPU "slots" (rows) over decoding steps (columns)
 * that contrasts STATIC batching with CONTINUOUS (in-flight) batching.
 *
 * Four requests arrive, each needing a different number of decode steps. The
 * GPU has SLOTS rows; each step it does one forward pass over whatever occupies
 * the slots.
 *
 *  • STATIC: a batch of requests is locked in together. A slot whose request has
 *    FINISHED sits IDLE (greyed) — the GPU still pays for an empty seat — until
 *    the WHOLE batch drains (the slowest request finishes). Only then can the
 *    next batch start. Lots of wasted cells.
 *
 *  • CONTINUOUS: the moment a request finishes and frees its slot, a waiting
 *    request is admitted into that slot on the very next step. Slots stay busy,
 *    so utilization is much higher and waiting requests start sooner.
 *
 * Utilization = busy cells / (slots × steps elapsed). Fully deterministic: the
 * schedule is computed from a fixed request list, no randomness or clocks.
 */

const SLOTS = 3; // GPU can run this many sequences in parallel

// Fixed request set: id, the step it ARRIVES, and how many decode steps it needs.
// Lengths differ a lot on purpose — that variance is exactly what static wastes.
const REQUESTS = [
  { id: "R1", arrive: 0, len: 3, color: "#5b7dff" },
  { id: "R2", arrive: 0, len: 7, color: "#22d3ee" },
  { id: "R3", arrive: 0, len: 2, color: "#a855f7" },
  { id: "R4", arrive: 0, len: 4, color: "#34d399" },
  { id: "R5", arrive: 1, len: 5, color: "#f59e0b" },
  { id: "R6", arrive: 2, len: 3, color: "#fb7185" },
];

/**
 * Build a schedule: schedule[step] = array of length SLOTS, each entry is either
 *   null (idle) | { reqId, kind: "run" } where kind distinguishes a productive
 *   token from an idle-but-locked seat. We return per-mode timelines plus the
 *   total number of steps and a running utilization series.
 *
 * Cell kinds:
 *   "run"   — slot is producing a token this step (busy / useful)
 *   "idle"  — slot is occupied by a finished request in STATIC (locked, wasted)
 *   "empty" — slot is genuinely empty (no request assigned)
 */
function buildSchedule(mode) {
  const grid = []; // grid[step][slot]
  // remaining decode steps for the request currently in each slot (or null)
  const slotReq = Array(SLOTS).fill(null); // { id, remaining, color }
  const queue = [...REQUESTS].sort((a, b) => a.arrive - b.arrive);
  let qi = 0; // next request not yet admitted
  let step = 0;
  // In static mode we admit a *batch* and refuse new admissions until the batch
  // fully drains. Track whether a batch is currently in flight.
  let staticBatchActive = false;

  const admit = (slot) => {
    if (qi < queue.length && queue[qi].arrive <= step) {
      const r = queue[qi++];
      slotReq[slot] = { id: r.id, remaining: r.len, color: r.color };
      return true;
    }
    return false;
  };

  // Safety bound so the loop always terminates.
  for (; step < 64; step++) {
    // ── admission policy ──────────────────────────────────────────────────
    if (mode === "static") {
      const allEmpty = slotReq.every((s) => s === null);
      if (allEmpty) {
        // start a fresh batch: fill as many slots as we can from the queue
        for (let s = 0; s < SLOTS; s++) admit(s);
        staticBatchActive = slotReq.some((s) => s !== null);
      }
      // while a batch is active we do NOT admit into freed slots (the wasteful bit)
    } else {
      // continuous: greedily fill ANY empty slot every step
      for (let s = 0; s < SLOTS; s++) {
        if (slotReq[s] === null) admit(s);
      }
    }

    // Stop once nothing is running and nothing is left to admit.
    const anythingLeft = slotReq.some((s) => s !== null) || qi < queue.length;
    if (!anythingLeft) break;

    // ── this step's row ───────────────────────────────────────────────────
    const row = slotReq.map((s) => {
      if (s === null) return { kind: "empty" };
      if (s.remaining > 0) return { kind: "run", id: s.id, color: s.color };
      return { kind: "idle", id: s.id, color: s.color }; // locked, finished, static-only
    });
    grid.push(row);

    // ── advance: consume one token from each running slot ─────────────────
    for (let s = 0; s < SLOTS; s++) {
      if (slotReq[s] && slotReq[s].remaining > 0) {
        slotReq[s].remaining -= 1;
      }
    }

    if (mode === "static") {
      // a finished request keeps its (idle) seat until the WHOLE batch is done
      const allDone = slotReq.every((s) => s === null || s.remaining === 0);
      if (allDone && staticBatchActive) {
        for (let s = 0; s < SLOTS; s++) slotReq[s] = null; // drain the batch
        staticBatchActive = false;
      }
    } else {
      // continuous: free finished slots immediately so they can be refilled
      for (let s = 0; s < SLOTS; s++) {
        if (slotReq[s] && slotReq[s].remaining === 0) slotReq[s] = null;
      }
    }
  }

  // ── utilization, cumulative over elapsed steps ────────────────────────────
  let busy = 0;
  const util = grid.map((row, i) => {
    busy += row.filter((c) => c.kind === "run").length;
    return busy / ((i + 1) * SLOTS);
  });
  const finalUtil = util.length ? util[util.length - 1] : 0;

  return { grid, util, finalUtil, steps: grid.length };
}

const CELL = 40;
const GAP = 5;
const LEFT = 64; // row-label gutter
const TOP = 26; // column-label gutter

export default function BatchingViz() {
  const [mode, setMode] = useState("static");
  const [step, setStep] = useState(99); // show full timeline by default; clamp below

  const sched = useMemo(() => buildSchedule(mode), [mode]);
  const maxStep = sched.steps;
  const shown = Math.min(step, maxStep);

  // utilization up to the currently revealed step
  const utilNow = shown === 0 ? 0 : sched.util[shown - 1];

  // Compare both modes' final utilization for the headline stat.
  const both = useMemo(
    () => ({
      static: buildSchedule("static").finalUtil,
      continuous: buildSchedule("continuous").finalUtil,
    }),
    []
  );

  const W = LEFT + maxStep * (CELL + GAP) + 8;
  const H = TOP + SLOTS * (CELL + GAP) + 8;

  return (
    <div className="space-y-5">
      {/* controls */}
      <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
        <SegmentedControl
          label="batching strategy"
          value={mode}
          onChange={(m) => {
            setMode(m);
          }}
          options={[
            { value: "static", label: "Static" },
            { value: "continuous", label: "Continuous" },
          ]}
        />
        <div className="flex flex-wrap items-center gap-2">
          <Button onClick={() => setStep((s) => Math.min(maxStep, Math.min(s, maxStep) + 1))} disabled={shown >= maxStep}>
            Step ▶
          </Button>
          <Button onClick={() => setStep(0)}>↻ To start</Button>
          <Button onClick={() => setStep(maxStep)}>Run to end ⏭</Button>
        </div>
        <span className="text-xs text-slate-400">
          step {shown}/{maxStep}
        </span>
      </div>

      {/* the time-grid */}
      <div className="rounded-xl border border-white/10 bg-ink-900/50 p-4">
        <div className="overflow-x-auto scrollbar-thin">
          <svg width={W} height={H} className="block">
            {/* column headers = decode steps */}
            {Array.from({ length: maxStep }, (_, t) => (
              <text
                key={t}
                x={LEFT + t * (CELL + GAP) + CELL / 2}
                y={TOP - 10}
                textAnchor="middle"
                fontSize="10"
                fill={t < shown ? "#94a3b8" : "#475569"}
                fontFamily="JetBrains Mono, monospace"
              >
                t{t + 1}
              </text>
            ))}

            {/* row labels = GPU slots */}
            {Array.from({ length: SLOTS }, (_, s) => (
              <text
                key={s}
                x={LEFT - 10}
                y={TOP + s * (CELL + GAP) + CELL / 2 + 4}
                textAnchor="end"
                fontSize="11"
                fill="#94a3b8"
                fontFamily="JetBrains Mono, monospace"
              >
                slot {s + 1}
              </text>
            ))}

            {/* cells */}
            {sched.grid.map((row, t) =>
              row.map((cell, s) => {
                const x = LEFT + t * (CELL + GAP);
                const y = TOP + s * (CELL + GAP);
                const revealed = t < shown;
                let fill = "#10182a";
                let stroke = "rgba(255,255,255,0.06)";
                let label = "";
                let labelColor = "#475569";
                if (cell.kind === "run") {
                  fill = cell.color;
                  stroke = "rgba(255,255,255,0.18)";
                  label = cell.id;
                  labelColor = "#0b0e1a";
                } else if (cell.kind === "idle") {
                  fill = "#1c2438";
                  stroke = "rgba(251,113,133,0.35)";
                  label = "idle";
                  labelColor = "#fb7185";
                }
                return (
                  <g key={`${t}-${s}`} opacity={revealed ? 1 : 0.18} className="transition-opacity duration-200">
                    <rect
                      x={x}
                      y={y}
                      width={CELL}
                      height={CELL}
                      rx={6}
                      fill={fill}
                      stroke={stroke}
                      strokeWidth={1}
                      strokeDasharray={cell.kind === "idle" ? "3 2" : undefined}
                    />
                    {label && revealed && (
                      <text
                        x={x + CELL / 2}
                        y={y + CELL / 2 + 4}
                        textAnchor="middle"
                        fontSize={cell.kind === "idle" ? "8.5" : "11"}
                        fontWeight={cell.kind === "run" ? 700 : 400}
                        fill={labelColor}
                        fontFamily="JetBrains Mono, monospace"
                      >
                        {label}
                      </text>
                    )}
                  </g>
                );
              })
            )}
          </svg>
        </div>

        {/* legend */}
        <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1.5 text-[11px] text-slate-400">
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-3 w-3 rounded-sm" style={{ background: "#5b7dff" }} />
            slot producing a token (busy)
          </span>
          <span className="flex items-center gap-1.5">
            <span
              className="inline-block h-3 w-3 rounded-sm border border-dashed border-rose-400/50"
              style={{ background: "#1c2438" }}
            />
            finished but locked (wasted — static only)
          </span>
          <span className="flex items-center gap-1.5">
            <span
              className="inline-block h-3 w-3 rounded-sm"
              style={{ background: "#10182a", border: "1px solid rgba(255,255,255,0.06)" }}
            />
            empty
          </span>
        </div>
      </div>

      {/* utilization readouts */}
      <div className="flex flex-wrap items-center gap-3">
        <Stat
          label={`${mode} util (so far)`}
          value={`${Math.round(utilNow * 100)}%`}
          accent={mode === "continuous" ? "text-emerald-200" : "text-rose-200"}
        />
        <span className="font-mono text-slate-600">·</span>
        <Stat label="static (final)" value={`${Math.round(both.static * 100)}%`} accent="text-rose-200" />
        <Stat
          label="continuous (final)"
          value={`${Math.round(both.continuous * 100)}%`}
          accent="text-emerald-200"
        />
        <Stat
          label="speedup"
          value={`${(both.continuous / Math.max(both.static, 1e-6)).toFixed(2)}×`}
          accent="text-amber-200"
        />
      </div>

      <p className="text-xs leading-relaxed text-slate-500">
        {mode === "static" ? (
          <>
            In <span className="text-rose-200">static</span> batching, short requests (R3, R1) finish
            early but their slots sit <span className="text-rose-200">idle</span> — the GPU keeps
            paying for empty seats until the slowest request in the batch (R2) drains and the next
            batch can begin. Switch to continuous and watch those gaps disappear.
          </>
        ) : (
          <>
            In <span className="text-emerald-200">continuous</span> batching, the instant a request
            finishes its slot is refilled from the waiting queue on the very next step. Slots stay
            busy, waiting requests start sooner, and GPU utilization climbs toward 100%.
          </>
        )}
      </p>
    </div>
  );
}
