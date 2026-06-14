/**
 * Shared interactive controls for visualizations: sliders, toggles,
 * segmented controls, play/step buttons. Kept intentionally minimal and
 * styled to match the site.
 */
import { useEffect, useRef, useState } from "react";

export function Slider({ label, value, min, max, step = 1, onChange, format, accent = "brand" }) {
  const pct = ((value - min) / (max - min)) * 100;
  const accentColor = accent === "brand" ? "#5b7dff" : accent === "cyan" ? "#22d3ee" : "#a855f7";
  return (
    <label className="flex flex-col gap-1.5 text-sm">
      <span className="flex items-center justify-between text-slate-400">
        <span>{label}</span>
        <span className="font-mono text-slate-200">{format ? format(value) : value}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="h-1.5 w-full cursor-pointer appearance-none rounded-full outline-none
          [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:appearance-none
          [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-white [&::-webkit-slider-thumb]:shadow
          [&::-moz-range-thumb]:h-4 [&::-moz-range-thumb]:w-4 [&::-moz-range-thumb]:rounded-full
          [&::-moz-range-thumb]:border-0 [&::-moz-range-thumb]:bg-white"
        style={{
          background: `linear-gradient(to right, ${accentColor} ${pct}%, #27314a ${pct}%)`,
        }}
      />
    </label>
  );
}

export function SegmentedControl({ options, value, onChange, label }) {
  return (
    <div className="flex flex-col gap-1.5 text-sm">
      {label && <span className="text-slate-400">{label}</span>}
      <div className="inline-flex rounded-lg border border-white/10 bg-ink-900/60 p-0.5">
        {options.map((opt) => {
          const v = typeof opt === "string" ? opt : opt.value;
          const l = typeof opt === "string" ? opt : opt.label;
          const active = v === value;
          return (
            <button
              key={v}
              onClick={() => onChange(v)}
              className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
                active ? "bg-brand-500 text-white shadow" : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {l}
            </button>
          );
        })}
      </div>
    </div>
  );
}

export function Toggle({ label, checked, onChange }) {
  // NOTE: the whole control is a single <button>. Do not wrap a <button> in a
  // <label> — a button is a labelable element, so the label re-dispatches the
  // click to it and the handler fires twice (net no-op, toggle looks broken).
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className="flex cursor-pointer items-center gap-2 text-left text-sm text-slate-300"
    >
      <span
        className={`relative inline-block h-5 w-9 shrink-0 rounded-full transition-colors ${
          checked ? "bg-brand-500" : "bg-ink-600"
        }`}
      >
        <span
          className={`absolute top-0.5 h-4 w-4 rounded-full bg-white transition-transform ${
            checked ? "translate-x-4" : "translate-x-0.5"
          }`}
        />
      </span>
      {label && <span>{label}</span>}
    </button>
  );
}

export function Button({ children, onClick, active, disabled }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`rounded-lg border px-3 py-1.5 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
        active
          ? "border-brand-400 bg-brand-500/20 text-brand-100"
          : "border-white/10 bg-ink-800/60 text-slate-300 hover:border-white/20 hover:text-white"
      }`}
    >
      {children}
    </button>
  );
}

/**
 * A play/pause + step control with an internal animation clock.
 * Calls onTick() at the given interval while playing. Returns controls UI.
 */
export function PlaybackBar({ playing, onPlayToggle, onStep, onReset, children, speedLabel }) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <Button onClick={onPlayToggle} active={playing}>
        {playing ? "❚❚ Pause" : "▶ Play"}
      </Button>
      {onStep && <Button onClick={onStep}>Step ▶</Button>}
      {onReset && <Button onClick={onReset}>↻ Reset</Button>}
      {speedLabel && <span className="ml-1 text-xs text-slate-500">{speedLabel}</span>}
      {children}
    </div>
  );
}

/**
 * useInterval — declarative setInterval that respects a `playing` flag.
 */
export function useInterval(callback, delay, playing) {
  const saved = useRef(callback);
  useEffect(() => {
    saved.current = callback;
  }, [callback]);
  useEffect(() => {
    if (!playing || delay == null) return undefined;
    const id = setInterval(() => saved.current(), delay);
    return () => clearInterval(id);
  }, [delay, playing]);
}

/** Small labeled stat pill. */
export function Stat({ label, value, accent = "text-brand-200" }) {
  return (
    <div className="rounded-lg border border-white/10 bg-ink-900/50 px-3 py-2">
      <div className="text-[10px] uppercase tracking-wider text-slate-500">{label}</div>
      <div className={`font-mono text-lg ${accent}`}>{value}</div>
    </div>
  );
}

/** A horizontal legend swatch row. */
export function Legend({ items }) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-400">
      {items.map((it) => (
        <span key={it.label} className="flex items-center gap-1.5">
          <span
            className="inline-block h-3 w-3 rounded-sm"
            style={{ background: it.color }}
          />
          {it.label}
        </span>
      ))}
    </div>
  );
}

/** Hook returning a state + helper for stepping through a bounded index. */
export function useStepper(max, initial = 0) {
  const [i, setI] = useState(initial);
  return {
    i,
    setI,
    next: () => setI((v) => Math.min(max, v + 1)),
    prev: () => setI((v) => Math.max(0, v - 1)),
    reset: () => setI(initial),
    atEnd: i >= max,
    atStart: i <= 0,
  };
}
