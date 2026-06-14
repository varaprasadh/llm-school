import { useState } from "react";

/**
 * The end-to-end LLM lifecycle as an animated pipeline.
 * Used on the home page and referenced in Ch.1 and the capstone.
 */
const STAGES = [
  { key: "data", label: "Data", icon: "🌐", color: "#34d399", desc: "Collect & clean trillions of tokens of text.", chapter: "data-curation" },
  { key: "token", label: "Tokenize", icon: "🔡", color: "#22d3ee", desc: "Split text into subword tokens → integer IDs.", chapter: "tokenization" },
  { key: "pretrain", label: "Pretrain", icon: "🧠", color: "#5b7dff", desc: "Next-token prediction over the whole corpus.", chapter: "transformer-overview" },
  { key: "sft", label: "Fine-tune", icon: "🎯", color: "#a855f7", desc: "Teach instruction-following on curated examples.", chapter: "supervised-finetuning" },
  { key: "align", label: "Align", icon: "⚖️", color: "#f59e0b", desc: "Shape behavior with human preferences (RLHF/DPO).", chapter: "rlhf" },
  { key: "eval", label: "Evaluate", icon: "📊", color: "#fb7185", desc: "Benchmark capability, safety and regressions.", chapter: "evaluation" },
  { key: "deploy", label: "Deploy", icon: "🚀", color: "#e9faff", desc: "Serve tokens to users, fast and cheap.", chapter: "deployment-serving" },
];

export default function LifecyclePipeline({ onPick }) {
  const [active, setActive] = useState(null);

  return (
    <div className="w-full">
      <div className="flex items-stretch gap-1 overflow-x-auto scrollbar-thin pb-2 sm:gap-2">
        {STAGES.map((s, i) => (
          <div key={s.key} className="flex min-w-0 flex-1 items-center">
            <button
              onMouseEnter={() => setActive(i)}
              onFocus={() => setActive(i)}
              onClick={() => onPick?.(s.chapter)}
              className="group relative flex min-w-[84px] flex-1 flex-col items-center gap-2 rounded-xl border px-2 py-3 transition-all"
              style={{
                borderColor: active === i ? s.color : "rgba(255,255,255,0.08)",
                background:
                  active === i ? `${s.color}1a` : "rgba(20,26,46,0.5)",
              }}
            >
              <span
                className="flex h-10 w-10 items-center justify-center rounded-lg text-xl transition-transform group-hover:scale-110"
                style={{ background: `${s.color}22` }}
              >
                {s.icon}
              </span>
              <span
                className="text-xs font-semibold sm:text-sm"
                style={{ color: active === i ? s.color : "#cbd5e1" }}
              >
                {s.label}
              </span>
            </button>
            {i < STAGES.length - 1 && (
              <svg width="22" height="24" viewBox="0 0 22 24" className="shrink-0">
                <line
                  x1="0"
                  y1="12"
                  x2="22"
                  y2="12"
                  stroke="#3a55f5"
                  strokeWidth="2"
                  strokeDasharray="4 4"
                  className="animate-flow-dash"
                  opacity="0.7"
                />
                <path d="M16 7 L22 12 L16 17" fill="none" stroke="#5b7dff" strokeWidth="2" />
              </svg>
            )}
          </div>
        ))}
      </div>

      <div className="mt-3 min-h-[2.5rem] rounded-lg border border-white/5 bg-ink-900/40 px-4 py-2 text-sm text-slate-400">
        {active === null ? (
          <span className="text-slate-500">
            Hover a stage to see what happens — click to jump to its chapter.
          </span>
        ) : (
          <span>
            <strong className="text-slate-200">{STAGES[active].label}.</strong>{" "}
            {STAGES[active].desc}
          </span>
        )}
      </div>
    </div>
  );
}
