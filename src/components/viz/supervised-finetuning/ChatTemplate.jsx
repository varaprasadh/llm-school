import { useMemo, useState } from "react";
import { Toggle, SegmentedControl, Legend } from "../ui";

/**
 * Chat-template + loss-masking visualizer.
 *
 * A fixed conversation (system + user + assistant) is rendered as the exact
 * token stream an SFT trainer sees, wrapped in ChatML-style special tokens
 * (<|im_start|>, <|im_end|>). Each token carries a `role` and a flag for
 * whether the loss is computed on it.
 *
 * The teaching point: during SFT we run the model over the WHOLE sequence, but
 * we only backpropagate loss on the *assistant's response* tokens. Every prompt
 * token (system, user, role markers, the structural tokens) is masked — its
 * label is set to -100 (PyTorch's ignore_index), so cross-entropy skips it.
 *
 * Everything here is deterministic: token ids are a small stable hash so the
 * numbers don't jump around between renders.
 */

// ChatML-style special tokens get fixed, low ids (as in a real added-tokens map).
const SPECIAL_IDS = {
  "<|im_start|>": 32001,
  "<|im_end|>": 32002,
  "\\n": 198,
};

/** Tiny stable hash -> a plausible-looking vocab id in [1000, 31000). */
function tokId(s) {
  if (s in SPECIAL_IDS) return SPECIAL_IDS[s];
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return 1000 + ((h >>> 0) % 30000);
}

// One conversation, pre-split into the "words" that become tokens. We keep the
// tokenization word-level for legibility; real BPE would split some of these
// further, but the masking logic is identical.
const TURNS = [
  {
    role: "system",
    text: ["You", "are", "a", "concise", "assistant", "."],
  },
  {
    role: "user",
    text: ["What", "is", "the", "capital", "of", "France", "?"],
  },
  {
    role: "assistant",
    text: ["The", "capital", "of", "France", "is", "Paris", "."],
  },
];

const ROLE_LABEL = { system: "system", user: "user", assistant: "assistant" };

// Colors per role for the chip backgrounds (used only when NOT showing the mask).
const ROLE_COLOR = {
  system: "#a855f7",
  user: "#22d3ee",
  assistant: "#34d399",
};

/**
 * Flatten the conversation into a flat list of tokens with full metadata,
 * inserting the ChatML structural tokens exactly where a real template would.
 *
 *   <|im_start|>system\nYou are ...<|im_end|>\n
 *   <|im_start|>user\n...<|im_end|>\n
 *   <|im_start|>assistant\nThe capital ...<|im_end|>\n
 *
 * Loss (trained === true) is on the assistant CONTENT plus its closing
 * <|im_end|> — and nothing else. The role tag, the opening marker and every
 * prompt token are masked.
 */
function buildTokens() {
  const out = [];
  const push = (text, role, kind, trained) =>
    out.push({ text, role, kind, trained, id: tokId(text) });

  for (const turn of TURNS) {
    const isAsst = turn.role === "assistant";
    // <|im_start|>  — structural, always masked (it's part of the prompt format)
    push("<|im_start|>", turn.role, "special", false);
    // role tag (e.g. "assistant") — masked: the model isn't asked to predict it
    push(ROLE_LABEL[turn.role], turn.role, "role", false);
    push("\\n", turn.role, "special", false);
    // content tokens — trained ONLY for the assistant turn
    for (const w of turn.text) push(w, turn.role, "content", isAsst);
    // <|im_end|>  — trained for assistant (model must learn to stop), else masked
    push("<|im_end|>", turn.role, "special", isAsst);
    push("\\n", turn.role, "special", false);
  }
  return out;
}

export default function ChatTemplate() {
  const [showMask, setShowMask] = useState(true);
  const [view, setView] = useState("chips"); // "chips" | "labels"
  const [hover, setHover] = useState(null);

  const tokens = useMemo(buildTokens, []);

  const trainedCount = tokens.filter((t) => t.trained).length;
  const maskedCount = tokens.length - trainedCount;

  return (
    <div className="space-y-5">
      {/* Controls */}
      <div className="flex flex-wrap items-end gap-5">
        <Toggle
          label="Highlight loss mask"
          checked={showMask}
          onChange={setShowMask}
        />
        <SegmentedControl
          label="Token annotation"
          options={[
            { value: "chips", label: "Text" },
            { value: "labels", label: "Labels (target ids)" },
          ]}
          value={view}
          onChange={setView}
        />
        <div className="text-xs text-slate-500">
          Hover any token to inspect its training target.
        </div>
      </div>

      {/* Legend */}
      {showMask ? (
        <Legend
          items={[
            { label: "trained — loss computed (label = token id)", color: "#34d399" },
            { label: "masked — no loss (label = −100)", color: "#27314a" },
          ]}
        />
      ) : (
        <Legend
          items={[
            { label: "system", color: ROLE_COLOR.system },
            { label: "user", color: ROLE_COLOR.user },
            { label: "assistant", color: ROLE_COLOR.assistant },
          ]}
        />
      )}

      {/* Token stream */}
      <div className="rounded-xl border border-white/10 bg-ink-900/50 p-4">
        <div className="flex flex-wrap gap-1.5 font-mono text-[13px] leading-7">
          {tokens.map((t, i) => {
            const masked = !t.trained;
            // choose background + text color
            let bg, fg, border;
            if (showMask) {
              if (masked) {
                bg = "#171d2e";
                fg = "#5b6577";
                border = "1px solid #232c42";
              } else {
                bg = "rgba(52,211,153,0.16)";
                fg = "#9af2c9";
                border = "1px solid rgba(52,211,153,0.55)";
              }
            } else {
              const c = ROLE_COLOR[t.role];
              bg = `${c}22`;
              fg = "#dbe4f3";
              border = `1px solid ${c}66`;
            }
            const isHover = hover === i;
            const display =
              view === "labels"
                ? masked
                  ? "−100"
                  : String(t.id)
                : t.text === "\\n"
                ? "\\n"
                : t.text;
            return (
              <span
                key={i}
                onMouseEnter={() => setHover(i)}
                onMouseLeave={() => setHover(null)}
                className="relative cursor-default select-none rounded px-1.5 py-0.5 transition-colors"
                style={{
                  background: bg,
                  color: fg,
                  border,
                  outline: isHover ? "2px solid #5b7dff" : "none",
                  opacity: showMask && masked ? 0.85 : 1,
                  fontStyle: t.kind === "special" ? "italic" : "normal",
                }}
              >
                {display}
                {isHover && (
                  <span
                    className="absolute left-1/2 top-full z-10 mt-1 -translate-x-1/2 whitespace-nowrap rounded-md border border-white/10 bg-ink-850 px-2.5 py-1.5 text-[11px] font-normal shadow-lg"
                    style={{ pointerEvents: "none" }}
                  >
                    <span className="text-slate-300">
                      “{t.text}”
                    </span>
                    <span className="mx-1.5 text-slate-600">·</span>
                    <span className="text-slate-400">{t.role}</span>
                    <span className="mx-1.5 text-slate-600">·</span>
                    <span className="text-slate-500">id {t.id}</span>
                    <span className="mx-1.5 text-slate-600">·</span>
                    {t.trained ? (
                      <span className="text-emerald-300">label {t.id} (trained)</span>
                    ) : (
                      <span className="text-slate-500">label −100 (masked)</span>
                    )}
                  </span>
                )}
              </span>
            );
          })}
        </div>
      </div>

      {/* Summary stats + the masked turns explained */}
      <div className="grid gap-3 sm:grid-cols-3">
        <div className="rounded-lg border border-white/10 bg-ink-900/50 px-3 py-2">
          <div className="text-[10px] uppercase tracking-wider text-slate-500">
            Total tokens
          </div>
          <div className="font-mono text-lg text-slate-200">{tokens.length}</div>
        </div>
        <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/5 px-3 py-2">
          <div className="text-[10px] uppercase tracking-wider text-emerald-400/80">
            Trained (loss)
          </div>
          <div className="font-mono text-lg text-emerald-300">{trainedCount}</div>
        </div>
        <div className="rounded-lg border border-white/10 bg-ink-900/50 px-3 py-2">
          <div className="text-[10px] uppercase tracking-wider text-slate-500">
            Masked (−100)
          </div>
          <div className="font-mono text-lg text-slate-400">{maskedCount}</div>
        </div>
      </div>

      <p className="text-xs leading-relaxed text-slate-500">
        The model still <em>reads</em> the system and user tokens — they’re in the
        forward pass and condition the prediction. But because their labels are{" "}
        <code className="text-slate-400">−100</code>, cross-entropy ignores them:
        gradients flow only from the{" "}
        <span className="text-emerald-300">{trainedCount} assistant-response tokens</span>.
        That is the whole trick of loss masking.
      </p>
    </div>
  );
}
