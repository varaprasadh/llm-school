/**
 * Admonition / callout boxes used throughout the chapters.
 * <Callout type="key" title="Key idea">...</Callout>
 */

const VARIANTS = {
  note: {
    icon: "📝",
    border: "border-brand-500/40",
    bg: "bg-brand-500/5",
    title: "text-brand-200",
    defaultTitle: "Note",
  },
  tip: {
    icon: "💡",
    border: "border-emerald-500/40",
    bg: "bg-emerald-500/5",
    title: "text-emerald-200",
    defaultTitle: "Tip",
  },
  key: {
    icon: "🔑",
    border: "border-amber-400/50",
    bg: "bg-amber-400/[0.07]",
    title: "text-amber-200",
    defaultTitle: "Key idea",
  },
  warning: {
    icon: "⚠️",
    border: "border-rose-500/40",
    bg: "bg-rose-500/5",
    title: "text-rose-200",
    defaultTitle: "Watch out",
  },
  pitfall: {
    icon: "🕳️",
    border: "border-rose-500/40",
    bg: "bg-rose-500/[0.06]",
    title: "text-rose-200",
    defaultTitle: "Common pitfall",
  },
  math: {
    icon: "∑",
    border: "border-violet-500/40",
    bg: "bg-violet-500/5",
    title: "text-violet-200",
    defaultTitle: "The math",
  },
  history: {
    icon: "📜",
    border: "border-slate-500/40",
    bg: "bg-slate-500/5",
    title: "text-slate-200",
    defaultTitle: "A bit of history",
  },
  industry: {
    icon: "🏭",
    border: "border-cyan-500/40",
    bg: "bg-cyan-500/5",
    title: "text-cyan-200",
    defaultTitle: "In practice",
  },
};

export default function Callout({ type = "note", title, children }) {
  const v = VARIANTS[type] || VARIANTS.note;
  return (
    <div className={`my-6 rounded-xl border ${v.border} ${v.bg} p-4 pl-5`}>
      <div className={`mb-1.5 flex items-center gap-2 font-semibold ${v.title}`}>
        <span className="text-base leading-none" aria-hidden>
          {v.icon}
        </span>
        <span className="text-sm uppercase tracking-wide">{title || v.defaultTitle}</span>
      </div>
      <div className="callout-body text-[0.97rem] leading-relaxed text-slate-300/90 [&>p]:my-2 [&_a]:prose-link [&_code:not(pre_code)]:font-mono [&_code:not(pre_code)]:text-accent-cyan [&_code:not(pre_code)]:text-[0.85em]">
        {children}
      </div>
    </div>
  );
}
