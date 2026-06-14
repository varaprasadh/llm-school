import ErrorBoundary from "./ErrorBoundary";

/**
 * A captioned wrapper for interactive visualizations and diagrams.
 * <Figure n="8.1" title="Scaled dot-product attention">...</Figure>
 */
export default function Figure({ n, title, children, caption, full = false }) {
  return (
    <figure
      className={`my-8 ${full ? "" : "mx-auto"} overflow-hidden rounded-2xl border border-white/10 bg-ink-850/50 card-glow`}
    >
      {(n || title) && (
        <figcaption className="flex flex-wrap items-baseline gap-x-2 border-b border-white/5 px-5 py-3">
          {n && (
            <span className="font-mono text-xs font-semibold text-brand-300">Figure {n}</span>
          )}
          {title && <span className="text-sm font-medium text-slate-200">{title}</span>}
        </figcaption>
      )}
      <div className="p-4 sm:p-6">
        <ErrorBoundary label="visualization">{children}</ErrorBoundary>
      </div>
      {caption && (
        <div className="border-t border-white/5 px-5 py-3 text-sm leading-relaxed text-slate-400">
          {caption}
        </div>
      )}
    </figure>
  );
}
