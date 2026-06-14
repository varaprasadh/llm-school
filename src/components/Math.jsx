import katex from "katex";
import { useMemo } from "react";

/**
 * Inline math. Usage: <M>{"a^2 + b^2 = c^2"}</M>
 */
export function M({ children }) {
  const html = useMemo(
    () =>
      katex.renderToString(String(children), {
        displayMode: false,
        throwOnError: false,
        strict: false,
        trust: true,
      }),
    [children]
  );
  return <span className="katex-inline" dangerouslySetInnerHTML={{ __html: html }} />;
}

/**
 * Display (block) math, centered, with horizontal scroll on overflow.
 * Usage: <MB>{String.raw`\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}`}</MB>
 */
export function MB({ children, label }) {
  const html = useMemo(
    () =>
      katex.renderToString(String(children), {
        displayMode: true,
        throwOnError: false,
        strict: false,
        trust: true,
      }),
    [children]
  );
  return (
    <div className="my-6 flex items-center justify-center gap-4">
      <div
        className="min-w-0 overflow-x-auto scrollbar-thin"
        dangerouslySetInnerHTML={{ __html: html }}
      />
      {label && <span className="shrink-0 text-sm text-slate-500 font-mono">({label})</span>}
    </div>
  );
}

export default M;
