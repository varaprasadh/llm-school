import { useState, useMemo } from "react";
import { Highlight } from "prism-react-renderer";

/**
 * A custom Prism theme tuned to the site's "ink" palette.
 */
const inkTheme = {
  plain: { color: "#cbd5e1", backgroundColor: "transparent" },
  styles: [
    { types: ["comment", "prolog", "doctype", "cdata"], style: { color: "#64748b", fontStyle: "italic" } },
    { types: ["punctuation"], style: { color: "#94a3b8" } },
    { types: ["property", "tag", "constant", "symbol", "deleted"], style: { color: "#fb7185" } },
    { types: ["boolean", "number"], style: { color: "#f59e0b" } },
    { types: ["selector", "attr-name", "string", "char", "builtin", "inserted"], style: { color: "#34d399" } },
    { types: ["operator", "entity", "url"], style: { color: "#22d3ee" } },
    { types: ["atrule", "attr-value", "keyword"], style: { color: "#8eabff" } },
    { types: ["function", "class-name"], style: { color: "#a855f7" } },
    { types: ["regex", "important", "variable"], style: { color: "#f59e0b" } },
    { types: ["decorator"], style: { color: "#22d3ee" } },
  ],
};

const LANG_LABEL = {
  python: "Python",
  py: "Python",
  bash: "Shell",
  sh: "Shell",
  shell: "Shell",
  jsx: "JSX",
  js: "JavaScript",
  json: "JSON",
  yaml: "YAML",
  text: "Text",
  diff: "Diff",
};

/**
 * <CodeBlock language="python" filename="model.py" highlight={[3,4]}>
 *   {`...source...`}
 * </CodeBlock>
 */
export default function CodeBlock({
  children,
  language = "python",
  filename,
  highlight = [],
  showLineNumbers = true,
}) {
  const [copied, setCopied] = useState(false);
  const code = useMemo(() => String(children).replace(/\n$/, ""), [children]);
  const highlightSet = useMemo(() => new Set(highlight), [highlight]);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      /* clipboard not available — ignore */
    }
  };

  return (
    <figure className="my-6 overflow-hidden rounded-xl border border-white/10 bg-ink-900/80 shadow-lg shadow-black/30">
      <figcaption className="flex items-center justify-between border-b border-white/5 bg-ink-850/70 px-4 py-2">
        <div className="flex items-center gap-2 min-w-0">
          <span className="flex gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full bg-rose-500/70" />
            <span className="h-2.5 w-2.5 rounded-full bg-amber-400/70" />
            <span className="h-2.5 w-2.5 rounded-full bg-emerald-400/70" />
          </span>
          {filename && (
            <span className="ml-2 truncate font-mono text-xs text-slate-400">{filename}</span>
          )}
        </div>
        <div className="flex items-center gap-3">
          <span className="rounded bg-ink-700/60 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider text-slate-400">
            {LANG_LABEL[language] || language}
          </span>
          <button
            onClick={copy}
            className="text-xs text-slate-400 transition-colors hover:text-brand-300"
            aria-label="Copy code"
          >
            {copied ? "✓ Copied" : "Copy"}
          </button>
        </div>
      </figcaption>

      <Highlight theme={inkTheme} code={code} language={language}>
        {({ className, style, tokens, getLineProps, getTokenProps }) => (
          <pre
            className={`${className} overflow-x-auto scrollbar-thin px-4 py-4 text-[0.84rem] leading-relaxed`}
            style={style}
          >
            <code>
              {tokens.map((line, i) => {
                const lineProps = getLineProps({ line });
                const isHi = highlightSet.has(i + 1);
                return (
                  <div
                    key={i}
                    {...lineProps}
                    className={`${lineProps.className} table-row ${
                      isHi ? "bg-brand-500/10" : ""
                    }`}
                  >
                    {showLineNumbers && (
                      <span
                        className={`table-cell select-none pr-4 text-right font-mono text-xs ${
                          isHi ? "text-brand-300" : "text-slate-600"
                        }`}
                        style={{ minWidth: "2.5rem" }}
                      >
                        {i + 1}
                      </span>
                    )}
                    <span className="table-cell">
                      {line.map((token, key) => (
                        <span key={key} {...getTokenProps({ token })} />
                      ))}
                    </span>
                  </div>
                );
              })}
            </code>
          </pre>
        )}
      </Highlight>
    </figure>
  );
}
