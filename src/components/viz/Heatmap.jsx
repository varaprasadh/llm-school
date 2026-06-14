import { heat, fmt } from "./scales";

/**
 * A generic matrix heatmap.
 *
 * props:
 *  - matrix: number[][]  (row-major). Values are normalized per the `domain`.
 *  - rowLabels, colLabels: string[]
 *  - domain: [min,max] for color mapping (defaults to data min/max)
 *  - showValues: render the numeric value in each cell
 *  - cell: pixel size of each cell
 *  - highlightRow / highlightCol: index to emphasize
 *  - onHoverCell: (r,c)=>void
 */
export default function Heatmap({
  matrix,
  rowLabels,
  colLabels,
  domain,
  showValues = false,
  cell = 38,
  gap = 3,
  highlightRow = -1,
  highlightCol = -1,
  colorOf,
  caption,
}) {
  const rows = matrix.length;
  const cols = matrix[0]?.length ?? 0;
  let min = Infinity;
  let max = -Infinity;
  for (const row of matrix)
    for (const v of row) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
  const [d0, d1] = domain ?? [min, max];
  const norm = (v) => (d1 === d0 ? 0.5 : (v - d0) / (d1 - d0));
  const color = colorOf || ((v) => heat(norm(v)));

  return (
    <div className="inline-block">
      <div className="flex">
        {/* row label gutter */}
        {rowLabels && <div style={{ width: 0 }} />}
        <div>
          {/* column labels */}
          {colLabels && (
            <div className="flex" style={{ gap, marginLeft: rowLabels ? 64 : 0 }}>
              {colLabels.map((l, c) => (
                <div
                  key={c}
                  className={`flex items-end justify-center pb-1 font-mono text-[11px] ${
                    c === highlightCol ? "text-brand-300" : "text-slate-500"
                  }`}
                  style={{ width: cell, height: 22 }}
                  title={l}
                >
                  <span className="truncate">{l}</span>
                </div>
              ))}
            </div>
          )}

          {matrix.map((row, r) => (
            <div key={r} className="flex items-center" style={{ gap, marginBottom: gap }}>
              {rowLabels && (
                <div
                  className={`pr-2 text-right font-mono text-[11px] ${
                    r === highlightRow ? "text-brand-300" : "text-slate-500"
                  }`}
                  style={{ width: 64 }}
                  title={rowLabels[r]}
                >
                  <span className="block truncate">{rowLabels[r]}</span>
                </div>
              )}
              {row.map((v, c) => {
                const emph =
                  (highlightRow === r || highlightRow < 0) &&
                  (highlightCol === c || highlightCol < 0);
                const t = norm(v);
                return (
                  <div
                    key={c}
                    className="flex items-center justify-center rounded-md transition-all duration-300"
                    style={{
                      width: cell,
                      height: cell,
                      background: color(v),
                      opacity: emph ? 1 : 0.32,
                      outline:
                        r === highlightRow && c === highlightCol
                          ? "2px solid #e9faff"
                          : "none",
                    }}
                    title={`${rowLabels ? rowLabels[r] + " → " : ""}${
                      colLabels ? colLabels[c] : ""
                    } = ${fmt(v, 3)}`}
                  >
                    {showValues && (
                      <span
                        className="font-mono text-[10px] font-medium"
                        style={{ color: t > 0.55 ? "#0b0e1a" : "#cbd5e1" }}
                      >
                        {fmt(v, 2)}
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </div>
      {caption && <div className="mt-2 text-xs text-slate-500">{caption}</div>}
    </div>
  );
}
