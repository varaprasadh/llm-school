import { useMemo, useRef, useState } from "react";

/**
 * A compact, dependency-free SVG line chart supporting multiple series,
 * linear or log axes, gridlines, a legend and a hover readout.
 *
 * series: [{ label, color, points: [[x,y], ...], dashed? }]
 */
export default function LineChart({
  series,
  width = 640,
  height = 320,
  xLabel,
  yLabel,
  xScale = "linear",
  yScale = "linear",
  xTicks = 5,
  yTicks = 5,
  margin = { top: 16, right: 18, bottom: 42, left: 56 },
  fmtX = (v) => trim(v),
  fmtY = (v) => trim(v),
  annotations = [],
}) {
  const wrapRef = useRef(null);
  const [hoverX, setHoverX] = useState(null);

  const all = series.flatMap((s) => s.points);
  const xs = all.map((p) => p[0]);
  const ys = all.map((p) => p[1]);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);

  const iw = width - margin.left - margin.right;
  const ih = height - margin.top - margin.bottom;

  const sx = useMemo(() => makeScale(xScale, xMin, xMax, 0, iw), [xScale, xMin, xMax, iw]);
  const sy = useMemo(() => makeScale(yScale, yMin, yMax, ih, 0), [yScale, yMin, yMax, ih]);

  const xTickVals = ticks(xScale, xMin, xMax, xTicks);
  const yTickVals = ticks(yScale, yMin, yMax, yTicks);

  // nearest point per series at hover
  const readout =
    hoverX == null
      ? null
      : series.map((s) => {
          let best = s.points[0];
          let bd = Infinity;
          for (const p of s.points) {
            const d = Math.abs(p[0] - hoverX);
            if (d < bd) {
              bd = d;
              best = p;
            }
          }
          return { label: s.label, color: s.color, p: best };
        });

  const onMove = (e) => {
    const rect = wrapRef.current.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * width - margin.left;
    if (px < 0 || px > iw) return setHoverX(null);
    setHoverX(sx.invert(px));
  };

  return (
    <div className="w-full">
      <div ref={wrapRef} className="overflow-x-auto scrollbar-thin">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="block w-full min-w-[420px]"
          onMouseMove={onMove}
          onMouseLeave={() => setHoverX(null)}
        >
          <g transform={`translate(${margin.left},${margin.top})`}>
            {/* gridlines + y ticks */}
            {yTickVals.map((v) => (
              <g key={`y${v}`} transform={`translate(0,${sy(v)})`}>
                <line x1={0} x2={iw} stroke="#1c2438" strokeWidth={1} />
                <text x={-10} y={4} textAnchor="end" fontSize={11} fill="#64748b">
                  {fmtY(v)}
                </text>
              </g>
            ))}
            {/* x ticks */}
            {xTickVals.map((v) => (
              <g key={`x${v}`} transform={`translate(${sx(v)},${ih})`}>
                <line y1={0} y2={5} stroke="#475569" />
                <text y={18} textAnchor="middle" fontSize={11} fill="#64748b">
                  {fmtX(v)}
                </text>
              </g>
            ))}

            {/* annotations (vertical markers) */}
            {annotations.map((a, i) => (
              <g key={i} transform={`translate(${sx(a.x)},0)`}>
                <line y1={0} y2={ih} stroke={a.color || "#a855f7"} strokeDasharray="3 3" opacity={0.6} />
                {a.label && (
                  <text y={12} x={5} fontSize={10} fill={a.color || "#a855f7"}>
                    {a.label}
                  </text>
                )}
              </g>
            ))}

            {/* series */}
            {series.map((s) => (
              <path
                key={s.label}
                d={linePath(s.points, sx, sy)}
                fill="none"
                stroke={s.color}
                strokeWidth={2}
                strokeDasharray={s.dashed ? "5 4" : undefined}
                strokeLinejoin="round"
                strokeLinecap="round"
              />
            ))}

            {/* hover crosshair + dots */}
            {readout && hoverX != null && (
              <>
                <line
                  x1={sx(hoverX)}
                  x2={sx(hoverX)}
                  y1={0}
                  y2={ih}
                  stroke="#5b7dff"
                  strokeWidth={1}
                  opacity={0.5}
                />
                {readout.map((r) => (
                  <circle
                    key={r.label}
                    cx={sx(r.p[0])}
                    cy={sy(r.p[1])}
                    r={4}
                    fill={r.color}
                    stroke="#0b0e1a"
                    strokeWidth={1.5}
                  />
                ))}
              </>
            )}

            {/* axis labels */}
            {xLabel && (
              <text x={iw / 2} y={ih + 38} textAnchor="middle" fontSize={12} fill="#94a3b8">
                {xLabel}
              </text>
            )}
            {yLabel && (
              <text
                transform={`translate(${-44},${ih / 2}) rotate(-90)`}
                textAnchor="middle"
                fontSize={12}
                fill="#94a3b8"
              >
                {yLabel}
              </text>
            )}
          </g>
        </svg>
      </div>

      {/* legend + readout */}
      <div className="mt-2 flex flex-wrap items-center justify-between gap-x-6 gap-y-1 text-xs">
        <div className="flex flex-wrap gap-x-4 gap-y-1">
          {series.map((s) => (
            <span key={s.label} className="flex items-center gap-1.5 text-slate-400">
              <span
                className="inline-block h-2.5 w-4 rounded-sm"
                style={{ background: s.color, opacity: s.dashed ? 0.6 : 1 }}
              />
              {s.label}
            </span>
          ))}
        </div>
        {readout && (
          <div className="flex flex-wrap gap-x-3 font-mono text-slate-400">
            <span className="text-slate-500">{xLabel || "x"}={fmtX(hoverX)}</span>
            {readout.map((r) => (
              <span key={r.label} style={{ color: r.color }}>
                {fmtY(r.p[1])}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ---- scale helpers ----
function makeScale(type, d0, d1, r0, r1) {
  if (type === "log") {
    const l0 = Math.log10(Math.max(d0, 1e-12));
    const l1 = Math.log10(Math.max(d1, 1e-12));
    const f = (v) => r0 + ((Math.log10(Math.max(v, 1e-12)) - l0) / (l1 - l0)) * (r1 - r0);
    f.invert = (px) => Math.pow(10, l0 + ((px - r0) / (r1 - r0)) * (l1 - l0));
    return f;
  }
  const f = (v) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);
  f.invert = (px) => d0 + ((px - r0) / (r1 - r0)) * (d1 - d0);
  return f;
}

function ticks(type, d0, d1, n) {
  if (type === "log") {
    const out = [];
    const lo = Math.floor(Math.log10(Math.max(d0, 1e-12)));
    const hi = Math.ceil(Math.log10(Math.max(d1, 1e-12)));
    for (let e = lo; e <= hi; e++) out.push(Math.pow(10, e));
    return out;
  }
  const out = [];
  for (let i = 0; i <= n; i++) out.push(d0 + ((d1 - d0) * i) / n);
  return out;
}

function linePath(points, sx, sy) {
  return points
    .map((p, i) => `${i === 0 ? "M" : "L"} ${sx(p[0]).toFixed(2)} ${sy(p[1]).toFixed(2)}`)
    .join(" ");
}

function trim(v) {
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e6 || (a < 1e-3 && a > 0)) return v.toExponential(0);
  if (a >= 100) return v.toFixed(0);
  if (a >= 1) return v.toFixed(1);
  return v.toFixed(2);
}
