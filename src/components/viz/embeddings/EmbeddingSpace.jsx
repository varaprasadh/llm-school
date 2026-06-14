import { useMemo, useState } from "react";
import { SegmentedControl } from "../ui";
import { fmt } from "../scales";

/**
 * Interactive 2-D embedding space.
 *
 * ~28 words live at hand-placed coordinates arranged into clear topical
 * clusters (royalty, gender, animals, numbers, verbs of motion, tech). The
 * 2-D positions are a stand-in for a real d-dimensional embedding table — the
 * geometry is what matters, not the axes.
 *
 * Two modes:
 *  • "Explore"  — hover a word to highlight it and its nearest neighbours by
 *                 Euclidean distance.
 *  • "Analogy"  — pick A, B, C; we draw D = C + (B − A) as vector arrows and
 *                 snap to the nearest real word. Coordinates are tuned so that
 *                 king + (woman − man) lands essentially on queen.
 *
 * Coordinate system is a logical 100×100 box; an SVG viewBox maps it to pixels
 * (y is flipped so "up" reads as up). Everything is deterministic.
 */

// --- hand-placed embedding "table": [x, y] in a 100×100 logical box ---------
// The gender axis points right (+x); king→queen mirrors man→woman so the
// analogy king + (woman − man) ≈ queen holds geometrically.
const WORDS = [
  // gender pair (the analogy basis)
  { w: "man", x: 30, y: 70, c: "gender" },
  { w: "woman", x: 44, y: 70, c: "gender" },
  // royalty — shifted down from the gender pair by the same x-offset
  { w: "king", x: 32, y: 56, c: "royalty" },
  { w: "queen", x: 46, y: 56, c: "royalty" },
  { w: "prince", x: 30, y: 47, c: "royalty" },
  { w: "princess", x: 44, y: 47, c: "royalty" },
  { w: "throne", x: 38, y: 40, c: "royalty" },
  // animals
  { w: "cat", x: 74, y: 78, c: "animal" },
  { w: "dog", x: 80, y: 74, c: "animal" },
  { w: "kitten", x: 70, y: 84, c: "animal" },
  { w: "puppy", x: 84, y: 80, c: "animal" },
  { w: "lion", x: 88, y: 66, c: "animal" },
  { w: "wolf", x: 82, y: 62, c: "animal" },
  // numbers
  { w: "one", x: 14, y: 22, c: "number" },
  { w: "two", x: 20, y: 18, c: "number" },
  { w: "three", x: 27, y: 21, c: "number" },
  { w: "ten", x: 22, y: 28, c: "number" },
  { w: "hundred", x: 31, y: 30, c: "number" },
  // verbs of motion
  { w: "run", x: 60, y: 22, c: "verb" },
  { w: "running", x: 67, y: 26, c: "verb" },
  { w: "jump", x: 56, y: 30, c: "verb" },
  { w: "walk", x: 64, y: 16, c: "verb" },
  { w: "swim", x: 70, y: 19, c: "verb" },
  // tech / ML vocabulary
  { w: "model", x: 80, y: 40, c: "tech" },
  { w: "data", x: 86, y: 45, c: "tech" },
  { w: "token", x: 78, y: 48, c: "tech" },
  { w: "neural", x: 88, y: 36, c: "tech" },
  { w: "vector", x: 82, y: 33, c: "tech" },
];

const CLUSTERS = {
  gender: { label: "gender", color: "#f59e0b" },
  royalty: { label: "royalty", color: "#a855f7" },
  animal: { label: "animals", color: "#34d399" },
  number: { label: "numbers", color: "#22d3ee" },
  verb: { label: "verbs", color: "#fb7185" },
  tech: { label: "ML terms", color: "#5b7dff" },
};

const byWord = Object.fromEntries(WORDS.map((d) => [d.w, d]));

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

// cosine similarity treats each word's [x, y] as a vector from the origin.
function cosine(a, b) {
  const na = Math.hypot(a.x, a.y);
  const nb = Math.hypot(b.x, b.y);
  if (na === 0 || nb === 0) return 0;
  return (a.x * b.x + a.y * b.y) / (na * nb);
}

function nearest(point, exclude = []) {
  let best = null;
  let bestD = Infinity;
  for (const d of WORDS) {
    if (exclude.includes(d.w)) continue;
    const dd = Math.hypot(point.x - d.x, point.y - d.y);
    if (dd < bestD) {
      bestD = dd;
      best = d;
    }
  }
  return { word: best, d: bestD };
}

// --- SVG layout -------------------------------------------------------------
const VB = 100; // logical box
const PAD = 8;
// map logical (x,y in 0..100) to svg coords; flip y so "up" is up.
const sx = (x) => PAD + (x / VB) * (200 - 2 * PAD);
const sy = (y) => PAD + ((VB - y) / VB) * (200 - 2 * PAD);

export default function EmbeddingSpace() {
  const [mode, setMode] = useState("Explore");
  const [hover, setHover] = useState("king");
  const [A, setA] = useState("man");
  const [B, setB] = useState("woman");
  const [C, setC] = useState("king");

  // Explore mode: nearest neighbours of the hovered word.
  const neighbors = useMemo(() => {
    if (!hover) return [];
    const h = byWord[hover];
    return WORDS.filter((d) => d.w !== hover)
      .map((d) => ({ w: d.w, d: dist(h, d) }))
      .sort((a, b) => a.d - b.d)
      .slice(0, 3);
  }, [hover]);

  // Analogy mode: D = C + (B − A), then snap to nearest real word.
  const analogy = useMemo(() => {
    const a = byWord[A];
    const b = byWord[B];
    const c = byWord[C];
    const target = { x: c.x + (b.x - a.x), y: c.y + (b.y - a.y) };
    const snap = nearest(target, [A, B, C]);
    return { a, b, c, target, snap };
  }, [A, B, C]);

  const cosPair = useMemo(() => {
    if (mode === "Analogy") return cosine(byWord[A], byWord[C]);
    if (hover && neighbors.length) return cosine(byWord[hover], byWord[neighbors[0].w]);
    return null;
  }, [mode, A, C, hover, neighbors]);

  const neighborSet = new Set(neighbors.map((n) => n.w));
  const allWords = WORDS.map((d) => d.w).sort();

  return (
    <div className="space-y-5">
      {/* Controls */}
      <div className="flex flex-wrap items-end gap-5">
        <SegmentedControl
          label="Mode"
          options={["Explore", "Analogy"]}
          value={mode}
          onChange={setMode}
        />
        {mode === "Explore" ? (
          <div className="text-xs text-slate-500">
            Hover a point to highlight its{" "}
            <span className="text-emerald-300">3 nearest neighbours</span>.
          </div>
        ) : (
          <div className="flex flex-wrap items-end gap-3">
            <WordPick label="A" value={A} onChange={setA} options={allWords} accent="#fb7185" />
            <WordPick label="B" value={B} onChange={setB} options={allWords} accent="#34d399" />
            <WordPick label="C" value={C} onChange={setC} options={allWords} accent="#5b7dff" />
          </div>
        )}
      </div>

      <div className="grid gap-5 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]">
        {/* Scatter plot */}
        <div className="rounded-xl border border-white/10 bg-ink-900/50 p-2">
          <svg viewBox="0 0 200 200" className="block w-full" style={{ aspectRatio: "1 / 1" }}>
            {/* faint grid */}
            {[20, 40, 60, 80].map((g) => (
              <g key={g} stroke="#1c2438" strokeWidth="0.4">
                <line x1={sx(g)} y1={sy(0)} x2={sx(g)} y2={sy(100)} />
                <line x1={sx(0)} y1={sy(g)} x2={sx(100)} y2={sy(g)} />
              </g>
            ))}

            {/* Analogy arrows: B−A reference, and the same vector from C */}
            {mode === "Analogy" && (
              <>
                <defs>
                  <marker
                    id="arrow-emb"
                    viewBox="0 0 10 10"
                    refX="8"
                    refY="5"
                    markerWidth="5"
                    markerHeight="5"
                    orient="auto-start-reverse"
                  >
                    <path d="M 0 0 L 10 5 L 0 10 z" fill="#cbd5e1" />
                  </marker>
                </defs>
                {/* reference vector A -> B */}
                <line
                  x1={sx(analogy.a.x)}
                  y1={sy(analogy.a.y)}
                  x2={sx(analogy.b.x)}
                  y2={sy(analogy.b.y)}
                  stroke="#64748b"
                  strokeWidth="1"
                  strokeDasharray="3 2"
                  markerEnd="url(#arrow-emb)"
                />
                {/* same displacement applied at C -> target */}
                <line
                  x1={sx(analogy.c.x)}
                  y1={sy(analogy.c.y)}
                  x2={sx(analogy.target.x)}
                  y2={sy(analogy.target.y)}
                  stroke="#f59e0b"
                  strokeWidth="1.4"
                  markerEnd="url(#arrow-emb)"
                />
                {/* dashed snap to the nearest real word */}
                <line
                  x1={sx(analogy.target.x)}
                  y1={sy(analogy.target.y)}
                  x2={sx(analogy.snap.word.x)}
                  y2={sy(analogy.snap.word.y)}
                  stroke="#f59e0b"
                  strokeWidth="0.8"
                  strokeDasharray="2 2"
                  opacity="0.7"
                />
                {/* predicted point D */}
                <circle
                  cx={sx(analogy.target.x)}
                  cy={sy(analogy.target.y)}
                  r="2.6"
                  fill="none"
                  stroke="#f59e0b"
                  strokeWidth="1.2"
                />
                <text
                  x={sx(analogy.target.x) + 3}
                  y={sy(analogy.target.y) - 3}
                  fontSize="4.5"
                  fill="#f59e0b"
                  fontFamily="JetBrains Mono, monospace"
                >
                  D
                </text>
              </>
            )}

            {/* word points */}
            {WORDS.map((d) => {
              const col = CLUSTERS[d.c].color;
              let highlighted = false;
              let dimmed = false;
              if (mode === "Explore") {
                highlighted = d.w === hover || neighborSet.has(d.w);
                dimmed = hover && !highlighted;
              } else {
                const inAnalogy = [A, B, C, analogy.snap.word.w].includes(d.w);
                highlighted = inAnalogy;
                dimmed = !inAnalogy;
              }
              const isHover = mode === "Explore" && d.w === hover;
              const isSnap = mode === "Analogy" && d.w === analogy.snap.word.w;
              return (
                <g
                  key={d.w}
                  onMouseEnter={() => mode === "Explore" && setHover(d.w)}
                  style={{ cursor: mode === "Explore" ? "pointer" : "default" }}
                  opacity={dimmed ? 0.28 : 1}
                >
                  <circle
                    cx={sx(d.x)}
                    cy={sy(d.y)}
                    r={isHover || isSnap ? 3.4 : highlighted ? 2.8 : 2.2}
                    fill={col}
                    stroke={isHover || isSnap ? "#fff" : "transparent"}
                    strokeWidth="0.8"
                  />
                  <text
                    x={sx(d.x) + 3}
                    y={sy(d.y) + 1.5}
                    fontSize={highlighted ? "4.6" : "4"}
                    fontFamily="JetBrains Mono, monospace"
                    fill={highlighted ? "#e2e8f0" : "#94a3b8"}
                    fontWeight={highlighted ? 600 : 400}
                  >
                    {d.w}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>

        {/* Side panel */}
        <div className="space-y-3">
          {mode === "Explore" ? (
            <div className="rounded-xl border border-white/10 bg-ink-900/50 p-4">
              <div className="mb-2 text-sm text-slate-300">
                Nearest neighbours of{" "}
                <span className="rounded bg-brand-500/30 px-1.5 py-0.5 font-mono text-brand-100">
                  {hover}
                </span>
              </div>
              <div className="space-y-1.5">
                {neighbors.map((n) => {
                  const col = CLUSTERS[byWord[n.w].c].color;
                  return (
                    <div key={n.w} className="flex items-center gap-2 text-sm">
                      <span
                        className="inline-block h-2.5 w-2.5 shrink-0 rounded-full"
                        style={{ background: col }}
                      />
                      <span className="w-24 font-mono text-slate-200">{n.w}</span>
                      <span className="font-mono text-xs text-slate-500">
                        dist {fmt(n.d, 1)}
                      </span>
                    </div>
                  );
                })}
              </div>
              <p className="mt-3 text-xs leading-relaxed text-slate-500">
                Neighbours come from the same topical cluster — meaning lives in geometry, not in the
                token ID.
              </p>
            </div>
          ) : (
            <div className="rounded-xl border border-white/10 bg-ink-900/50 p-4">
              <div className="mb-1 text-sm text-slate-300">Vector analogy</div>
              <div className="font-mono text-sm leading-relaxed text-slate-200">
                <span style={{ color: "#5b7dff" }}>{C}</span>
                {" + ("}
                <span style={{ color: "#34d399" }}>{B}</span>
                {" − "}
                <span style={{ color: "#fb7185" }}>{A}</span>
                {")"}
              </div>
              <div className="mt-2 flex items-center gap-2 text-sm">
                <span className="text-slate-500">≈</span>
                <span className="rounded bg-amber-400/20 px-2 py-0.5 font-mono text-base text-amber-200">
                  {analogy.snap.word.w}
                </span>
                <span className="font-mono text-xs text-slate-500">
                  (nearest, dist {fmt(analogy.snap.d, 1)})
                </span>
              </div>
              <p className="mt-3 text-xs leading-relaxed text-slate-500">
                The grey arrow is the direction <span className="font-mono">B − A</span>. Re-applying
                that <em>same</em> displacement at <span className="font-mono">{C}</span> (amber
                arrow) lands at <span className="font-mono">D</span>; the closest real word is the
                answer. Try <span className="font-mono">man, woman, king</span>.
              </p>
            </div>
          )}

          {/* cosine similarity readout */}
          {cosPair != null && (
            <div className="rounded-xl border border-white/10 bg-ink-900/50 p-4">
              <div className="text-[10px] uppercase tracking-wider text-slate-500">
                cosine similarity
              </div>
              <div className="mt-1 flex items-baseline gap-2">
                <span className="font-mono text-2xl text-cyan-200">{fmt(cosPair, 3)}</span>
                <span className="font-mono text-xs text-slate-500">
                  {mode === "Analogy" ? `${C} · ${A}` : `${hover} · ${neighbors[0]?.w}`}
                </span>
              </div>
              <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-ink-800">
                <div
                  className="h-full rounded-full bg-cyan-400"
                  style={{ width: `${Math.max(0, cosPair) * 100}%` }}
                />
              </div>
              <p className="mt-2 text-xs text-slate-500">
                1 = same direction, 0 = orthogonal. Direction encodes meaning more than raw distance.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* cluster legend */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-400">
        {Object.values(CLUSTERS).map((c) => (
          <span key={c.label} className="flex items-center gap-1.5">
            <span className="inline-block h-3 w-3 rounded-full" style={{ background: c.color }} />
            {c.label}
          </span>
        ))}
      </div>
    </div>
  );
}

function WordPick({ label, value, onChange, options, accent }) {
  return (
    <label className="flex flex-col gap-1 text-xs">
      <span className="font-mono" style={{ color: accent }}>
        {label}
      </span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded-md border border-white/10 bg-ink-800 px-2 py-1 font-mono text-xs text-slate-200 outline-none focus:border-brand-400"
      >
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    </label>
  );
}
