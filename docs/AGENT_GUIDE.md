# Chapter Authoring Guide — The LLM Bible

You are writing **one chapter** (or a small set) of an interactive, visual textbook on
training and deploying LLMs from scratch. It is a React + Vite + Tailwind site. This guide is
everything you need. **Match the quality and house style of the existing reference chapters:**
`src/chapters/what-is-an-llm.jsx`, `src/chapters/tokenization.jsx`, and
`src/chapters/self-attention.jsx`. Read at least `self-attention.jsx` before you start — it is the
gold standard.

---

## Hard rules (do not break these)

1. **Only CREATE new files.** Never edit any shared/existing file (no edits to
   `data/chapters.js`, `package.json`, `index.css`, `App.jsx`, or any existing component). The
   routing auto-discovers your chapter file.
2. **Your chapter file path is exactly** `src/chapters/<slug>.jsx` where `<slug>` is given in your
   assignment. It must `export default function Chapter() { ... }`.
3. **Chapter-specific visualizations** you create go in `src/components/viz/<slug>/<Name>.jsx`
   (namespaced by your slug so nothing collides with other authors). Example:
   `src/components/viz/embeddings/EmbeddingSpace.jsx`.
4. **No new npm dependencies.** Use only what's installed: React, react-router-dom, framer-motion,
   katex (via the `Math` component), and plain SVG/CSS for visuals. Do **not** import d3, recharts,
   three, etc.
5. **No `Date.now()` / `Math.random()` at module top-level or during render** in a way that breaks
   determinism is fine in event handlers, but prefer the seeded `mulberry32` from `viz/scales`.
6. Keep everything **dark-theme** friendly (the site is dark). Use the palette below.

---

## Imports & exact paths

From a chapter file at `src/chapters/<slug>.jsx`:

```jsx
import { M, MB } from "../components/Math";          // inline + block math (KaTeX)
import Callout from "../components/Callout";
import Figure from "../components/Figure";
import CodeBlock from "../components/CodeBlock";
import LineChart from "../components/viz/LineChart";  // reusable chart
import Heatmap from "../components/viz/Heatmap";       // reusable matrix heatmap
import { Slider, SegmentedControl, Toggle, Button, Stat, Legend, useInterval, useStepper } from "../components/viz/ui";
import { heat, diverge, softmax, mulberry32, dot, fmt, clamp, lerp } from "../components/viz/scales";
// your own chapter viz:
import EmbeddingSpace from "../components/viz/<slug>/EmbeddingSpace";
```

From your own viz at `src/components/viz/<slug>/<Name>.jsx`, the shared helpers are one level up:

```jsx
import { heat, softmax, fmt } from "../scales";
import { Slider, Button } from "../ui";
import Heatmap from "../Heatmap";
```

---

## Content components (use these constantly)

### Prose
Just write semantic HTML inside the returned fragment. The chapter is auto-wrapped in a
`.chapter-prose` container that styles `<p> <h2> <h3> <h4> <ul> <ol> <li> <strong> <em> <code>
<table> <blockquote>` beautifully. **Use `<h2>` for top-level sections** (they become the
auto-generated table of contents) and `<h3>` for subsections.

```jsx
<h2>Section title</h2>
<p>A paragraph with <strong>bold</strong>, <em>italics</em>, and <code>inline code</code>.</p>
<ul><li>bullet</li></ul>
```

### Math — `<M>` inline, `<MB>` block
KaTeX. Pass a string (use `String.raw` for backslashes in block math).
```jsx
<p>The variance is <M>{"\\sigma^2"}</M> and we scale by <M>{"1/\\sqrt{d_k}"}</M>.</p>
<MB>{String.raw`\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}`}</MB>
```

### Callout — admonition boxes
`type` ∈ `note | tip | key | warning | pitfall | math | history | industry`. Optional `title`.
```jsx
<Callout type="key" title="The whole game"><p>One crisp takeaway.</p></Callout>
<Callout type="pitfall"><p>A common mistake and how to avoid it.</p></Callout>
```
Use 3–6 callouts per chapter. `key` for the central idea, `pitfall`/`warning` for gotchas,
`industry` for real-world practice, `math` for derivations, `history` for context.

### Figure — caption wrapper for any visual
```jsx
<Figure n="6.1" title="The embedding space" caption="One sentence explaining what to notice.">
  <EmbeddingSpace />
</Figure>
```
`n` is `"<chapterNumber>.<index>"` (e.g. the 2nd figure in ch.12 is `"12.2"`).

### CodeBlock — syntax-highlighted, copyable
`language` ∈ `python | bash | json | text | jsx`. Optional `filename`, `highlight={[lineNums]}`.
Put the code as a template-literal child. **Include real, correct, runnable code** (PyTorch for
modeling/training).
```jsx
<CodeBlock language="python" filename="train.py" highlight={[3]}>
{`import torch
model = GPT(...)
loss.backward()   # highlighted`}
</CodeBlock>
```

---

## Visualization toolkit

### Reusable: `LineChart`
For loss curves, LR schedules, scaling laws, throughput, anything x/y.
```jsx
<LineChart
  series={[
    { label: "train", color: "#22d3ee", points: [[0, 4.2], [1000, 2.1], [5000, 1.6]] },
    { label: "val",   color: "#fb7185", points: [[0, 4.3], [1000, 2.3], [5000, 1.9]], dashed: true },
  ]}
  xLabel="step" yLabel="loss" xScale="linear" yScale="linear" height={300}
/>
```
Supports `xScale="log"` / `yScale="log"`, `annotations={[{x, label, color}]}`,
`fmtX`/`fmtY` formatters.

### Reusable: `Heatmap`
For matrices (weights, attention, confusion, grids).
```jsx
<Heatmap matrix={[[0.1,0.9],[0.5,0.5]]} rowLabels={["a","b"]} colLabels={["x","y"]} showValues cell={40} />
```

### Reusable controls (`viz/ui`)
`Slider`, `SegmentedControl`, `Toggle`, `Button`, `Stat`, `Legend`, `PlaybackBar`,
`useInterval(cb, ms, playing)`, `useStepper(max)`. See `self-attention.jsx` / `BPETokenizer.jsx`
for usage patterns.

### Color helpers (`viz/scales`)
`heat(t)` t∈[0,1] → ink→indigo→cyan→white. `diverge(t)` t∈[-1,1] → rose→slate→emerald.
`softmax(arr, temp)`, `mulberry32(seed)` (deterministic RNG), `dot(a,b)`, `fmt(x, digits)`.

### Palette (hex)
brand `#5b7dff`, cyan `#22d3ee`, violet `#a855f7`, amber `#f59e0b`, emerald `#34d399`,
rose `#fb7185`. Surfaces: `#0b0e1a` (bg), `#141a2e` / `#1c2438` (panels), borders `rgba(255,255,255,0.1)`.

### Flagship viz you may import if relevant
- `viz/NextTokenDemo` — autoregressive sampling + temperature (ch1/3)
- `viz/AttentionExplorer` — self-attention matrix (ch8/9)
- `viz/BPETokenizer` — BPE training + encoding (ch5)
- `viz/LifecyclePipeline` — end-to-end pipeline (ch1/25)

---

## Every chapter MUST include

- **1–2 interactive or custom SVG visualizations** wrapped in `<Figure>`. Build a bespoke one for
  your topic (animated, draggable, steppable, or hover-driven). Static labeled SVG diagrams count
  if they're genuinely illustrative, but prefer at least one *interactive* element (slider/toggle/
  step). Reuse `LineChart`/`Heatmap` where they fit.
- **At least one realistic `CodeBlock`** (PyTorch/Python/bash as appropriate), correct and runnable.
- **3–6 `Callout`s**, including exactly one `type="key"` with the central insight.
- **Several `<h2>` sections** (5–9) so the TOC is useful, plus math via `<M>`/`<MB>` where it
  clarifies.
- A short closing `<h2>` that **bridges to the next chapter**.
- Length: substantial — comparable to the reference chapters (roughly 250–450 lines of JSX).

## Voice & pedagogy
- Direct, warm, second person ("you"), confident. Explain *why*, not just *what*.
- Define every term on first use. Assume a smart beginner who can write a for-loop.
- Lead with intuition, then formalize with math, then show code. Use concrete examples.
- Accuracy is paramount — this teaches people. No hand-waving, no incorrect formulas. If you state
  a number (params, FLOPs, dates), make it correct and realistic.
- Don't repeat long derivations that belong to another chapter — link with
  `<a href="/chapter/<slug>" className="prose-link">Chapter N</a>`.

## Self-check before finishing
- File at `src/chapters/<slug>.jsx`, `export default function Chapter()`.
- All imports resolve to the exact paths above; every JSX tag is closed; no stray `{}`.
- Block math uses `String.raw`; backslashes are escaped in inline `<M>{"..."}</M>` strings.
- Any component you reference is either listed here or created by you under your slug namespace.
- It reads like the reference chapters: visual, rigorous, and genuinely useful.
