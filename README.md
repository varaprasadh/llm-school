# The LLM School 📖

> A complete, visual, end-to-end guide to **training and deploying a large language model from scratch** — every concept, every example, every visualization. Built as an open educational resource for the next generation of ML engineers.

This is a React single-page application: **25 chapters across 6 parts**, each with rigorous prose, runnable PyTorch code, friendly math (KaTeX), and **custom interactive visualizations** (animated self-attention, a live BPE tokenizer, the embedding space, training curves, the KV cache, continuous batching, and more).

---

## Quick start

```bash
npm install      # install dependencies
npm run dev      # start the dev server → http://localhost:5173
```

Then open **http://localhost:5173** and start at Chapter 1.

```bash
npm run build    # production build → dist/
npm run preview  # serve the production build locally
```

Requires Node 18+ (developed on Node 20).

---

## The curriculum

| Part | Chapters | Topics |
|------|----------|--------|
| **I · Foundations** | 1–3 | What an LLM is · neural-network primer · the language-modeling objective |
| **II · Data & Representation** | 4–6 | Data curation · tokenization (BPE) · embeddings |
| **III · The Transformer** | 7–11 | Architecture overview · self-attention · multi-head · positional encoding (RoPE) · FFN/norm/residuals |
| **IV · Training** | 12–17 | Cross-entropy & perplexity · optimizers & schedules · building a GPT in PyTorch · the training loop · distributed training · scaling laws |
| **V · Post-Training** | 18–20 | Supervised fine-tuning · RLHF & DPO · LoRA / QLoRA |
| **VI · Evaluation & Deployment** | 21–25 | Benchmarks · inference optimization · serving at scale · monitoring & safety · capstone |

---

## Tech stack

- **Vite + React 18** — fast dev server, code-split per chapter (`React.lazy` + `import.meta.glob`)
- **React Router** — client-side routing, one route per chapter
- **Tailwind CSS** — dark "ink" theme + design tokens
- **KaTeX** — math typesetting
- **prism-react-renderer** — syntax-highlighted, copyable code blocks
- **Framer Motion** — landing-page animation
- Visualizations are **hand-built SVG/Canvas** — no charting dependency.

## Project structure

```
src/
├── main.jsx                 # entry; mounts the router
├── App.jsx                  # routes (Home + /chapter/:slug)
├── data/chapters.js         # ← single source of truth: parts, chapters, ordering
├── chapterRegistry.js       # auto-discovers src/chapters/*.jsx via import.meta.glob
├── pages/Home.jsx           # landing page + chapter map
├── components/
│   ├── Layout.jsx           # responsive sidebar + reading-progress bar
│   ├── Sidebar.jsx          # searchable chapter nav
│   ├── ChapterPage.jsx      # hero, auto-generated TOC w/ scroll-spy, prev/next
│   ├── Math.jsx Callout.jsx Figure.jsx CodeBlock.jsx   # content primitives
│   └── viz/                 # reusable (Heatmap, LineChart, ui, scales) +
│       └── <slug>/          # per-chapter custom visualizations
└── chapters/<slug>.jsx      # one file per chapter (the content)
```

### Adding or editing a chapter

1. Add/adjust the entry in `src/data/chapters.js`.
2. Create `src/chapters/<slug>.jsx` that `export default function Chapter()`.
3. That's it — routing, the sidebar, the TOC and prev/next wiring are automatic.

See [`docs/AGENT_GUIDE.md`](docs/AGENT_GUIDE.md) for the full component API and house style used to author the chapters.

---

## A note on accuracy

This is meant to teach. The code is real PyTorch (nanoGPT-faithful), the math is correct, and the numbers (parameter counts, FLOPs, dates) are grounded in the literature. Where a visualization simplifies for clarity, it says so. If you find an error, it's a bug — fix it.
