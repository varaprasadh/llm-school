# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

"The LLM Bible" — an interactive, visual textbook (React + Vite SPA) teaching how to train and deploy an LLM from scratch. 25 chapters across 6 parts, each with prose, runnable PyTorch examples, KaTeX math, and custom interactive visualizations. It is content-heavy and presentation-only: there is no backend, no data fetching, no auth.

## Commands

```bash
npm install        # dependencies (Node 18+, developed on Node 20)
npm run dev        # Vite dev server → http://localhost:5173 (HMR)
npm run build      # production build → dist/  (this is also the lint/typecheck — see below)
npm run preview    # serve the production build (vite preview --host)
```

There is **no test runner, no ESLint, and no TypeScript**. `npm run build` is the only correctness gate — it must pass before considering any change done. The codebase is plain JavaScript + JSX.

## Architecture: content-as-code with auto-discovery

The whole site is driven by three files. Understand these and everything else follows:

- **`src/data/chapters.js`** — the single source of truth. Exports `parts` (the 6 parts, each with ordered `chapters` carrying `{num, slug, title, summary, minutes}`) plus derived helpers (`allChapters`, `chapterBySlug`, `neighbors`). Ordering, numbering, the sidebar, the home-page map, and prev/next all read from here.
- **`src/chapters/<slug>.jsx`** — one file per chapter. Each `export default function Chapter()` and returns a bare JSX fragment of content (headings, paragraphs, `<Figure>`, `<CodeBlock>`, `<Callout>`, math). The filename **must** equal the `slug` in `chapters.js`.
- **`src/chapterRegistry.js`** — `import.meta.glob("./chapters/*.jsx")` + `React.lazy` builds a `slug → lazy component` map automatically. `App.jsx` has a single route `/chapter/:slug`.

**Consequence:** adding a chapter is exactly two steps — add an entry to `chapters.js`, create `src/chapters/<slug>.jsx`. Routing, the searchable sidebar (`Sidebar.jsx`), code-splitting, and prev/next wire up with no further edits. Do **not** hand-register routes.

**`src/components/ChapterPage.jsx`** is the runtime glue and the most non-obvious file:
- Renders the chapter hero from `chapters.js` metadata, then the lazy chapter inside `<Suspense>` + `<ErrorBoundary>`.
- Wraps the chapter body in `<div className="chapter-prose">`. **Chapters write semantic HTML (`<h2> <p> <ul> <table>`) and are styled by the `.chapter-prose` rules in `src/index.css`** — they do not import heading/paragraph components.
- Auto-generates the "On this page" TOC by scanning rendered `<h2>/<h3>` with a `MutationObserver` (the lazy content mounts after first paint), slugifying and assigning `id`s, with scroll-spy via `IntersectionObserver`. So a chapter just needs to use `<h2>`/`<h3>` for sections; the TOC is free.

## Content primitives (used by every chapter)

Located in `src/components/`. These are the only building blocks chapters should reach for:
- **`Math.jsx`** — `<M>` inline, `<MB>` block (KaTeX, `throwOnError:false`).
- **`Callout.jsx`** — admonitions; `type` ∈ `note|tip|key|warning|pitfall|math|history|industry`.
- **`Figure.jsx`** — captioned wrapper for visuals; **wraps its children in an `ErrorBoundary`** so a broken viz degrades to an inline message instead of blanking the chapter.
- **`CodeBlock.jsx`** — syntax highlighting via `prism-react-renderer`; children are a template literal.

## Visualization system

`src/components/viz/` holds all interactive graphics (hand-built SVG/Canvas — **no charting library**, and do not add one).
- **Shared/reusable:** `Heatmap.jsx`, `LineChart.jsx` (multi-series, linear/log axes, hover), `ui.jsx` (`Slider`, `SegmentedControl`, `Toggle`, `Button`, `Stat`, `Legend`, `PlaybackBar`, `useInterval`, `useStepper`), `scales.js` (`heat`, `diverge`, `softmax`, `mulberry32`, `dot`, `fmt`, `clamp`, `lerp`).
- **Flagship, importable from any chapter:** `AttentionExplorer`, `BPETokenizer`, `NextTokenDemo`, `LifecyclePipeline`.
- **Per-chapter custom viz** live namespaced under `src/components/viz/<slug>/`.

**Import paths matter:** from a chapter (`src/chapters/x.jsx`) use `../components/viz/...`; from a per-chapter viz (`src/components/viz/<slug>/Y.jsx`) the shared helpers are one level up: `../scales`, `../ui`, `../Heatmap`, `../LineChart`.

## Conventions and gotchas (these have actually broken things)

- **No backticks inside `CodeBlock` content.** Children are a JS template literal; a stray `` ` `` (even inside a Python comment) terminates the string and breaks the build. Write `accum_steps`, not `` `accum_steps` ``.
- **Math escaping:** block math uses `` String.raw`...` ``; inline `<M>{"..."}</M>` must escape backslashes (`"\\sqrt{d}"`).
- **Determinism in viz:** never call `Math.random()` or `Date.now()` during render — use seeded `mulberry32(seed)` from `scales.js` for fixed-but-arbitrary values. Visuals must be reproducible.
- **Never wrap an interactive `<button>` in a `<label>`** (e.g. the `Toggle` in `ui.jsx`). A `<button>` is a labelable element, so the label re-dispatches the click and the handler fires twice (net no-op). Make the whole control a single `<button>` with presentational `<span>`s inside.
- **Cross-references:** link with `<a href="/chapter/<slug>" className="prose-link">`. Keep the cited "Chapter N" consistent with `num` in `chapters.js` — they have drifted before. Every `/chapter/<slug>` must resolve to a real file.
- **Styling/theme:** Tailwind with `darkMode:"class"` (the `<html>` has `class="dark"`); custom `ink`/`brand`/`accent` palette in `tailwind.config.js`. Long-form reading styles are in `.chapter-prose` (index.css), not in components.

## Authoring reference

**`docs/AGENT_GUIDE.md`** is the full chapter-authoring spec (component API, house style, required structure, exact import paths). Read it before writing or substantially editing chapter content. The cleanest existing chapters to copy as templates are `src/chapters/self-attention.jsx` and `src/chapters/tokenization.jsx`.

## Verifying rendering (non-obvious)

`npm run build` catches compile/import errors but not render-time bugs. To verify pages actually render, **do not rely on headless Chrome `--dump-dom`/`--screenshot` with `--virtual-time-budget`** — it hangs on KaTeX-heavy pages because lazy font loading prevents the virtual clock from settling, producing misleading empty output. Instead drive a real browser and wait for actual content (e.g. Puppeteer `page.waitForFunction(() => document.body.innerText.length > 5000)`), checking for `.katex-error` nodes and page errors.
