import { Suspense, useEffect, useRef, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { chapterBySlug, neighbors } from "../data/chapters";
import { chapterComponents } from "../chapterRegistry";
import ErrorBoundary from "./ErrorBoundary";

const ACCENT_TEXT = {
  cyan: "text-accent-cyan",
  violet: "text-accent-violet",
  amber: "text-accent-amber",
  emerald: "text-accent-emerald",
  rose: "text-accent-rose",
};

function slugify(s) {
  return s
    .toLowerCase()
    .replace(/[^\w\s-]/g, "")
    .trim()
    .replace(/\s+/g, "-")
    .slice(0, 60);
}

export default function ChapterPage() {
  const { slug } = useParams();
  const chapter = chapterBySlug[slug];
  const Comp = chapterComponents[slug];
  const contentRef = useRef(null);
  const lastSigRef = useRef("");
  const [toc, setToc] = useState([]);
  const [activeId, setActiveId] = useState(null);

  // Build the table of contents from rendered headings; keep it in sync as the
  // lazy chapter content mounts (MutationObserver) and dedupe ids.
  //
  // The chapter body can be a very large subtree (math-heavy chapters render
  // thousands of KaTeX nodes), and interactive visualizations mutate their own
  // DOM on every slider tick. To keep that cheap we (1) coalesce mutation bursts
  // into a single rebuild per animation frame, and (2) skip the state update
  // entirely when the set of headings hasn't actually changed — so dragging a
  // slider no longer forces a TOC re-render and IntersectionObserver rebuild.
  useEffect(() => {
    const el = contentRef.current;
    if (!el) return undefined;
    lastSigRef.current = "";
    let raf = 0;
    const build = () => {
      const heads = el.querySelectorAll("h2, h3");
      const seen = {};
      const items = [];
      heads.forEach((h) => {
        if (!h.id) {
          let base = slugify(h.textContent || "section");
          seen[base] = (seen[base] || 0) + 1;
          h.id = seen[base] > 1 ? `${base}-${seen[base]}` : base;
        }
        items.push({
          id: h.id,
          text: h.textContent,
          level: h.tagName === "H2" ? 2 : 3,
        });
      });
      const sig = items.map((i) => `${i.level}|${i.id}|${i.text}`).join("§");
      if (sig === lastSigRef.current) return; // headings unchanged — no churn
      lastSigRef.current = sig;
      setToc(items);
    };
    const schedule = () => {
      if (raf) return;
      raf = requestAnimationFrame(() => {
        raf = 0;
        build();
      });
    };
    build(); // initial pass, synchronous
    const obs = new MutationObserver(schedule);
    obs.observe(el, { childList: true, subtree: true });
    return () => {
      obs.disconnect();
      if (raf) cancelAnimationFrame(raf);
    };
  }, [slug]);

  // Scroll-spy for the active TOC entry.
  useEffect(() => {
    if (!toc.length) return undefined;
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (visible[0]) setActiveId(visible[0].target.id);
      },
      { rootMargin: "-80px 0px -70% 0px", threshold: 0 }
    );
    toc.forEach((t) => {
      const el = document.getElementById(t.id);
      if (el) observer.observe(el);
    });
    return () => observer.disconnect();
  }, [toc]);

  if (!chapter || !Comp) {
    return (
      <div className="mx-auto max-w-2xl px-6 py-32 text-center">
        <h1 className="text-3xl">Chapter not found</h1>
        <p className="mt-4 text-slate-400">
          The chapter “{slug}” doesn’t exist yet.
        </p>
        <Link to="/" className="mt-6 inline-block prose-link">
          ← Back to the table of contents
        </Link>
      </div>
    );
  }

  const { prev, next } = neighbors(slug);
  const accent = ACCENT_TEXT[chapter.accent] || "text-brand-300";

  return (
    <div className="mx-auto max-w-screen-xl px-4 py-8 sm:px-8 sm:py-12">
      <div className="xl:grid xl:grid-cols-[minmax(0,1fr)_220px] xl:gap-10">
        <article className="min-w-0">
          {/* Hero */}
          <header className="mx-auto max-w-3xl border-b border-white/5 pb-8">
            <div className="flex items-center gap-3 text-sm">
              <span className={`font-mono font-semibold ${accent}`}>
                Chapter {String(chapter.num).padStart(2, "0")}
              </span>
              <span className="text-slate-600">·</span>
              <span className="text-slate-500">{chapter.partTitle}</span>
              <span className="text-slate-600">·</span>
              <span className="text-slate-500">{chapter.minutes} min read</span>
            </div>
            <h1 className="mt-3 text-3xl font-bold leading-tight sm:text-4xl md:text-5xl">
              {chapter.title}
            </h1>
            <p className="mt-4 text-lg leading-relaxed text-slate-400">{chapter.summary}</p>
          </header>

          {/* Body */}
          <div ref={contentRef} className="chapter-prose mx-auto max-w-3xl pt-6">
            <Suspense
              fallback={
                <div className="flex items-center gap-3 py-20 text-slate-500">
                  <span className="h-4 w-4 animate-spin rounded-full border-2 border-brand-500 border-t-transparent" />
                  Loading chapter…
                </div>
              }
            >
              <ErrorBoundary label="chapter">
                <Comp />
              </ErrorBoundary>
            </Suspense>
          </div>

          {/* Prev / Next */}
          <nav className="mx-auto mt-16 flex max-w-3xl flex-col gap-3 border-t border-white/5 pt-8 sm:flex-row sm:justify-between">
            {prev ? (
              <Link
                to={`/chapter/${prev.slug}`}
                className="group flex-1 rounded-xl border border-white/10 bg-ink-850/50 p-4 transition-colors hover:border-brand-500/40"
              >
                <div className="text-xs text-slate-500">← Previous</div>
                <div className="mt-1 font-medium text-slate-200 group-hover:text-white">
                  {prev.num}. {prev.title}
                </div>
              </Link>
            ) : (
              <span className="flex-1" />
            )}
            {next ? (
              <Link
                to={`/chapter/${next.slug}`}
                className="group flex-1 rounded-xl border border-white/10 bg-ink-850/50 p-4 text-right transition-colors hover:border-brand-500/40"
              >
                <div className="text-xs text-slate-500">Next →</div>
                <div className="mt-1 font-medium text-slate-200 group-hover:text-white">
                  {next.num}. {next.title}
                </div>
              </Link>
            ) : (
              <span className="flex-1" />
            )}
          </nav>
        </article>

        {/* On-this-page TOC */}
        <aside className="hidden xl:block">
          <div className="sticky top-8">
            {toc.length > 0 && (
              <>
                <div className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-500">
                  On this page
                </div>
                <ul className="space-y-1.5 border-l border-white/10 text-sm">
                  {toc.map((t) => (
                    <li key={t.id} style={{ paddingLeft: t.level === 3 ? 16 : 0 }}>
                      <a
                        href={`#${t.id}`}
                        className={`-ml-px block border-l-2 pl-3 leading-snug transition-colors ${
                          activeId === t.id
                            ? "border-brand-400 text-brand-200"
                            : "border-transparent text-slate-500 hover:text-slate-300"
                        }`}
                      >
                        {t.text}
                      </a>
                    </li>
                  ))}
                </ul>
              </>
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}
