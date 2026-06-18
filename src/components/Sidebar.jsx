import { useMemo, useState } from "react";
import { NavLink, Link } from "react-router-dom";
import { parts, allChapters } from "../data/chapters";
import UserMenu from "./UserMenu";

const ACCENT_DOT = {
  cyan: "bg-accent-cyan",
  violet: "bg-accent-violet",
  amber: "bg-accent-amber",
  emerald: "bg-accent-emerald",
  rose: "bg-accent-rose",
};

export default function Sidebar({ onNavigate }) {
  const [query, setQuery] = useState("");

  const filtered = useMemo(() => {
    if (!query.trim()) return null;
    const q = query.toLowerCase();
    return allChapters.filter(
      (c) =>
        c.title.toLowerCase().includes(q) ||
        c.summary.toLowerCase().includes(q) ||
        String(c.num) === q
    );
  }, [query]);

  return (
    <div className="flex h-full flex-col">
      {/* Brand */}
      <Link
        to="/"
        onClick={onNavigate}
        className="flex items-center gap-3 px-5 py-5 transition-opacity hover:opacity-90"
      >
        <img src="/favicon.svg" alt="" className="h-9 w-9" />
        <div className="leading-tight">
          <div className="font-semibold text-white">The LLM School</div>
          <div className="text-[11px] uppercase tracking-wider text-slate-500">
            Train & deploy from scratch
          </div>
        </div>
      </Link>

      {/* Search */}
      <div className="px-4 pb-3">
        <div className="relative">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search chapters…"
            className="w-full rounded-lg border border-white/10 bg-ink-900/70 py-2 pl-9 pr-3 text-sm text-slate-200 placeholder:text-slate-600 focus:border-brand-500/60 focus:outline-none"
          />
          <svg
            className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <circle cx="11" cy="11" r="7" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
        </div>
      </div>

      {/* Nav list */}
      <nav className="scrollbar-thin flex-1 overflow-y-auto px-3 pb-8">
        {filtered ? (
          <div className="space-y-0.5">
            <div className="px-2 py-2 text-xs text-slate-500">
              {filtered.length} result{filtered.length === 1 ? "" : "s"}
            </div>
            {filtered.map((c) => (
              <ChapterLink key={c.slug} chapter={c} onNavigate={onNavigate} />
            ))}
          </div>
        ) : (
          parts.map((part) => (
            <div key={part.id} className="mb-4">
              <div className="flex items-center gap-2 px-2 pb-1 pt-2">
                <span className={`h-1.5 w-1.5 rounded-full ${ACCENT_DOT[part.accent]}`} />
                <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                  {part.label} · {part.title}
                </span>
              </div>
              <div className="space-y-0.5">
                {part.chapters.map((c) => (
                  <ChapterLink key={c.slug} chapter={c} onNavigate={onNavigate} />
                ))}
              </div>
            </div>
          ))
        )}
      </nav>

      <UserMenu />
    </div>
  );
}

function ChapterLink({ chapter, onNavigate }) {
  return (
    <NavLink
      to={`/chapter/${chapter.slug}`}
      onClick={onNavigate}
      className={({ isActive }) =>
        `group flex items-start gap-2.5 rounded-lg px-2 py-1.5 text-sm transition-colors ${
          isActive
            ? "bg-brand-500/15 text-white"
            : "text-slate-400 hover:bg-white/5 hover:text-slate-200"
        }`
      }
    >
      {({ isActive }) => (
        <>
          <span
            className={`mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-md font-mono text-[10px] ${
              isActive ? "bg-brand-500 text-white" : "bg-ink-700/70 text-slate-500 group-hover:text-slate-300"
            }`}
          >
            {chapter.num}
          </span>
          <span className="leading-snug">{chapter.title}</span>
        </>
      )}
    </NavLink>
  );
}
