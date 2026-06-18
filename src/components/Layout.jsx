import { useEffect, useState } from "react";
import { Outlet, useLocation } from "react-router-dom";
import Sidebar from "./Sidebar";

export default function Layout() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [progress, setProgress] = useState(0);
  const location = useLocation();

  // Close the mobile drawer & scroll to top on route change.
  useEffect(() => {
    setMobileOpen(false);
    window.scrollTo({ top: 0, behavior: "instant" in window ? "instant" : "auto" });
  }, [location.pathname]);

  // Reading-progress bar.
  useEffect(() => {
    const onScroll = () => {
      const h = document.documentElement;
      const max = h.scrollHeight - h.clientHeight;
      setProgress(max > 0 ? (h.scrollTop / max) * 100 : 0);
    };
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, [location.pathname]);

  return (
    <div className="min-h-screen lg:flex">
      {/* Reading progress */}
      <div className="fixed left-0 top-0 z-50 h-0.5 w-full bg-transparent">
        <div
          className="h-full bg-gradient-to-r from-brand-500 via-accent-cyan to-accent-violet transition-[width] duration-150"
          style={{ width: `${progress}%` }}
        />
      </div>

      {/* Desktop sidebar */}
      <aside className="sticky top-0 hidden h-screen w-72 shrink-0 border-r border-white/5 bg-ink-900/60 backdrop-blur-md lg:block">
        <Sidebar />
      </aside>

      {/* Mobile top bar */}
      <header className="sticky top-0 z-40 flex items-center justify-between border-b border-white/5 bg-ink-900/80 px-4 py-3 backdrop-blur-md lg:hidden">
        <button
          onClick={() => setMobileOpen(true)}
          className="flex items-center gap-2 text-slate-300"
          aria-label="Open navigation"
        >
          <span className="text-xl">☰</span>
          <span className="font-semibold text-white">The LLM School</span>
        </button>
        <img src="/favicon.svg" alt="" className="h-7 w-7" />
      </header>

      {/* Mobile drawer */}
      {mobileOpen && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <div
            className="absolute inset-0 bg-black/60 backdrop-blur-sm"
            onClick={() => setMobileOpen(false)}
          />
          <div className="absolute left-0 top-0 h-full w-80 max-w-[85vw] border-r border-white/10 bg-ink-900 shadow-2xl">
            <button
              onClick={() => setMobileOpen(false)}
              className="absolute right-3 top-4 z-10 text-slate-400 hover:text-white"
              aria-label="Close navigation"
            >
              ✕
            </button>
            <Sidebar onNavigate={() => setMobileOpen(false)} />
          </div>
        </div>
      )}

      {/* Content */}
      <main className="min-w-0 flex-1">
        <Outlet />
      </main>
    </div>
  );
}
