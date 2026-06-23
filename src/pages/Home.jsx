import { Link, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { parts, allChapters, totalMinutes, chapterCount } from "../data/chapters";
import LifecyclePipeline from "../components/viz/LifecyclePipeline";

const ACCENT = {
  cyan: { text: "text-accent-cyan", ring: "hover:border-accent-cyan/50", dot: "bg-accent-cyan" },
  violet: { text: "text-accent-violet", ring: "hover:border-accent-violet/50", dot: "bg-accent-violet" },
  amber: { text: "text-accent-amber", ring: "hover:border-accent-amber/50", dot: "bg-accent-amber" },
  emerald: { text: "text-accent-emerald", ring: "hover:border-accent-emerald/50", dot: "bg-accent-emerald" },
  rose: { text: "text-accent-rose", ring: "hover:border-accent-rose/50", dot: "bg-accent-rose" },
};

export default function Home() {
  const navigate = useNavigate();
  const hours = (totalMinutes / 60).toFixed(1);

  return (
    <div className="mx-auto max-w-screen-xl px-4 pb-24 sm:px-8">
      {/* Hero */}
      <section className="relative overflow-hidden pt-12 sm:pt-20">
        <div className="pointer-events-none absolute -top-32 left-1/2 h-72 w-[42rem] -translate-x-1/2 rounded-full bg-brand-500/20 blur-[120px]" />
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="relative mx-auto max-w-3xl text-center"
        >
          <span className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-ink-850/60 px-4 py-1.5 text-xs font-medium text-slate-400">
            <span className="h-1.5 w-1.5 animate-pulse-soft rounded-full bg-accent-emerald" />
            The complete, visual field guide
          </span>
          <h1 className="mt-6 text-4xl font-extrabold leading-[1.05] tracking-tight sm:text-6xl">
            Train & Deploy a{" "}
            <span className="bg-gradient-to-r from-brand-400 via-accent-cyan to-accent-violet bg-clip-text text-transparent">
              Large Language Model
            </span>{" "}
            from Scratch
          </h1>
          <p className="mx-auto mt-6 max-w-2xl text-lg leading-relaxed text-slate-400">
            Every concept, every line of code, every visualization — from a raw web page to a
            token streaming back to a user. {chapterCount} chapters that take you the whole way,
            assuming nothing.
          </p>
          <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
            <Link
              to={`/chapter/${allChapters[0].slug}`}
              className="rounded-xl bg-brand-500 px-6 py-3 font-semibold text-white shadow-lg shadow-brand-500/30 transition-transform hover:scale-[1.03]"
            >
              Start from Chapter 1 →
            </Link>
            <a
              href="#map"
              className="rounded-xl border border-white/10 bg-ink-850/60 px-6 py-3 font-semibold text-slate-200 transition-colors hover:border-white/20"
            >
              Browse the map
            </a>
          </div>

          <div className="mt-10 flex items-center justify-center gap-8 text-center">
            <Stat value={chapterCount} label="Chapters" />
            <Stat value={`${hours}h`} label="Reading" />
            <Stat value="6" label="Parts" />
            <Stat value="∞" label="Curiosity" />
          </div>
        </motion.div>

        {/* Lifecycle pipeline */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="mx-auto mt-14 max-w-4xl rounded-2xl border border-white/10 bg-ink-850/40 p-5 card-glow sm:p-7"
        >
          <div className="mb-4 text-center text-sm font-medium uppercase tracking-wider text-slate-500">
            The journey, end to end
          </div>
          <LifecyclePipeline onPick={(slug) => navigate(`/chapter/${slug}`)} />
        </motion.div>
      </section>

      {/* Who is this for */}
      <section className="mx-auto mt-20 max-w-4xl">
        <div className="grid gap-4 sm:grid-cols-3">
          {[
            {
              icon: "🧑‍🎓",
              title: "Assumes nothing",
              body: "We start at vectors and derivatives. If you can write a for-loop, you can follow along.",
            },
            {
              icon: "👁️",
              title: "Visual first",
              body: "Interactive diagrams for attention, tokenization, embeddings, training curves and more.",
            },
            {
              icon: "⌨️",
              title: "Real code",
              body: "Runnable PyTorch — a full GPT you can read end to end, plus training & serving scripts.",
            },
          ].map((c) => (
            <div key={c.title} className="rounded-2xl border border-white/10 bg-ink-850/40 p-5">
              <div className="text-2xl">{c.icon}</div>
              <div className="mt-3 font-semibold text-white">{c.title}</div>
              <p className="mt-1.5 text-sm leading-relaxed text-slate-400">{c.body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Chapter map */}
      <section id="map" className="mx-auto mt-24 max-w-5xl scroll-mt-24">
        <div className="mb-10 text-center">
          <h2 className="text-3xl font-bold sm:text-4xl">The complete curriculum</h2>
          <p className="mx-auto mt-3 max-w-2xl text-slate-400">
            Six parts, building strictly on one another. Read top to bottom, or jump to what you
            need.
          </p>
        </div>

        <div className="space-y-12">
          {parts.map((part, pi) => {
            const a = ACCENT[part.accent];
            return (
              <motion.div
                key={part.id}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-80px" }}
                transition={{ duration: 0.5 }}
              >
                <div className="mb-4 flex items-center gap-3">
                  <span className={`flex h-8 w-8 items-center justify-center rounded-lg bg-ink-800 font-mono text-sm font-bold ${a.text}`}>
                    {pi + 1}
                  </span>
                  <div>
                    <div className="flex items-baseline gap-2">
                      <span className={`text-xs font-semibold uppercase tracking-wider ${a.text}`}>
                        {part.label}
                      </span>
                      <h3 className="text-xl font-bold text-white">{part.title}</h3>
                    </div>
                    <p className="text-sm text-slate-500">{part.blurb}</p>
                  </div>
                </div>

                <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                  {part.chapters.map((c) => (
                    <Link
                      key={c.slug}
                      to={`/chapter/${c.slug}`}
                      className={`group flex flex-col rounded-xl border border-white/10 bg-ink-850/40 p-4 transition-all hover:-translate-y-0.5 ${a.ring}`}
                    >
                      <div className="flex items-center justify-between">
                        <span className={`font-mono text-xs font-bold ${a.text}`}>
                          {String(c.num).padStart(2, "0")}
                        </span>
                        <span className="text-[11px] text-slate-600">{c.minutes} min</span>
                      </div>
                      <div className="mt-2 font-semibold leading-snug text-slate-100 group-hover:text-white">
                        {c.title}
                      </div>
                      <p className="mt-1.5 line-clamp-3 text-xs leading-relaxed text-slate-500">
                        {c.summary}
                      </p>
                    </Link>
                  ))}
                </div>
              </motion.div>
            );
          })}
        </div>
      </section>

      <footer className="mx-auto mt-24 max-w-3xl border-t border-white/5 pt-8 text-center text-sm text-slate-600">
        <p>
          Built as an open educational resource — a complete school for how modern language models
          are made. Start at{" "}
          <Link to={`/chapter/${allChapters[0].slug}`} className="prose-link">
            Chapter 1
          </Link>{" "}
          and don’t skip the visualizations.
        </p>
      </footer>
    </div>
  );
}

function Stat({ value, label }) {
  return (
    <div>
      <div className="text-2xl font-bold text-white sm:text-3xl">{value}</div>
      <div className="text-xs uppercase tracking-wider text-slate-500">{label}</div>
    </div>
  );
}
