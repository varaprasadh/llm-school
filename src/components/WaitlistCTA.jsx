import { useState } from "react";
import { useAuth } from "../auth/AuthProvider";
import { supabase, supabaseConfigured } from "../lib/supabase";

/**
 * "Practice playground — coming soon · join the waitlist" section.
 * Captures an email into the Supabase `waitlist` table. Pre-fills the email
 * for signed-in users.
 */
export default function WaitlistCTA() {
  const { user } = useAuth();
  const [email, setEmail] = useState(user?.email || "");
  const [status, setStatus] = useState("idle"); // idle | sending | done | error
  const [error, setError] = useState("");

  const submit = async (e) => {
    e.preventDefault();
    const trimmed = email.trim();
    if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(trimmed)) {
      setStatus("error");
      setError("Please enter a valid email address.");
      return;
    }
    setStatus("sending");
    setError("");

    if (!supabaseConfigured) {
      // No backend yet — acknowledge optimistically so the UI is testable.
      setStatus("done");
      return;
    }

    const { error: dbError } = await supabase
      .from("waitlist")
      .insert({ email: trimmed, source: "practice-playground" });

    // 23505 = unique violation → already on the list, treat as success.
    if (dbError && dbError.code !== "23505") {
      setStatus("error");
      setError("Something went wrong. Please try again.");
      return;
    }
    setStatus("done");
  };

  return (
    <section className="mx-auto mt-24 max-w-4xl">
      <div className="relative overflow-hidden rounded-2xl border border-white/10 bg-gradient-to-br from-ink-850/80 to-ink-900/60 p-8 sm:p-10">
        <div className="pointer-events-none absolute -right-20 -top-20 h-64 w-64 rounded-full bg-accent-violet/20 blur-[100px]" />
        <div className="relative">
          <span className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-ink-900/60 px-4 py-1.5 text-xs font-medium text-slate-400">
            <span className="h-1.5 w-1.5 animate-pulse-soft rounded-full bg-accent-violet" />
            Coming soon
          </span>
          <h2 className="mt-5 text-2xl font-bold sm:text-3xl">
            The Practice Playground
          </h2>
          <p className="mt-3 max-w-2xl text-slate-400">
            Reading is step one. Soon you'll <em>build</em> the pieces yourself — write a
            tokenizer, implement attention, train a tiny model — right in the browser, with
            instant feedback on every exercise. Join the waitlist to get early access.
          </p>

          {status === "done" ? (
            <div className="mt-6 inline-flex items-center gap-2 rounded-xl border border-accent-emerald/30 bg-accent-emerald/10 px-5 py-3 text-sm font-medium text-accent-emerald">
              ✓ You're on the list — we'll be in touch.
            </div>
          ) : (
            <form onSubmit={submit} className="mt-6 flex flex-col gap-3 sm:flex-row sm:items-start">
              <div className="flex-1">
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@example.com"
                  className="w-full rounded-xl border border-white/10 bg-ink-900/70 px-4 py-3 text-slate-200 placeholder:text-slate-600 focus:border-brand-500/60 focus:outline-none"
                  aria-label="Email address"
                />
                {status === "error" && (
                  <p className="mt-2 text-sm text-accent-rose">{error}</p>
                )}
              </div>
              <button
                type="submit"
                disabled={status === "sending"}
                className="rounded-xl bg-brand-500 px-6 py-3 font-semibold text-white shadow-lg shadow-brand-500/30 transition-transform hover:scale-[1.03] disabled:opacity-60"
              >
                {status === "sending" ? "Joining…" : "Join the waitlist"}
              </button>
            </form>
          )}
        </div>
      </div>
    </section>
  );
}
