import { Link } from "react-router-dom";
import { useAuth } from "../auth/AuthProvider";
import { allChapters } from "../data/chapters";
import GoogleButton from "./GoogleButton";

const FREE_SLUG = allChapters[0].slug;

/**
 * Gates its children behind a Google sign-in.
 *
 * - If Supabase is not configured, gating is inactive (open access) so the
 *   site works as before until keys are added.
 * - While the session is resolving, shows a spinner.
 * - Anonymous users get a login wall instead of the content.
 */
export default function RequireAuth({ children }) {
  const { configured, loading, session } = useAuth();

  if (!configured) return children;

  if (loading) {
    return (
      <div className="flex items-center justify-center py-32 text-slate-500">
        <span className="h-5 w-5 animate-spin rounded-full border-2 border-brand-500 border-t-transparent" />
      </div>
    );
  }

  if (!session) return <LoginWall />;

  return children;
}

function LoginWall() {
  return (
    <div className="mx-auto max-w-lg px-6 py-24 text-center">
      <div className="pointer-events-none absolute left-1/2 top-32 -z-10 h-64 w-[36rem] -translate-x-1/2 rounded-full bg-brand-500/15 blur-[120px]" />
      <span className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-ink-850/60 px-4 py-1.5 text-xs font-medium text-slate-400">
        <span className="h-1.5 w-1.5 rounded-full bg-accent-amber" />
        Members only
      </span>
      <h1 className="mt-6 text-3xl font-bold leading-tight sm:text-4xl">
        Sign in to unlock the full course
      </h1>
      <p className="mt-4 text-lg leading-relaxed text-slate-400">
        Chapter 1 is free for everyone. Create a free account with Google to read all{" "}
        {allChapters.length} chapters — your progress will follow you.
      </p>
      <div className="mt-8 flex justify-center">
        <GoogleButton />
      </div>
      <p className="mt-8 text-sm text-slate-500">
        Just exploring?{" "}
        <Link to={`/chapter/${FREE_SLUG}`} className="prose-link">
          Read Chapter 1 free →
        </Link>
      </p>
    </div>
  );
}
