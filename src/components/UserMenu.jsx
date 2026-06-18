import { useAuth } from "../auth/AuthProvider";

/**
 * Account control for the sidebar footer.
 * Renders nothing until Supabase is configured; otherwise a sign-in button
 * (anonymous) or the signed-in identity + sign-out.
 */
export default function UserMenu() {
  const { configured, loading, user, signInWithGoogle, signOut } = useAuth();

  if (!configured) return null;

  if (loading) {
    return (
      <div className="border-t border-white/5 px-4 py-3 text-sm text-slate-600">Loading…</div>
    );
  }

  if (!user) {
    return (
      <div className="border-t border-white/5 px-4 py-3">
        <button
          onClick={() => signInWithGoogle()}
          className="w-full rounded-lg bg-brand-500 px-3 py-2 text-sm font-semibold text-white transition-colors hover:bg-brand-400"
        >
          Sign in
        </button>
      </div>
    );
  }

  const name = user.user_metadata?.full_name || user.email || "Account";
  const avatar = user.user_metadata?.avatar_url;

  return (
    <div className="flex items-center gap-2.5 border-t border-white/5 px-4 py-3">
      {avatar ? (
        <img src={avatar} alt="" className="h-7 w-7 shrink-0 rounded-full" />
      ) : (
        <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-brand-500/30 text-xs font-semibold text-brand-200">
          {name.charAt(0).toUpperCase()}
        </span>
      )}
      <span className="min-w-0 flex-1 truncate text-sm text-slate-300" title={name}>
        {name}
      </span>
      <button
        onClick={() => signOut()}
        className="shrink-0 text-xs text-slate-500 transition-colors hover:text-slate-300"
      >
        Sign out
      </button>
    </div>
  );
}
