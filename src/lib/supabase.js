import { createClient } from "@supabase/supabase-js";

/**
 * Supabase client.
 *
 * Reads public config from Vite env vars (safe to ship — the anon key is
 * protected by Row Level Security on the database side):
 *   VITE_SUPABASE_URL
 *   VITE_SUPABASE_ANON_KEY
 *
 * If they are absent the client is `null` and `supabaseConfigured` is false.
 * Auth gating treats "not configured" as open access, so the site keeps
 * working exactly as before until you add the keys (incremental rollout).
 */
const url = import.meta.env.VITE_SUPABASE_URL;
const anonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

export const supabaseConfigured = Boolean(url && anonKey);

export const supabase = supabaseConfigured ? createClient(url, anonKey) : null;
