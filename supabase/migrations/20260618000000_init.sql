-- The LLM School — initial schema.
-- Apply with:  npx supabase db push   (after `supabase link`)
-- This file is the single source of truth for the database; never edit tables
-- by hand in the dashboard — add a new migration instead.

-- ---------------------------------------------------------------------------
-- waitlist: emails captured by the "Practice Playground — coming soon" CTA.
-- ---------------------------------------------------------------------------
create table if not exists public.waitlist (
  id         uuid primary key default gen_random_uuid(),
  email      text not null unique,
  source     text,
  created_at timestamptz not null default now()
);

alter table public.waitlist enable row level security;

-- Anyone (signed in or not) may add themselves to the waitlist...
drop policy if exists "anyone can join waitlist" on public.waitlist;
create policy "anyone can join waitlist"
  on public.waitlist
  for insert
  to anon, authenticated
  with check (true);

-- ...but nobody can read the list through the API (only the service role /
-- dashboard can). No SELECT policy = no client reads.
