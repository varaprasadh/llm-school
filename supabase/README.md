# Supabase — database as code

The schema lives in `migrations/` as versioned SQL. It is the **single source of
truth** — never change tables by hand in the dashboard; add a migration instead.

## One-time setup

```bash
npx supabase login                       # opens a browser, stores a token
npm run db:link                          # paste your project ref (Settings → General)
npm run db:push                          # applies migrations/*.sql to the remote DB
```

That creates the `waitlist` table (and its RLS policies) on your project.

You still configure **Auth → Google provider** in the dashboard (that part isn't
captured by migrations). See the project root setup notes / `.env.example`.

## Changing the schema later

```bash
npm run db:new add_progress_table        # creates migrations/<timestamp>_add_progress_table.sql
# ...edit that file with your CREATE/ALTER...
npm run db:push                          # apply to remote
```

`npm run db:diff` shows the difference between your migrations and the remote DB.

## Notes

- `config.toml` is for local development (`supabase start`, needs Docker) — not
  required for `db push` to a hosted project.
- The anon key in the app can only **insert** into `waitlist`; there is no SELECT
  policy, so the list is only readable via the dashboard / service role.
