# zen — the zenlm.org website

**Repository**: https://github.com/zenlm/zen · **Site**: https://zenlm.org

Zen LM is the open model family of Zoo Labs Foundation, a 501(c)(3) non-profit.
This repository is its website: a Next.js static export whose built pages are
committed under `docs/`.

## The story the site tells

- Two jobs lead every overview: agentic coding that runs on your own machine,
  and marketing work.
- **Zen 6** and **Zen 6 Flash** are available now (`zen6`, `zen6-flash` on
  api.hanzo.ai; weights `zenlm/zen6`, `zenlm/zen6-flash`). Numbers come from
  those two model cards and nowhere else.
- **Zen 7** is a research preview: no weights, not callable. "Request access"
  goes to https://hanzo.ai/research-access.
- Zen 5, Zen 4 and Zen 3 are earlier generations. Zen 5.8 and zen6-coder are
  retired and are not listed.

## Layout

- `src/app/` — pages: home, `/models`, `/datasets`, `/research`, `/blog`.
- `src/components/` — Header, Footer, CatalogSection (renders `src/data/catalog.json`).
- `src/lib/blog.ts` — reads `content/blog/*.md(x)` (gray-matter), lowers the four
  post components (Figure, LinkButton, Video, Fullwidth) to HTML and renders with
  unified/remark/rehype. It was missing from git until 2026-09-30 because an
  unanchored `lib/` in `.gitignore` hid it; that rule is anchored now.
- `content/blog/` — a copy of github.com/zenlm/zen-blog `content/`. Keep the two in
  step: a post fixed there is copied here.
- `docs/` — the committed export. `.github/workflows/pages.yml` uploads it as is;
  CI does not build.

## Build

```bash
npm install
npm run export   # next build → scripts/check-zen.mjs out → out/ becomes docs/
npm run typecheck
```

## Zen copy names Zen

`scripts/check-zen.mjs` reads every built page the way a reader meets it (title,
meta and social tags, text, alt) and fails on a name from `zen.upstream` outside a
`data-upstream` element. Upstream names appear in two marked places only: the
Architecture value (the loader string) and the License & attribution section. A
post whose subject is another lab's release sets `upstream: true` in its
frontmatter; its page then carries `<meta name="data-upstream">` and its article
and index card carry the attribute. It runs in `npm run export` and again in
`pages.yml` over `docs/`.

## Serving

zenlm.org is served by the edge's `staticFiles` middleware from
`s3://hanzo-sites/zen/zenlm` (universe `infra/aws/routes/sites.yaml`, route
`zenlm-org`), a copy of the Pages build. GitHub Pages still deploys on push, but
no DNS name points at it: a change reaches zenlm.org only when `docs/` is
published into that prefix.
