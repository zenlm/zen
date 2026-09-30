// Zen is told in Zen's names.
//
//   node scripts/check-zen.mjs <export dir>
//
// Reads every page of a static export the way a reader meets it — the title,
// the description and social tags, headings, prose, tables, alt text — and
// fails when a name from zen.upstream appears outside a `data-upstream`
// element. Two places carry one, and both are marked: the Architecture value
// (the loader string from the weights' own config, which a developer passes
// verbatim) and the License & attribution fine print. A post whose subject is
// another lab's release is marked the same way: its page carries
// <meta name="data-upstream">, and its card on an index carries the attribute.
//
// It reads the built HTML rather than the source, so a name that arrives from
// a JSON file, a component prop or a frontmatter field is caught exactly where
// it is shown. The names live in zen.upstream and nowhere else.

import fs from 'node:fs';
import path from 'node:path';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const NAMES = path.join(ROOT, 'zen.upstream');

/** One name per line; `#` starts a comment. */
export function loadNames(file = NAMES) {
  const names = fs
    .readFileSync(file, 'utf8')
    .split('\n')
    .map((l) => l.replace(/#.*$/, '').trim())
    .filter(Boolean);
  if (!names.length) throw new Error(`${file} names nothing, so the check would pass anything`);
  return names;
}

/** Any name as a whole word, in any case: no letter directly before or after it. */
export function matcher(names) {
  const alt = names.map((n) => n.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|');
  return new RegExp(`(?<![a-z])(?:${alt})(?![a-z])`, 'gi');
}

const VOID = new Set([
  'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'source', 'track', 'wbr',
]);

/** The page with every element that carries `data-upstream` cut out, children and all. */
export function dropUpstream(html) {
  const open = /<([a-zA-Z][\w-]*)\b[^>]*?\sdata-upstream(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]+))?[^>]*>/g;
  let out = html;
  for (let m = open.exec(out); m; m = open.exec(out)) {
    const tag = m[1].toLowerCase();
    const start = m.index;
    let end = start + m[0].length;
    if (!VOID.has(tag) && !m[0].endsWith('/>')) {
      const scan = new RegExp(`<(/?)${tag}\\b[^>]*>`, 'gi');
      scan.lastIndex = end;
      let depth = 1;
      for (let t = scan.exec(out); t; t = scan.exec(out)) {
        if (t[0].endsWith('/>')) continue;
        depth += t[1] ? -1 : 1;
        if (depth === 0) {
          end = t.index + t[0].length;
          break;
        }
      }
      if (depth !== 0) throw new Error(`unclosed <${tag} data-upstream> at offset ${start}`);
    }
    out = out.slice(0, start) + out.slice(end);
    open.lastIndex = start;
  }
  return out;
}

const ENTITIES = { amp: '&', lt: '<', gt: '>', quot: '"', apos: "'", nbsp: ' ', '#39': "'", '#x27': "'" };
const decode = (s) =>
  s.replace(/&(#x[0-9a-f]+|#\d+|\w+);/gi, (e, k) => {
    if (k[0] === '#') return String.fromCodePoint(k[1] === 'x' ? parseInt(k.slice(2), 16) : +k.slice(1));
    return ENTITIES[k] ?? e;
  });

/** Every string of a page a reader reads, leaving out `data-upstream` elements. */
export function readThrough(html) {
  // The page's subject is another lab's release, so its title and text name it.
  if (/<meta\b[^>]*\bname\s*=\s*"data-upstream"/i.test(html)) return [];
  const page = dropUpstream(
    html
      .replace(/<!--[\s\S]*?-->/g, '')
      .replace(/<(script|style|noscript|template)\b[\s\S]*?<\/\1>/gi, ''),
  );
  const out = [];
  // What the tab, a search result and a share card show.
  for (const m of page.matchAll(/<meta\b[^>]*>/gi)) {
    const tag = m[0];
    const key = /\b(?:name|property)\s*=\s*"([^"]*)"/i.exec(tag)?.[1] ?? '';
    const content = /\bcontent\s*=\s*"([^"]*)"/i.exec(tag)?.[1];
    if (content && /^(description|keywords|og:|twitter:)/i.test(key) && !/image|url|card|site$/i.test(key))
      out.push(decode(content));
  }
  // Text a reader is shown without it being a text node.
  for (const m of page.matchAll(/\s(?:alt|title|aria-label|placeholder)\s*=\s*"([^"]*)"/gi)) out.push(decode(m[1]));
  // The text nodes, <title> included.
  for (const t of page.replace(/<[^>]+>/g, '\n').split('\n')) {
    const s = decode(t).trim();
    if (s) out.push(s);
  }
  return out;
}

export function pagesOf(dir) {
  return fs
    .readdirSync(dir, { recursive: true, encoding: 'utf8' })
    .filter((rel) => rel.endsWith('.html'))
    .sort();
}

export function check(dir, names = loadNames()) {
  const re = matcher(names);
  const pages = pagesOf(dir);
  // A check over no pages passes everything: the export moved, not the story.
  if (!pages.length) throw new Error(`no pages under ${dir}`);
  const found = [];
  for (const rel of pages) {
    for (const text of readThrough(fs.readFileSync(path.join(dir, rel), 'utf8'))) {
      for (const m of text.matchAll(re)) {
        const at = m.index ?? 0;
        found.push([rel, m[0], text.slice(Math.max(0, at - 40), at + m[0].length + 40).replace(/\s+/g, ' ')]);
      }
    }
  }
  return { pages, found };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const dir = path.resolve(process.argv[2] ?? 'out');
  const { pages, found } = check(dir);
  if (found.length) {
    console.error(`[zen] FAIL: ${found.length} upstream name(s) outside data-upstream`);
    for (const [page, name, text] of found) console.error(`   ${page}: "${name}" in …${text}…`);
    process.exit(1);
  }
  console.log(`[zen] ${pages.length} pages read; no name from zen.upstream outside data-upstream`);
}
