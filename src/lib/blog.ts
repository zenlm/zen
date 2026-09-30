import fs from 'node:fs';
import path from 'node:path';
import matter from 'gray-matter';
import readingTime from 'reading-time';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import remarkRehype from 'remark-rehype';
import rehypeRaw from 'rehype-raw';
import rehypeKatex from 'rehype-katex';
import rehypeSlug from 'rehype-slug';
import rehypeStringify from 'rehype-stringify';

// The posts are the zen-blog repository's content/, copied into content/blog.
// They were written for fumadocs-mdx, so the four components they use are
// lowered to HTML here before markdown sees them.

const DIR = path.join(process.cwd(), 'content/blog');

export type Post = {
  slug: string;
  title: string;
  date: string;
  description: string;
  tags: string[];
  readMins: number;
  content: string;
  /** The post's subject is another lab's release; its page and card carry data-upstream. */
  upstream: boolean;
};

function read(file: string): Post {
  const { data, content } = matter(fs.readFileSync(path.join(DIR, file), 'utf8'));
  return {
    slug: file.replace(/\.mdx?$/, ''),
    title: String(data.title ?? ''),
    date: String(data.date ?? ''),
    description: String(data.description ?? ''),
    tags: Array.isArray(data.tags) ? data.tags.map(String) : [],
    readMins: Math.max(1, Math.round(readingTime(content).minutes)),
    content,
    upstream: data.upstream === true,
  };
}

export function getAllPosts(): Post[] {
  return fs
    .readdirSync(DIR)
    .filter((f) => /\.mdx?$/.test(f))
    .map(read)
    .sort((a, b) => (a.date < b.date ? 1 : a.date > b.date ? -1 : a.slug.localeCompare(b.slug)));
}

export function getPost(slug: string): Post | undefined {
  const file = [`${slug}.mdx`, `${slug}.md`].find((f) => fs.existsSync(path.join(DIR, f)));
  return file ? read(file) : undefined;
}

const esc = (s: string) => s.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');

function attrs(src: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const m of src.matchAll(/([\w-]+)(?:=(?:"([^"]*)"|'([^']*)'|\{([^}]*)\}))?/g)) {
    out[m[1]] = m[2] ?? m[3] ?? m[4] ?? 'true';
  }
  return out;
}

/** Figure, LinkButton, Video and Fullwidth as the HTML they stand for. */
function lower(md: string): string {
  return md
    .replace(/\{\/\*[\s\S]*?\*\/\}/g, '')
    .replace(/<\/?Fullwidth>/g, '')
    .replace(/<Figure\b([^>]*?)\/>/g, (_, a) => {
      const p = attrs(a);
      const src = (p.src ?? '').replace(/#center$/, '');
      const alt = p.alt ?? p.caption ?? '';
      const cap = p.caption ? `<figcaption>${esc(p.caption)}</figcaption>` : '';
      return `<figure><img src="${esc(src)}" alt="${esc(alt)}" loading="lazy">${cap}</figure>`;
    })
    .replace(/<LinkButton\b([^>]*?)\/>/g, (_, a) => {
      const p = attrs(a);
      return `<a href="${esc(p.href ?? '#')}">${esc(p.label ?? p.href ?? '')}</a>`;
    })
    .replace(/<Video\b([^>]*?)\/>/g, (_, a) => {
      const p = attrs(a);
      return `<video src="${esc(p.src ?? '')}" controls playsinline preload="metadata" style="width:100%"></video>`;
    });
}

export async function renderPost(content: string): Promise<string> {
  const file = await unified()
    .use(remarkParse)
    .use(remarkGfm)
    .use(remarkMath)
    .use(remarkRehype, { allowDangerousHtml: true })
    .use(rehypeRaw)
    .use(rehypeKatex)
    .use(rehypeSlug)
    .use(rehypeStringify)
    .process(lower(content));
  return String(file);
}
