import 'katex/dist/katex.min.css';
import Link from 'next/link';
import { Metadata } from 'next';
import { getAllPosts, getPost, renderPost } from '../../../lib/blog';

export function generateStaticParams() {
  return getAllPosts().map((p) => ({ slug: p.slug }));
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }): Promise<Metadata> {
  const { slug } = await params;
  const post = getPost(slug);
  if (!post) return { title: 'Zen Blog' };
  return { title: `${post.title} — Zen Blog`, description: post.description };
}

function fmtDate(d: string) {
  if (!d) return '';
  const dt = new Date(d);
  return isNaN(dt.getTime()) ? d : dt.toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });
}

export default async function BlogPost({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const post = getPost(slug);
  if (!post) {
    return (
      <main className="container" style={{ padding: '6rem 2rem' }}>
        <p>Post not found. <Link href="/blog">Back to blog</Link></p>
      </main>
    );
  }

  let html: string;
  try {
    html = await renderPost(post.content);
  } catch {
    html = `<p class="blog-fallback">This post couldn’t be rendered inline. <a href="https://github.com/zenlm/zen-blog/blob/main/content/${post.slug}.mdx" target="_blank" rel="noopener noreferrer">Read the source on GitHub →</a></p>`;
  }

  return (
    <main>
      <article className="blog-article">
        <Link className="blog-back" href="/blog">← Blog</Link>
        <div className="blog-post-meta">
          {fmtDate(post.date)} {post.date && '·'} {post.readMins} min read
        </div>
        <h1 className="blog-post-title">{post.title}</h1>
        {post.description && <p className="blog-post-lede">{post.description}</p>}
        <div className="blog-prose" dangerouslySetInnerHTML={{ __html: html }} />
      </article>
    </main>
  );
}
