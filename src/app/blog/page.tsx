import Link from 'next/link';
import { Metadata } from 'next';
import { getAllPosts } from '../../lib/blog';

export const metadata: Metadata = {
  title: 'Zen Blog — research notes & releases',
  description: 'Architecture deep-dives, training research, and release notes from the Zen LM team.',
};

function fmtDate(d: string) {
  if (!d) return '';
  const dt = new Date(d);
  return isNaN(dt.getTime()) ? d : dt.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' });
}

export default function BlogIndex() {
  const posts = getAllPosts();
  return (
    <main>
      <section className="hero">
        <div className="container">
          <h1 className="hero-title">Blog</h1>
          <p className="hero-subtitle">Research notes, deep dives, and releases</p>
          <p className="hero-description">{posts.length} posts on Zen architecture, training, and the open-model ecosystem.</p>
        </div>
      </section>

      <section className="catalog-section">
        <div className="container">
          <div className="blog-list">
            {posts.map((p) => (
              <Link className="blog-card" href={`/blog/${p.slug}`} key={p.slug}>
                <div className="blog-card-meta">
                  {fmtDate(p.date)} {p.date && '·'} {p.readMins} min read
                </div>
                <h2 className="blog-card-title">{p.title}</h2>
                {p.description && <p className="blog-card-desc">{p.description}</p>}
                {p.tags.length > 0 && (
                  <div className="blog-card-tags">
                    {p.tags.slice(0, 3).map((t) => (
                      <span className="blog-tag" key={t}>{t}</span>
                    ))}
                  </div>
                )}
              </Link>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}
