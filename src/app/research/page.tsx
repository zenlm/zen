import { Metadata } from 'next';
import papers from '../../data/papers.json';

export const metadata: Metadata = {
  title: 'Zen Research Papers — 84 technical reports',
  description:
    'Technical reports and whitepapers across the Zen model family: architecture, training, safety, reasoning, and applications. Hosted at papers.zenlm.org.',
};

type Paper = { slug: string; title: string; theme: string; pdf: string };

const THEME_ORDER = ['Model Whitepapers', 'Research', 'Applications', 'Protocols'];

export default function ResearchPage() {
  const all = papers as Paper[];
  const byTheme = new Map<string, Paper[]>();
  for (const p of all) {
    if (!byTheme.has(p.theme)) byTheme.set(p.theme, []);
    byTheme.get(p.theme)!.push(p);
  }
  for (const list of byTheme.values()) list.sort((a, b) => a.title.localeCompare(b.title));

  return (
    <main>
      <section className="hero">
        <div className="container">
          <h1 className="hero-title">Research Papers</h1>
          <p className="hero-subtitle">{all.length} technical reports across the Zen family</p>
          <p className="hero-description">
            Architecture, training methodology, safety, reasoning, multimodality, and applications —
            the full archive is published at papers.zenlm.org.
          </p>
          <div className="hero-cta">
            <a href="https://papers.zenlm.org" className="btn btn-primary" target="_blank" rel="noopener noreferrer">
              Browse papers.zenlm.org
            </a>
            <a href="https://github.com/zenlm/papers" className="btn btn-secondary" target="_blank" rel="noopener noreferrer">
              GitHub
            </a>
          </div>
        </div>
      </section>

      <section className="catalog-section">
        <div className="container">
          <div className="catalog-groups" style={{ gridTemplateColumns: '1fr' }}>
            {THEME_ORDER.filter((t) => byTheme.has(t)).map((theme) => (
              <div className="cat-group" key={theme}>
                <div className="cat-group-head">
                  <h3>{theme}</h3>
                  <span className="cat-count">{byTheme.get(theme)!.length}</span>
                </div>
                <div className="paper-grid">
                  {byTheme.get(theme)!.map((p) => (
                    <a
                      className="paper-chip"
                      key={p.slug}
                      href={p.pdf}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      <span className="paper-chip-title">{p.title}</span>
                      <span className="paper-chip-action">PDF →</span>
                    </a>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}
