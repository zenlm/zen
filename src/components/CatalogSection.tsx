import catalog from '../data/catalog.json';

type Entry = {
  id: string;
  name: string;
  category: string;
  params?: string;
  context?: string;
  license?: string;
  pipeline?: string;
  paper_slug?: string;
  blurb?: string;
};

const CATEGORY_ORDER: { key: string; label: string }[] = [
  { key: 'chat', label: 'Chat & Reasoning' },
  { key: 'coder', label: 'Code' },
  { key: 'vision', label: 'Vision-Language' },
  { key: 'omni', label: 'Omni / Multimodal' },
  { key: 'embedding', label: 'Embeddings' },
  { key: 'reranker', label: 'Rerankers' },
  { key: 'guard', label: 'Safety' },
  { key: 'image', label: 'Image' },
  { key: 'video', label: 'Video' },
  { key: 'audio', label: 'Audio & Speech' },
  { key: '3d', label: '3D' },
  { key: 'agent', label: 'Agents' },
  { key: 'world', label: 'World Models' },
  { key: 'other', label: 'Specialty' },
];

const paperUrl = (slug?: string) =>
  slug ? `https://github.com/zenlm/papers/raw/main/pdfs/${slug}.pdf` : null;

function ModelRow({ m }: { m: Entry }) {
  const paper = paperUrl(m.paper_slug);
  return (
    <div className="cat-row">
      <div className="cat-row-main">
        <span className="cat-name">{m.name}</span>
        <span className="cat-meta">
          {[m.params, m.context, m.pipeline].filter(Boolean).join(' · ')}
        </span>
      </div>
      <div className="cat-links">
        <a className="cat-link" href={`https://huggingface.co/${m.id}`} target="_blank" rel="noopener noreferrer">
          Weights
        </a>
        {paper && (
          <a className="cat-link cat-link-ghost" href={paper} target="_blank" rel="noopener noreferrer">
            Paper
          </a>
        )}
      </div>
    </div>
  );
}

export default function CatalogSection() {
  const entries = catalog as Entry[];
  if (!entries.length) return null;

  const byCat = new Map<string, Entry[]>();
  for (const e of entries) {
    const k = CATEGORY_ORDER.some((c) => c.key === e.category) ? e.category : 'other';
    if (!byCat.has(k)) byCat.set(k, []);
    byCat.get(k)!.push(e);
  }
  for (const list of byCat.values()) list.sort((a, b) => a.name.localeCompare(b.name));

  return (
    <section id="catalog" className="catalog-section">
      <div className="container">
        <h2 className="section-title">The Complete Catalog</h2>
        <p className="section-subtitle">
          All {entries.length} open Zen models — every one linked to its weights on HuggingFace and its paper.
        </p>
        <div className="catalog-groups">
          {CATEGORY_ORDER.filter((c) => byCat.has(c.key)).map((c) => (
            <div className="cat-group" key={c.key}>
              <div className="cat-group-head">
                <h3>{c.label}</h3>
                <span className="cat-count">{byCat.get(c.key)!.length}</span>
              </div>
              <div className="cat-list">
                {byCat.get(c.key)!.map((m) => (
                  <ModelRow m={m} key={m.id} />
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
