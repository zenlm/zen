import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Zen LM Research Papers - Hanzo AI',
  description: 'Collection of technical papers and research publications for the Zen Language Model family',
};

export default function ResearchPage() {
  return (
    <main style={{ minHeight: '100vh', background: '#0a0a0a', color: '#e5e5e5', padding: '2rem 0' }}>
      <div style={{ maxWidth: '1400px', margin: '0 auto', padding: '0 2rem' }}>
        {/* Hero */}
        <div style={{ textAlign: 'center', padding: '5rem 2rem' }}>
          <h1 style={{ fontSize: '3rem', marginBottom: '1rem', background: 'linear-gradient(135deg, #fff, #a3a3a3)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
            Zen Research Papers
          </h1>
          <p style={{ fontSize: '1.25rem', color: '#a3a3a3', maxWidth: '700px', margin: '0 auto' }}>
            Technical documentation and research publications for the Zen Language Model family, covering architecture, training methodology, and performance analysis.
          </p>
        </div>

        {/* Stats */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '2rem', padding: '3rem 2rem', background: '#1a1a1a', borderRadius: '12px', marginBottom: '3rem' }}>
          <Stat value="15" label="Research Papers" />
          <Stat value="5" label="Categories" />
          <Stat value="0.6B-1T+" label="Model Range" />
          <Stat value="2025" label="Latest Publication" />
        </div>

        {/* Papers */}
        <PaperSection
          title="Technical Report"
          icon="📊"
          papers={[
            { name: 'Zen Technical Report', size: '181KB', pages: '7 pages', description: 'Comprehensive technical overview of the Zen Language Model family, including architecture details, training methodology, and performance benchmarks across all model variants.', pdf: '/papers/zen-technical-report.pdf' }
          ]}
        />

        <PaperSection
          title="Core Language Models"
          icon="🧠"
          papers={[
            { name: 'Zen Pro', size: '109KB', label: 'Professional Model', description: 'Technical documentation for Zen Pro, our advanced professional language model optimized for complex reasoning and professional applications.', pdf: '/papers/zen-pro.pdf' },
            { name: 'Zen Coder', size: '109KB', label: 'Code Generation', description: 'Architecture and training methodology for Zen Coder, specialized in code generation, debugging, and software engineering tasks.', pdf: '/papers/zen-coder.pdf' },
            { name: 'Zen Agent', size: '127KB', label: 'Agent Systems', description: 'Research on autonomous agent capabilities, tool use, and multi-step reasoning in the Zen Agent model variant.', pdf: '/papers/zen-agent.pdf' },
            { name: 'Zen Guard', size: '109KB', label: 'Safety Model', description: 'Safety alignment and content moderation capabilities of Zen Guard, our specialized safety and compliance model.', pdf: '/papers/zen-guard.pdf' }
          ]}
        />

        <PaperSection
          title="Multimodal Models"
          icon="🎨"
          papers={[
            { name: 'Zen Omni', size: '109KB', label: 'Multimodal', description: 'Unified multimodal architecture supporting text, vision, audio, and video understanding in a single model.', pdf: '/papers/zen-omni.pdf' },
            { name: 'Zen Designer', size: '109KB', label: 'Design & UI', description: 'Visual design generation and UI/UX capabilities, trained on design patterns and user interface principles.', pdf: '/papers/zen-designer.pdf' },
            { name: 'Zen Artist', size: '109KB', label: 'Image Generation', description: 'Advanced image generation and artistic style transfer capabilities using diffusion and transformer architectures.', pdf: '/papers/zen-artist.pdf' }
          ]}
        />
      </div>
    </main>
  );
}

function Stat({ value, label }: { value: string; label: string }) {
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{ fontSize: '2.5rem', fontWeight: 700, background: 'linear-gradient(135deg, #6366f1, #22c55e)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
        {value}
      </div>
      <div style={{ color: '#a3a3a3', fontSize: '0.9rem', marginTop: '0.5rem' }}>
        {label}
      </div>
    </div>
  );
}

interface Paper {
  name: string;
  size: string;
  label?: string;
  pages?: string;
  description: string;
  pdf: string;
}

function PaperSection({ title, icon, papers }: { title: string; icon: string; papers: Paper[] }) {
  return (
    <div style={{ marginBottom: '4rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '2rem', paddingBottom: '1rem', borderBottom: '1px solid #333' }}>
        <div style={{ width: '40px', height: '40px', background: 'linear-gradient(135deg, #6366f1, #4f46e5)', borderRadius: '10px', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.5rem' }}>
          {icon}
        </div>
        <h2 style={{ fontSize: '1.8rem', color: '#e5e5e5' }}>{title}</h2>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(350px, 1fr))', gap: '2rem' }}>
        {papers.map((paper, idx) => (
          <PaperCard key={idx} {...paper} />
        ))}
      </div>
    </div>
  );
}

function PaperCard({ name, size, label, pages, description, pdf }: Paper) {
  return (
    <div style={{ background: '#1a1a1a', border: '1px solid #333', borderRadius: '12px', padding: '2rem', transition: 'all 0.3s ease' }}>
      <h3 style={{ fontSize: '1.25rem', marginBottom: '1rem', color: '#e5e5e5' }}>{name}</h3>
      <div style={{ display: 'flex', gap: '1rem', marginBottom: '1rem', fontSize: '0.875rem', color: '#94a3b8' }}>
        <span>📄 {size}</span>
        {pages && <span>{pages}</span>}
        {label && <span>{label}</span>}
      </div>
      <p style={{ color: '#a3a3a3', marginBottom: '1.5rem', lineHeight: '1.6' }}>
        {description}
      </p>
      <a
        href={pdf}
        target="_blank"
        rel="noopener noreferrer"
        style={{
          display: 'inline-block',
          padding: '0.75rem 1.5rem',
          background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
          color: 'white',
          borderRadius: '0.5rem',
          textDecoration: 'none',
          fontWeight: 600,
          transition: 'all 0.3s ease'
        }}
      >
        Download PDF
      </a>
    </div>
  );
}
