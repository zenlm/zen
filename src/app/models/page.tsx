import Link from 'next/link';
import { Metadata } from 'next';
import CatalogSection from '../../components/CatalogSection';

export const metadata: Metadata = {
  title: 'Zen Models - 95 open foundation models from 0.6B to 1T+',
  description: 'The complete Zen catalog across Zen3, Zen4, and Zen5. Chat, code, vision, audio, embeddings, rerankers, image generation, and safety.',
};

interface ModelCardProps {
  name: string;
  status: string;
  specs: Array<{ label: string; value: string }>;
  description: string;
  formats?: string[];
  flagship?: boolean;
  frontier?: boolean;
  hfLink?: string;
  githubLink?: string;
  docsLink?: string;
}

function ModelCard({ name, status, specs, description, formats, flagship, frontier, hfLink, githubLink, docsLink }: ModelCardProps) {
  const badgeClass =
    status === 'Coming Soon' ? 'badge badge-planned' :
    status === 'Cloud Only' ? 'badge badge-cloud' :
    'badge badge-complete';

  return (
    <div className={`model-card complete${flagship ? ' flagship' : ''}${frontier ? ' frontier' : ''}`}>
      <div className="model-header">
        <h3>{name}</h3>
        <span className={badgeClass}>{status}</span>
      </div>
      <div className="model-specs">
        {specs.map((spec, idx) => (
          <div key={idx} className="spec">
            <span className="spec-label">{spec.label}</span>
            <span className="spec-value">{spec.value}</span>
          </div>
        ))}
      </div>
      <p className="model-description">{description}</p>
      {formats && formats.length > 0 && (
        <div className="model-formats">
          {formats.map((format, idx) => (
            <span key={idx} className="format-tag">{format}</span>
          ))}
        </div>
      )}
      <div className="model-actions">
        {hfLink && (
          <a href={hfLink} className="btn btn-sm btn-primary" target="_blank" rel="noopener noreferrer">
            HuggingFace
          </a>
        )}
        {githubLink && (
          <a href={githubLink} className="btn btn-sm btn-secondary" target="_blank" rel="noopener noreferrer">
            GitHub
          </a>
        )}
        {docsLink && (
          <a href={docsLink} className="btn btn-sm btn-secondary" target="_blank" rel="noopener noreferrer">
            Docs
          </a>
        )}
      </div>
    </div>
  );
}

export default function ModelsPage() {
  return (
    <main>
      <section className="hero">
        <div className="container">
          <h2 className="hero-title">Zen Model Catalog</h2>
          <p className="hero-subtitle">95 open foundation models across Zen3, Zen4, and Zen5</p>
          <p className="hero-description">
            Chat, code, vision-language, web agentic, embeddings, rerankers, image generation, streaming ASR,
            and TTS. From edge-class Zen5 Nano 0.8B to the Zen5 Max frontier MoE. 8K - 1M context. OpenAI- and
            Anthropic-compatible API.
          </p>
          <div className="hero-cta">
            <a href="https://huggingface.co/zenlm" className="btn btn-primary" target="_blank" rel="noopener noreferrer">Browse on HuggingFace</a>
            <a href="https://api.hanzo.ai" className="btn btn-secondary" target="_blank" rel="noopener noreferrer">Get API Key</a>
            <Link href="/datasets" className="btn btn-outline">Training Data</Link>
          </div>
        </div>
      </section>

      {/* Zen5 Chat */}
      <section id="zen5" className="models-section featured-section">
        <div className="container">
          <h2 className="section-title">Zen 5 - Next-Generation Agentic</h2>
          <p className="section-subtitle">Native chain-of-thought, large-scale RL on 200K+ environments, OpenAI + Anthropic API.</p>

          <div className="models-grid">
            <ModelCard
              name="Zen5 Nano 0.8B"
              status="Available"
              specs={[
                { label: 'Parameters', value: '0.9B dense' },
                { label: 'Modalities', value: 'Multimodal' },
                { label: 'Context', value: '32K' },
                { label: 'Target', value: 'Edge / on-device' },
              ]}
              description="Edge / on-device tier. Raspberry Pi, phone, browser WASM class."
              formats={['SafeTensors', 'GGUF', 'MLX']}
              hfLink="https://huggingface.co/zenlm"
            />
            <ModelCard
              name="Zen5 Nano 2B"
              status="Available"
              specs={[
                { label: 'Parameters', value: '2B dense' },
                { label: 'Modalities', value: 'Multimodal' },
                { label: 'Context', value: '32K' },
                { label: 'Target', value: '8 GB RAM laptop' },
              ]}
              description="Low-end multimodal dense at 2B parameters. Low iGPU / 8 GB RAM laptop class."
              formats={['SafeTensors', 'GGUF', 'MLX']}
              hfLink="https://huggingface.co/zenlm"
            />
            <ModelCard
              name="Zen5 Nano 4B"
              status="Available"
              specs={[
                { label: 'Parameters', value: '5B dense' },
                { label: 'Modalities', value: 'Multimodal' },
                { label: 'Context', value: '32K' },
                { label: 'Target', value: '16 GB / mobile NPU' },
              ]}
              description="Mid multimodal dense at 5B. 16 GB RAM laptop / mobile NPU class."
              formats={['SafeTensors', 'GGUF', 'MLX']}
              hfLink="https://huggingface.co/zenlm"
            />
            <ModelCard
              name="Zen5 Nano 9B"
              status="Available"
              specs={[
                { label: 'Parameters', value: '10B dense' },
                { label: 'Modalities', value: 'Multimodal' },
                { label: 'Context', value: '32K' },
                { label: 'Target', value: '24 GB+ / consumer GPU' },
              ]}
              description="Upper-nano multimodal dense. 24 GB+ unified RAM / consumer GPU class."
              formats={['SafeTensors', 'GGUF', 'MLX']}
              hfLink="https://huggingface.co/zenlm"
            />
            <ModelCard
              name="Zen5 Flash"
              status="Available"
              specs={[
                { label: 'Parameters', value: '4B dense' },
                { label: 'Architecture', value: 'Zen dense' },
                { label: 'Context', value: '32K' },
                { label: 'TTFT', value: 'Sub-100ms' },
              ]}
              description="Smallest and cheapest text-only Zen5 chat tier. For high-volume routing and simple agent loops."
              formats={['SafeTensors']}
              hfLink="https://huggingface.co/zenlm"
            />
            <ModelCard
              name="Zen5 Mini"
              status="Available"
              specs={[
                { label: 'Parameters', value: '230B (10B active)' },
                { label: 'Architecture', value: 'Zen agentic MoE' },
                { label: 'Context', value: '192K' },
                { label: 'SWE-Bench', value: '80.2%' },
              ]}
              description="Frontier agentic at the lowest $/token in the family. 230B MoE / 10B active, trained on 200K+ real environments via large-scale RL."
              formats={['SafeTensors']}
              hfLink="https://huggingface.co/zenlm"
            />
            <ModelCard
              name="Zen5"
              status="Available"
              flagship={true}
              specs={[
                { label: 'Parameters', value: '35B (3B active)' },
                { label: 'Architecture', value: 'Zen frontier MoE' },
                { label: 'Context', value: '256K' },
                { label: 'License', value: 'Apache-2.0' },
              ]}
              description="Canonical Zen5 default. 35B frontier MoE (3B active per token). The everyday Zen5 chat model. OpenAI + Anthropic API."
              formats={['SafeTensors', 'GGUF', 'MLX']}
              hfLink="https://huggingface.co/zenlm/zen-5"
            />
            <ModelCard
              name="Zen5 Coder"
              status="Available"
              specs={[
                { label: 'Parameters', value: '80B sparse MoE' },
                { label: 'Specialization', value: 'Repo-scale code' },
                { label: 'Context', value: '256K' },
                { label: 'Focus', value: 'Agentic / tool-use' },
              ]}
              description="Code-specialized Zen5 tier. Sparse MoE tuned for repo-scale code understanding, agentic refactoring, and tool-use coding loops."
              formats={['SafeTensors', 'GGUF']}
              hfLink="https://huggingface.co/zenlm/zen-5-coder"
            />
            <ModelCard
              name="Zen5 Pro"
              status="Available"
              flagship={true}
              specs={[
                { label: 'Parameters', value: '284B (37B active)' },
                { label: 'Architecture', value: 'Zen Flash MoE' },
                { label: 'Context', value: '1M' },
                { label: 'Quant', value: 'IQ2_XXS-imatrix (81 GB)' },
              ]}
              description="Zen Flash IQ2_XXS-imatrix weights (81 GB GGUF). 284B total / 37B active. Fits a single 128 GB Apple Silicon / DGX Spark / H100 80 GB."
              formats={['GGUF']}
              hfLink="https://huggingface.co/zenlm/zen-5-pro-gguf"
            />
            <ModelCard
              name="Zen5 Max"
              status="Cloud Only"
              frontier={true}
              specs={[
                { label: 'Parameters', value: 'Full Zen Pro (432 GB)' },
                { label: 'Architecture', value: 'Zen Pro MoE' },
                { label: 'Context', value: '1M+' },
                { label: 'Hardware', value: '512 GB+ unified / 8x H100' },
              ]}
              description="Top quality tier in the family. Requires Mac Studio M3 Ultra 512 GB or 8x H100/H200 class GPU pool."
              formats={['SafeTensors']}
              hfLink="https://huggingface.co/zenlm"
            />
          </div>
        </div>
      </section>

      {/* Zen5 Embedding */}
      <section id="zen5-embedding" className="models-section">
        <div className="container">
          <h2 className="section-title">Zen 5 Embedding</h2>
          <p className="section-subtitle">Three-SKU embedding lineup served on <code>/v1/embeddings</code>.</p>
          <div className="models-grid">
            <ModelCard
              name="Zen5 Embedding 0.6B"
              status="Available"
              specs={[
                { label: 'Parameters', value: '0.6B' },
                { label: 'Dimensions', value: '1024' },
                { label: 'Context', value: '32K' },
                { label: 'Endpoint', value: '/v1/embeddings' },
              ]}
              description="Lightweight embedding model for high-throughput RAG and search."
            />
            <ModelCard
              name="Zen5 Embedding 4B"
              status="Available"
              specs={[
                { label: 'Parameters', value: '4B' },
                { label: 'Dimensions', value: '2560' },
                { label: 'Context', value: '32K' },
                { label: 'Endpoint', value: '/v1/embeddings' },
              ]}
              description="Balanced embedding model for production RAG."
            />
            <ModelCard
              name="Zen5 Embedding 8B"
              status="Available"
              specs={[
                { label: 'Parameters', value: '8B' },
                { label: 'Dimensions', value: '4096' },
                { label: 'Context', value: '32K' },
                { label: 'Endpoint', value: '/v1/embeddings' },
              ]}
              description="High-quality embeddings for production RAG, semantic search, and classification."
            />
          </div>
        </div>
      </section>

      {/* Zen4 Chat */}
      <section id="zen4" className="models-section">
        <div className="container">
          <h2 className="section-title">Zen 4 - Production Chat</h2>
          <p className="section-subtitle">The everyday Zen production line: MoE flagships, thinking models, and long-context.</p>
          <div className="models-grid">
            <ModelCard
              name="Zen4 Mini"
              status="Available"
              specs={[
                { label: 'Architecture', value: 'Dense' },
                { label: 'Context', value: '128K' },
                { label: 'Tier', value: 'Starter' },
                { label: 'Use', value: 'Free tier / edge' },
              ]}
              description="Ultra-fast lightweight model optimized for speed and cost efficiency. Ideal for free tier."
              hfLink="https://huggingface.co/zenlm"
            />
            <ModelCard
              name="Zen4"
              status="Available"
              specs={[
                { label: 'Parameters', value: '744B (40B active)' },
                { label: 'Architecture', value: 'MoE' },
                { label: 'Context', value: '202K' },
                { label: 'Tier', value: 'Ultra Max' },
              ]}
              description="Flagship MoE model for complex reasoning and multi-domain tasks."
              hfLink="https://huggingface.co/zenlm"
            />
            <ModelCard
              name="Zen4 Pro"
              status="Available"
              specs={[
                { label: 'Parameters', value: '80B (3B active)' },
                { label: 'Architecture', value: 'MoE' },
                { label: 'Context', value: '131K' },
                { label: 'Tier', value: 'Ultra' },
              ]}
              description="Efficient MoE model for demanding workloads with strong reasoning at production-grade cost."
              hfLink="https://huggingface.co/zenlm"
            />
            <ModelCard
              name="Zen4 Thinking"
              status="Available"
              specs={[
                { label: 'Parameters', value: '80B (3B active)' },
                { label: 'Architecture', value: 'MoE + CoT' },
                { label: 'Context', value: '131K' },
                { label: 'Tier', value: 'Pro Max' },
              ]}
              description="Dedicated reasoning model with explicit chain-of-thought capabilities."
              hfLink="https://huggingface.co/zenlm"
            />
            <ModelCard
              name="Zen4 Ultra"
              status="Available"
              specs={[
                { label: 'Parameters', value: '744B (40B active)' },
                { label: 'Architecture', value: 'MoE + CoT' },
                { label: 'Context', value: '262K' },
                { label: 'Tier', value: 'Ultra Max' },
              ]}
              description="Maximum reasoning capability with extended chain-of-thought on MoE architecture."
              hfLink="https://huggingface.co/zenlm"
            />
            <ModelCard
              name="Zen4.1"
              status="Available"
              specs={[
                { label: 'Architecture', value: 'Dense' },
                { label: 'Context', value: '1M' },
                { label: 'Tier', value: 'Ultra' },
                { label: 'Focus', value: 'Long-document / agentic' },
              ]}
              description="High-performance 1M context model for long-document analysis, large codebase reasoning, and agentic workflows."
              hfLink="https://huggingface.co/zenlm"
            />
            <ModelCard
              name="Zen4 Max"
              status="Available"
              flagship={true}
              specs={[
                { label: 'Architecture', value: 'Dense' },
                { label: 'Context', value: '1M' },
                { label: 'Tier', value: 'Ultra Max' },
                { label: 'Focus', value: 'Frontier intelligence' },
              ]}
              description="Most capable model for complex reasoning, analysis, and agentic tasks. 1M token context window."
              hfLink="https://huggingface.co/zenlm"
            />
          </div>
        </div>
      </section>

      {/* Zen4 Coder */}
      <section id="zen4-coder" className="models-section">
        <div className="container">
          <h2 className="section-title">Zen 4 Coder</h2>
          <p className="section-subtitle">Code-specialized MoE and dense models tuned for generation, review, debugging, and agentic programming.</p>
          <div className="models-grid">
            <ModelCard
              name="Zen4 Coder Flash"
              status="Available"
              specs={[
                { label: 'Parameters', value: '30B (3B active)' },
                { label: 'Architecture', value: 'MoE' },
                { label: 'Context', value: '262K' },
                { label: 'Tier', value: 'Pro Max' },
              ]}
              description="Lightweight code model optimized for speed and inline completions."
              hfLink="https://huggingface.co/zenlm"
            />
            <ModelCard
              name="Zen4 Coder"
              status="Available"
              flagship={true}
              specs={[
                { label: 'Parameters', value: '480B (35B active)' },
                { label: 'Architecture', value: 'MoE' },
                { label: 'Context', value: '163K' },
                { label: 'Tier', value: 'Ultra' },
              ]}
              description="Code-specialized MoE model for generation, review, debugging, and agentic programming."
              hfLink="https://huggingface.co/zenlm"
            />
            <ModelCard
              name="Zen4 Coder Pro"
              status="Available"
              frontier={true}
              specs={[
                { label: 'Parameters', value: '480B' },
                { label: 'Architecture', value: 'Dense BF16' },
                { label: 'Context', value: '131K' },
                { label: 'Tier', value: 'Ultra Max' },
              ]}
              description="Full-precision BF16 code model for maximum accuracy on complex codebases."
              hfLink="https://huggingface.co/zenlm"
            />
          </div>
        </div>
      </section>

      {/* Zen3 Multimodal */}
      <section id="zen3" className="models-section">
        <div className="container">
          <h2 className="section-title">Zen 3 - Multimodal &amp; Specialty</h2>
          <p className="section-subtitle">Vision, audio, web agentic, safety, and edge.</p>
          <div className="models-grid">
            <ModelCard
              name="Zen3 Omni"
              status="Available"
              specs={[
                { label: 'Parameters', value: '~200B' },
                { label: 'Modalities', value: 'Text + Vision + Audio' },
                { label: 'Context', value: '202K' },
                { label: 'Architecture', value: 'Dense Multimodal' },
              ]}
              description="Hypermodal model supporting text, vision, audio, and structured output."
              hfLink="https://huggingface.co/zenlm"
            />
            <ModelCard
              name="Zen3 VL"
              status="Available"
              specs={[
                { label: 'Parameters', value: '30B (3B active)' },
                { label: 'Modalities', value: 'Vision + Language' },
                { label: 'Context', value: '262K' },
                { label: 'Sizes', value: '2B / 8B / 32B / 235B-A22B' },
              ]}
              description="Vision-language model for image understanding and visual reasoning. Default 30B-A3B MoE plus 2B, 8B, 32B, and frontier 235B-A22B variants."
              hfLink="https://huggingface.co/zenlm"
            />
            <ModelCard
              name="Zen3 Web"
              status="Available"
              specs={[
                { label: 'Sizes', value: '8B / 14B / 32B' },
                { label: 'Architecture', value: 'Zen Web dense' },
                { label: 'Context', value: '32K' },
                { label: 'Focus', value: 'Browser agentic' },
              ]}
              description="Web-agentic models for browser automation, scraping, and on-page reasoning. Three tiers from edge to top-end."
              hfLink="https://huggingface.co/zenlm"
            />
            <ModelCard
              name="Zen3 Nano"
              status="Available"
              specs={[
                { label: 'Parameters', value: '8B dense' },
                { label: 'Context', value: '128K' },
                { label: 'Tier', value: 'Starter' },
                { label: 'Use', value: 'Edge / free tier' },
              ]}
              description="Ultra-lightweight model for edge deployment and low-latency tasks. Available on free tier."
              hfLink="https://huggingface.co/zenlm/zen-nano-0.6b"
            />
            <ModelCard
              name="Zen3 Guard"
              status="Available"
              specs={[
                { label: 'Parameters', value: '4B dense' },
                { label: 'Context', value: '65K' },
                { label: 'Categories', value: '9 safety' },
                { label: 'Languages', value: '119' },
              ]}
              description="Content safety classifier for moderation and guardrails. 9 safety categories, 119 languages."
              hfLink="https://huggingface.co/zenlm"
            />
          </div>
        </div>
      </section>

      {/* Zen3 Embedding & Reranker */}
      <section id="zen3-embedding" className="models-section">
        <div className="container">
          <h2 className="section-title">Zen 3 Embedding &amp; Reranker</h2>
          <p className="section-subtitle">Text and multimodal embeddings plus rerankers for retrieval pipelines.</p>
          <div className="models-grid">
            <ModelCard
              name="Zen3 Embedding"
              status="Available"
              specs={[
                { label: 'Sizes', value: 'small / medium / default' },
                { label: 'Parameters', value: '0.6B / 4B / N/A' },
                { label: 'Context', value: '8K - 40K' },
                { label: 'Endpoint', value: '/v1/embeddings' },
              ]}
              description="High-quality text embeddings for RAG, search, and classification. OpenAI-compatible endpoint available."
            />
            <ModelCard
              name="Zen3 Reranker"
              status="Available"
              specs={[
                { label: 'Sizes', value: 'small / medium / default' },
                { label: 'Parameters', value: '0.6B / 4B / 8B' },
                { label: 'Context', value: '40K' },
                { label: 'Endpoint', value: '/v1/rerank' },
              ]}
              description="High-quality rerankers for improving retrieval accuracy in RAG pipelines."
            />
            <ModelCard
              name="Zen3 VL Embedding"
              status="Available"
              specs={[
                { label: 'Sizes', value: '2B / 8B' },
                { label: 'Modalities', value: 'Text + Image' },
                { label: 'Context', value: '32K' },
                { label: 'Endpoint', value: '/v1/embeddings' },
              ]}
              description="Multimodal embeddings (text + image) for vision-aware retrieval and semantic search."
            />
            <ModelCard
              name="Zen3 VL Reranker"
              status="Available"
              specs={[
                { label: 'Sizes', value: '2B / 8B' },
                { label: 'Modalities', value: 'Text + Image' },
                { label: 'Context', value: '32K' },
                { label: 'Endpoint', value: '/v1/rerank' },
              ]}
              description="Vision-language rerankers for multimodal RAG. Reranks (query, image+text) pairs."
            />
          </div>
        </div>
      </section>

      {/* Zen3 Image */}
      <section id="zen3-image" className="models-section">
        <div className="container">
          <h2 className="section-title">Zen 3 Image Generation</h2>
          <p className="section-subtitle">Eight image-generation SKUs from fast diffusion to broadcast-quality.</p>
          <div className="models-grid">
            <ModelCard
              name="Zen3 Image"
              status="Available"
              specs={[
                { label: 'Type', value: 'Text-to-image + edit' },
                { label: 'Tier', value: 'Pro Max' },
                { label: 'Pricing', value: '$0.04 / image' },
                { label: 'Endpoint', value: '/v1/images/generations' },
              ]}
              description="Best general-purpose image generation."
            />
            <ModelCard
              name="Zen3 Image Max"
              status="Available"
              flagship={true}
              specs={[
                { label: 'Type', value: 'Text-to-image' },
                { label: 'Tier', value: 'Ultra Max' },
                { label: 'Pricing', value: '$0.08 / image' },
                { label: 'Quality', value: 'Maximum' },
              ]}
              description="Maximum quality image generation for professional creative work."
            />
            <ModelCard
              name="Zen3 Image Fast"
              status="Available"
              specs={[
                { label: 'Type', value: 'Text-to-image' },
                { label: 'Tier', value: 'Pro' },
                { label: 'Pricing', value: '$0.00035 / step' },
                { label: 'Latency', value: 'Ultra-fast' },
              ]}
              description="Fastest image model for real-time generation."
            />
            <ModelCard
              name="Zen3 Image SDXL / Dev / Playground / SSD / JP"
              status="Available"
              specs={[
                { label: 'Variants', value: '5 specialty models' },
                { label: 'Resolution', value: 'up to 1024px' },
                { label: 'Pricing', value: 'from $0.00013/step' },
                { label: 'Endpoint', value: '/v1/images/generations' },
              ]}
              description="Specialized image models: SDXL (1024px), Dev (experimentation), Playground (aesthetic), SSD (fastest diffusion), JP (Japanese-specialized)."
            />
          </div>
        </div>
      </section>

      {/* Zen3 Audio + TTS */}
      <section id="zen3-audio" className="models-section">
        <div className="container">
          <h2 className="section-title">Zen 3 Audio &amp; Speech</h2>
          <p className="section-subtitle">Speech-to-text, text-to-speech, streaming ASR, voice cloning, and forced alignment.</p>
          <div className="models-grid">
            <ModelCard
              name="Zen3 Audio (STT)"
              status="Available"
              specs={[
                { label: 'Variants', value: 'audio / audio-fast' },
                { label: 'Languages', value: '100+' },
                { label: 'Pricing', value: 'from $0.0012 / min' },
                { label: 'Endpoint', value: '/v1/audio/transcriptions' },
              ]}
              description="High-quality and fast speech-to-text transcription. 100+ languages."
            />
            <ModelCard
              name="Zen3 ASR (Streaming)"
              status="Available"
              specs={[
                { label: 'Variants', value: 'asr / asr-0.6B / asr-aligner / asr-v1' },
                { label: 'Latency', value: 'Sub-200ms - Sub-500ms' },
                { label: 'Pricing', value: 'from $0.002 / min' },
                { label: 'Endpoint', value: '/v1/audio/transcriptions' },
              ]}
              description="Real-time streaming ASR for voice agents. Edge variant (0.6B) for on-device, aligner for word-level timestamps."
            />
            <ModelCard
              name="Zen3 TTS"
              status="Available"
              specs={[
                { label: 'Variants', value: 'tts / tts-hd / tts-fast / tts-0.6B' },
                { label: 'Voices', value: '40+ across 8 languages' },
                { label: 'Pricing', value: 'from $2 / 1M chars' },
                { label: 'Endpoint', value: '/v1/audio/speech' },
              ]}
              description="High-quality text-to-speech with natural prosody. Four tiers from edge to broadcast-grade HD."
            />
            <ModelCard
              name="Zen3 TTS Voice Design &amp; Custom Voice"
              status="Available"
              flagship={true}
              specs={[
                { label: 'Variants', value: 'voice-design / custom-voice' },
                { label: 'Features', value: 'Prompt-driven + few-shot clone' },
                { label: 'Pricing', value: '$8 - $10 / 1M chars' },
                { label: 'Endpoint', value: '/v1/audio/speech' },
              ]}
              description="Premium TTS with prompt-driven voice design and few-shot voice cloning from a short audio sample."
            />
          </div>
        </div>
      </section>

      {/* Comparison Table */}
      <section id="comparison" className="featured-section">
        <div className="container">
          <h2 className="section-title">Full Zen Catalog Summary</h2>
          <p className="section-subtitle">Live catalog from the Zen API. Pricing fetched at runtime.</p>
          <div className="model-lineup">
            <table className="models-table">
              <thead>
                <tr>
                  <th>Generation</th>
                  <th>Family</th>
                  <th>SKUs</th>
                  <th>Endpoint(s)</th>
                </tr>
              </thead>
              <tbody>
                <tr className="flagship-row">
                  <td><strong>Zen 5</strong></td>
                  <td>Chat ladder</td>
                  <td>10 (nano 0.8B / 2B / 4B / 9B, flash, mini, default, coder, pro, max)</td>
                  <td><code>/v1/chat/completions</code></td>
                </tr>
                <tr>
                  <td><strong>Zen 5</strong></td>
                  <td>Embedding</td>
                  <td>3 (0.6B / 4B / 8B)</td>
                  <td><code>/v1/embeddings</code></td>
                </tr>
                <tr>
                  <td><strong>Zen 4</strong></td>
                  <td>Chat</td>
                  <td>7 (mini, default, pro, thinking, ultra, 4.1, max)</td>
                  <td><code>/v1/chat/completions</code></td>
                </tr>
                <tr>
                  <td><strong>Zen 4</strong></td>
                  <td>Coder</td>
                  <td>3 (flash, coder, pro)</td>
                  <td><code>/v1/chat/completions</code></td>
                </tr>
                <tr>
                  <td><strong>Zen 3</strong></td>
                  <td>Chat &amp; VL</td>
                  <td>10+ (omni, nano, guard, vl x5, web x3)</td>
                  <td><code>/v1/chat/completions</code></td>
                </tr>
                <tr>
                  <td><strong>Zen 3</strong></td>
                  <td>Embedding &amp; Reranker</td>
                  <td>11 (text + multimodal embeddings, rerankers)</td>
                  <td><code>/v1/embeddings</code>, <code>/v1/rerank</code></td>
                </tr>
                <tr>
                  <td><strong>Zen 3</strong></td>
                  <td>Image</td>
                  <td>8 (image, max, dev, fast, sdxl, playground, ssd, jp)</td>
                  <td><code>/v1/images/generations</code></td>
                </tr>
                <tr>
                  <td><strong>Zen 3</strong></td>
                  <td>Audio</td>
                  <td>6 STT/ASR + 6 TTS</td>
                  <td><code>/v1/audio/transcriptions</code>, <code>/v1/audio/speech</code></td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <CatalogSection />
    </main>
  );
}
