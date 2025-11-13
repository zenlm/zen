import Link from 'next/link';
import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Zen LM Models - Complete Model Family',
  description: 'Complete collection of Zen Language Models - from nano to next-gen, language to multimodal',
};

export default function ModelsPage() {
  return (
    <main>
      <section className="hero">
        <div className="container">
          <h2 className="hero-title">Zen Model Family</h2>
          <p className="hero-subtitle">24+ models spanning language, vision, audio, video, 3D, and specialized tasks</p>
          <p className="hero-description">
            Complete model collection from 0.6B to 1T+ parameters. From efficient edge deployment to powerful
            cloud inference, each model is optimized for specific use cases while maintaining the same high
            standards of performance, transparency, and open-source accessibility.
          </p>
        </div>
      </section>

      <section id="core-models" className="models-section">
        <div className="container">
          <h2 className="section-title">Core Language Models</h2>
          <p className="section-subtitle">Foundational models from nano to next-gen</p>

          <div className="models-grid">
            {/* zen-nano */}
            <ModelCard
              name="zen-nano"
              status="Available"
              specs={[
                { label: 'Parameters', value: '0.6B' },
                { label: 'Base', value: 'Qwen3-0.6B' },
                { label: 'Context', value: '32K tokens' },
                { label: 'Architecture', value: '28 layers, GQA' },
              ]}
              description="Ultra-efficient model for edge deployment and embedded systems. Perfect for on-device AI applications with minimal resource requirements."
              formats={['SafeTensors', 'GGUF', 'MLX']}
              hfLink="https://huggingface.co/zenlm/zen-nano"
              githubLink="https://github.com/zenlm/zen-nano"
              docsLink="https://github.com/zenlm/zen-nano/tree/main/docs"
              paperLink="/papers/zen-nano_whitepaper.pdf"
            />

            {/* zen-eco */}
            <ModelCard
              name="zen-eco"
              status="Available"
              specs={[
                { label: 'Parameters', value: '4B' },
                { label: 'Base', value: 'Qwen3-3B' },
                { label: 'Context', value: '32K tokens' },
                { label: 'Variants', value: 'Instruct, Agent, Coder, Thinking' },
              ]}
              description="Balanced performance and efficiency for general-purpose applications. Multiple specialized variants for different use cases."
              formats={['SafeTensors', 'GGUF', 'MLX']}
              hfLink="https://huggingface.co/zenlm/zen-eco"
              githubLink="https://github.com/zenlm/zen-eco"
            />

            {/* zen-omni */}
            <ModelCard
              name="zen-omni"
              status="Available"
              specs={[
                { label: 'Parameters', value: '7B' },
                { label: 'Base', value: 'Qwen3-Omni' },
                { label: 'Modalities', value: 'Text + Vision + Audio' },
                { label: 'Type', value: 'Multimodal' },
              ]}
              description="Multimodal model based on Qwen3-Omni supporting text, vision, and audio understanding simultaneously. NOT Qwen2.5!"
              formats={['SafeTensors']}
              hfLink="https://huggingface.co/zenlm/zen-omni"
              githubLink="https://github.com/zenlm/zen-omni"
              docsLink="https://github.com/zenlm/zen-omni/tree/main/docs"
              paperLink="/papers/zen-omni_whitepaper.pdf"
            />

            {/* zen-coder */}
            <ModelCard
              name="zen-coder"
              status="Available"
              specs={[
                { label: 'Parameters', value: '14B' },
                { label: 'Base', value: 'Qwen3-Coder-14B' },
                { label: 'Context', value: '128K tokens' },
                { label: 'Focus', value: 'Code Generation' },
              ]}
              description="Specialized for code generation, debugging, and software engineering tasks. Supports 100+ programming languages with extended context."
              formats={['SafeTensors', 'GGUF', 'MLX']}
              hfLink="https://huggingface.co/zenlm/zen-coder"
              githubLink="https://github.com/zenlm/zen-coder"
            />

            {/* zen-next */}
            <ModelCard
              name="zen-next"
              status="Available"
              specs={[
                { label: 'Parameters', value: '32B' },
                { label: 'Base', value: 'Qwen3-32B' },
                { label: 'Context', value: '32K tokens' },
                { label: 'Focus', value: 'Frontier' },
              ]}
              description="Our flagship model pushing the boundaries of performance and capability. For the most demanding applications requiring maximum intelligence."
              formats={['SafeTensors', 'GGUF']}
              githubLink="https://github.com/zenlm/zen-next"
            />
          </div>
        </div>
      </section>

      <section id="multimodal" className="models-section">
        <div className="container">
          <h2 className="section-title">Multimodal Models</h2>
          <p className="section-subtitle">Vision, Audio, Video, and 3D Generation</p>

          <div className="models-grid">
              <ModelCard
                name="zen-vl"
                status="Available"
                specs={[
                  { label: 'Type', value: 'Vision-Language' },
                  { label: 'Base', value: 'Qwen3-VL' },
                  { label: 'Sizes', value: '4B, 8B, 30B' },
                  { label: 'Variants', value: 'Instruct, Agent' },
                  { label: 'Focus', value: 'Function Calling' },
                ]}
                description="Next-generation vision-language model with advanced function calling capabilities. Trained on Agent Data Protocol (ADP) and xLAM datasets for superior agent performance and tool use."
                formats={['SafeTensors', 'GGUF']}
                hfLink="https://huggingface.co/zenlm/zen-vl-4b-instruct"
                githubLink="https://github.com/zenlm/zen-vl"
              />

            <ModelCard
              name="zen-designer"
              status="Available"
              specs={[
                { label: 'Type', value: 'Vision-Language' },
                { label: 'Base', value: 'Qwen-VL' },
                { label: 'Variants', value: 'Instruct, Thinking' },
                { label: 'Focus', value: 'Visual Understanding' },
              ]}
              description="Advanced vision-language model for image understanding, analysis, and reasoning. Supports visual question answering, OCR, and detailed scene description."
              formats={['SafeTensors']}
              githubLink="https://github.com/zenlm/zen-designer"
            />

            <ModelCard
              name="zen-artist"
              status="Available"
              specs={[
                { label: 'Type', value: 'Text-to-Image' },
                { label: 'Base', value: 'Qwen-Image' },
                { label: 'Variants', value: 'Base, Edit' },
                { label: 'Focus', value: 'Image Generation' },
              ]}
              description="High-quality image generation from text descriptions. zen-artist-edit provides advanced image editing capabilities with natural language instructions."
              formats={['SafeTensors', 'Diffusers']}
              githubLink="https://github.com/zenlm/zen-artist"
            />

            <ModelCard
              name="zen-video"
              status="Available"
              specs={[
                { label: 'Type', value: 'Text-to-Video' },
                { label: 'Base', value: 'HunyuanVideo' },
                { label: 'Variants', value: 'T2V, I2V' },
                { label: 'Focus', value: 'Video Generation' },
              ]}
              description="State-of-the-art video generation from text descriptions. zen-video-i2v provides image-to-video generation with fine control over motion and dynamics."
              formats={['SafeTensors']}
              githubLink="https://github.com/zenlm/zen-video"
            />

            <ModelCard
              name="zen-3d"
              status="Available"
              specs={[
                { label: 'Type', value: '3D Generation' },
                { label: 'Input', value: 'Text, Image, Point Cloud' },
                { label: 'Output', value: '3D Meshes' },
                { label: 'Focus', value: '3D Assets' },
              ]}
              description="Generate high-quality 3D models from various input modalities. Perfect for game development, AR/VR, and 3D content creation."
              formats={['SafeTensors']}
              githubLink="https://github.com/zenlm/zen-3d"
            />

            <ModelCard
              name="zen-musician"
              status="Available"
              specs={[
                { label: 'Type', value: 'Music Generation' },
                { label: 'Input', value: 'Text, Audio' },
                { label: 'Output', value: 'Music, Audio' },
                { label: 'Focus', value: 'Music Creation' },
              ]}
              description="Generate high-quality music from text descriptions or audio samples. Supports multiple genres, instruments, and musical styles."
              formats={['SafeTensors']}
              githubLink="https://github.com/zenlm/zen-musician"
            />
          </div>
        </div>
      </section>
    </main>
  );
}

interface ModelCardProps {
  name: string;
  status: string;
  specs: Array<{ label: string; value: string }>;
  description: string;
  formats: string[];
  hfLink?: string;
  githubLink?: string;
  docsLink?: string;
  paperLink?: string;
}

function ModelCard({ name, status, specs, description, formats, hfLink, githubLink, docsLink, paperLink }: ModelCardProps) {
  return (
    <div className="model-card complete">
      <div className="model-header">
        <h3>{name}</h3>
        <span className="badge badge-complete">{status}</span>
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
      <div className="model-formats">
        {formats.map((format, idx) => (
          <span key={idx} className="format-tag">{format}</span>
        ))}
      </div>
      <div className="model-actions">
        {hfLink && (
          <a href={hfLink} className="btn btn-sm btn-primary" target="_blank" rel="noopener noreferrer">
            🤗 HF
          </a>
        )}
        {githubLink && (
          <a href={githubLink} className="btn btn-sm btn-secondary" target="_blank" rel="noopener noreferrer">
            📦 GitHub
          </a>
        )}
        {docsLink && (
          <a href={docsLink} className="btn btn-sm btn-secondary" target="_blank" rel="noopener noreferrer">
            📖 Docs
          </a>
        )}
        {paperLink && (
          <a href={paperLink} className="btn btn-sm btn-secondary" target="_blank" rel="noopener noreferrer">
            📄 Paper
          </a>
        )}
      </div>
    </div>
  );
}
