import Link from 'next/link';
import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Zen4 Models - Foundation Models from 4B to 1T+',
  description: 'Complete Zen4 model family - Consumer, Coder, and Ultra tiers built on abliterated Qwen3 and frontier MoE architectures',
};

export default function ModelsPage() {
  return (
    <main>
      <section className="hero">
        <div className="container">
          <h2 className="hero-title">Zen4 Model Family</h2>
          <p className="hero-subtitle">Foundation models from 4B to 1T+ parameters</p>
          <p className="hero-description">
            Open, uncensored AI models built on abliterated weights from Qwen3 and frontier MoE architectures.
            From edge deployment to cloud-scale reasoning, Zen4 models run unrestricted with no safety theater.
          </p>
          <div className="hero-cta">
            <a href="https://huggingface.co/zenlm" className="btn btn-primary" target="_blank" rel="noopener noreferrer">Browse All Models</a>
            <Link href="/datasets" className="btn btn-secondary">Training Data</Link>
          </div>
        </div>
      </section>

      {/* Zen4 Consumer Line */}
      <section id="consumer" className="models-section">
        <div className="container">
          <h2 className="section-title">Consumer Line</h2>
          <p className="section-subtitle">Dense and MoE models for desktop and edge deployment</p>

          <div className="models-grid">
            <ModelCard
              name="Zen4 Mini"
              status="Available"
              specs={[
                { label: 'Parameters', value: '4B dense' },
                { label: 'Base', value: 'Qwen3-4B-Instruct-2507' },
                { label: 'Context', value: '32K tokens' },
                { label: 'Architecture', value: 'Dense, abliterated' },
              ]}
              description="Ultra-efficient model for edge and mobile deployment. Full Qwen3 quality at 4B parameters with no safety restrictions."
              formats={['SafeTensors', 'GGUF', 'MLX']}
              hfLink="https://huggingface.co/zenlm/zen4-mini"
              githubLink="https://github.com/zenlm/zen4-mini"
            />

            <ModelCard
              name="Zen4"
              status="Available"
              specs={[
                { label: 'Parameters', value: '8B dense' },
                { label: 'Base', value: 'Qwen3-8B' },
                { label: 'Context', value: '32K tokens' },
                { label: 'Architecture', value: 'Dense, abliterated' },
              ]}
              description="The standard Zen4 model. Excellent balance of quality and efficiency for general-purpose AI assistance."
              formats={['SafeTensors', 'GGUF', 'MLX']}
              hfLink="https://huggingface.co/zenlm/zen4"
              githubLink="https://github.com/zenlm/zen4"
            />

            <ModelCard
              name="Zen4 Pro"
              status="Available"
              specs={[
                { label: 'Parameters', value: '14B dense' },
                { label: 'Base', value: 'Qwen3-14B' },
                { label: 'Context', value: '32K tokens' },
                { label: 'Architecture', value: 'Dense, abliterated' },
              ]}
              description="Professional-grade model for demanding tasks. Strong reasoning and code generation at a size that fits on most GPUs."
              formats={['SafeTensors', 'GGUF', 'MLX']}
              hfLink="https://huggingface.co/zenlm/zen4-pro"
              githubLink="https://github.com/zenlm/zen4-pro"
            />

            <ModelCard
              name="Zen4 Max"
              status="Available"
              flagship={true}
              specs={[
                { label: 'Parameters', value: '30B MoE (3B active)' },
                { label: 'Base', value: 'Qwen3-30B-A3B-Instruct-2507' },
                { label: 'Context', value: '256K tokens' },
                { label: 'Architecture', value: 'MoE, abliterated' },
              ]}
              description="Flagship efficient model. 30B total parameters with only 3B active via Mixture-of-Experts - frontier performance on consumer hardware."
              formats={['SafeTensors', 'GGUF', 'MLX']}
              hfLink="https://huggingface.co/zenlm/zen4-max"
              githubLink="https://github.com/zenlm/zen4-max"
            />

            <ModelCard
              name="Zen4 Max Pro"
              status="Available"
              flagship={true}
              specs={[
                { label: 'Parameters', value: '80B MoE (3B active)' },
                { label: 'Base', value: 'Qwen3-Next-80B-A3B-Instruct' },
                { label: 'Context', value: '256K tokens' },
                { label: 'Architecture', value: 'Hybrid DeltaNet + MoE' },
              ]}
              description="The ultimate consumer model. 80B parameters with hybrid Gated DeltaNet + Gated Attention + MoE architecture, running at just 3B active parameters."
              formats={['SafeTensors', 'GGUF', 'MLX']}
              hfLink="https://huggingface.co/zenlm/zen4-max-pro"
              githubLink="https://github.com/zenlm/zen4-max-pro"
            />
          </div>
        </div>
      </section>

      {/* Zen4 Coder Line */}
      <section id="coder" className="models-section">
        <div className="container">
          <h2 className="section-title">Coder Line</h2>
          <p className="section-subtitle">Specialized models for agentic programming</p>

          <div className="models-grid">
            <ModelCard
              name="Zen4 Coder Flash"
              status="Available"
              specs={[
                { label: 'Parameters', value: '31B MoE (3B active)' },
                { label: 'Base', value: 'GLM-4.7-Flash' },
                { label: 'Context', value: '131K tokens' },
                { label: 'License', value: 'MIT' },
              ]}
              description="Fast coding model for rapid iteration. GLM-4.7-Flash base with MoE efficiency - perfect for real-time code completion and quick edits."
              formats={['SafeTensors', 'GGUF', 'MLX']}
              hfLink="https://huggingface.co/zenlm/zen4-coder-flash"
              githubLink="https://github.com/zenlm/zen4-coder-flash"
            />

            <ModelCard
              name="Zen4 Coder"
              status="Available"
              flagship={true}
              specs={[
                { label: 'Parameters', value: '80B MoE (3B active)' },
                { label: 'Base', value: 'Qwen3-Coder-Next' },
                { label: 'Context', value: '256K tokens' },
                { label: 'Architecture', value: 'Hybrid DeltaNet + MoE (512 experts)' },
              ]}
              description="Flagship coding model. 80B parameters with 512-expert MoE architecture for state-of-the-art code generation, debugging, and agentic programming."
              formats={['SafeTensors', 'GGUF', 'MLX']}
              hfLink="https://huggingface.co/zenlm/zen4-coder"
              githubLink="https://github.com/zenlm/zen4-coder"
            />

            <ModelCard
              name="Zen4 Coder Pro"
              status="Cloud Only"
              frontier={true}
              specs={[
                { label: 'Parameters', value: '355B dense' },
                { label: 'Base', value: 'GLM-4.7' },
                { label: 'Context', value: '200K tokens' },
                { label: 'License', value: 'MIT' },
              ]}
              description="Dense 355B coding powerhouse. GLM-4.7 base for maximum code intelligence - available via cloud API."
              formats={['SafeTensors']}
              hfLink="https://huggingface.co/zenlm/zen4-coder-pro"
              githubLink="https://github.com/zenlm/zen4-coder-pro"
            />
          </div>
        </div>
      </section>

      {/* Zen4 Ultra Line */}
      <section id="ultra" className="models-section">
        <div className="container">
          <h2 className="section-title">Ultra Line</h2>
          <p className="section-subtitle">Trillion-parameter models for cloud deployment</p>

          <div className="models-grid">
            <ModelCard
              name="Zen4 Ultra"
              status="Cloud Only"
              frontier={true}
              specs={[
                { label: 'Parameters', value: '1.04T MoE (32B active)' },
                { label: 'Base', value: 'Kimi K2.5 Thinking' },
                { label: 'Context', value: '256K tokens' },
                { label: 'Architecture', value: 'MoE (384 experts), Multimodal' },
              ]}
              description="Trillion-parameter frontier model with 384 experts and vision capabilities via MoonViT. Cloud-only deployment for maximum intelligence."
              formats={['SafeTensors']}
              hfLink="https://huggingface.co/zenlm/zen4-ultra"
              githubLink="https://github.com/zenlm/zen4-ultra"
            />

            <ModelCard
              name="Zen4 Ultra Max"
              status="Coming Soon"
              specs={[
                { label: 'Parameters', value: '1T+ MoE' },
                { label: 'Base', value: 'DeepSeek V4' },
                { label: 'Context', value: '1M tokens' },
                { label: 'Architecture', value: 'MoE' },
              ]}
              description="Next-generation trillion-parameter model with 1M context window. Based on DeepSeek V4 architecture."
              formats={['SafeTensors']}
            />
          </div>
        </div>
      </section>

      {/* Multimodal Models */}
      <section id="multimodal" className="models-section">
        <div className="container">
          <h2 className="section-title">Multimodal &amp; Specialized</h2>
          <p className="section-subtitle">Vision, audio, video, 3D, and domain-specific models</p>

          <div className="models-grid">
            <ModelCard
              name="zen-omni"
              status="Available"
              specs={[
                { label: 'Parameters', value: '7B' },
                { label: 'Base', value: 'Qwen3-Omni' },
                { label: 'Modalities', value: 'Text + Vision + Audio' },
                { label: 'Type', value: 'Multimodal' },
              ]}
              description="Multimodal model based on Qwen3-Omni supporting text, vision, and audio understanding simultaneously."
              formats={['SafeTensors']}
              hfLink="https://huggingface.co/zenlm/zen-omni"
              githubLink="https://github.com/zenlm/zen-omni"
            />

            <ModelCard
              name="zen-vl"
              status="Available"
              specs={[
                { label: 'Type', value: 'Vision-Language' },
                { label: 'Base', value: 'Qwen3-VL' },
                { label: 'Sizes', value: '4B, 8B, 30B' },
                { label: 'Focus', value: 'Function Calling' },
              ]}
              description="Vision-language model with advanced function calling. Trained on Agent Data Protocol and xLAM datasets for tool use."
              formats={['SafeTensors', 'GGUF']}
              hfLink="https://huggingface.co/zenlm/zen-vl-4b-instruct"
              githubLink="https://github.com/zenlm/zen-vl"
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
              description="State-of-the-art video generation from text descriptions with image-to-video capabilities."
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
              description="Generate high-quality 3D models from various inputs. For game dev, AR/VR, and 3D content creation."
              formats={['SafeTensors']}
              githubLink="https://github.com/zenlm/zen-3d"
            />
          </div>
        </div>
      </section>

      {/* Comparison Table */}
      <section id="comparison" className="featured-section">
        <div className="container">
          <h2 className="section-title">Full Zen4 Lineup</h2>
          <div className="model-lineup">
            <table className="models-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Parameters</th>
                  <th>Active</th>
                  <th>Base</th>
                  <th>Context</th>
                  <th>License</th>
                  <th>HuggingFace</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><strong>Zen4 Mini</strong></td>
                  <td>4B</td>
                  <td>4B</td>
                  <td>Qwen3-4B</td>
                  <td>32K</td>
                  <td>Apache 2.0</td>
                  <td><a href="https://huggingface.co/zenlm/zen4-mini" target="_blank" rel="noopener noreferrer">zenlm/zen4-mini</a></td>
                </tr>
                <tr>
                  <td><strong>Zen4</strong></td>
                  <td>8B</td>
                  <td>8B</td>
                  <td>Qwen3-8B</td>
                  <td>32K</td>
                  <td>Apache 2.0</td>
                  <td><a href="https://huggingface.co/zenlm/zen4" target="_blank" rel="noopener noreferrer">zenlm/zen4</a></td>
                </tr>
                <tr>
                  <td><strong>Zen4 Pro</strong></td>
                  <td>14B</td>
                  <td>14B</td>
                  <td>Qwen3-14B</td>
                  <td>32K</td>
                  <td>Apache 2.0</td>
                  <td><a href="https://huggingface.co/zenlm/zen4-pro" target="_blank" rel="noopener noreferrer">zenlm/zen4-pro</a></td>
                </tr>
                <tr>
                  <td><strong>Zen4 Max</strong></td>
                  <td>30B MoE</td>
                  <td>3B</td>
                  <td>Qwen3-30B-A3B</td>
                  <td>256K</td>
                  <td>Apache 2.0</td>
                  <td><a href="https://huggingface.co/zenlm/zen4-max" target="_blank" rel="noopener noreferrer">zenlm/zen4-max</a></td>
                </tr>
                <tr className="flagship-row">
                  <td><strong>Zen4 Max Pro</strong></td>
                  <td>80B MoE</td>
                  <td>3B</td>
                  <td>Qwen3-Next-80B</td>
                  <td>256K</td>
                  <td>Apache 2.0</td>
                  <td><a href="https://huggingface.co/zenlm/zen4-max-pro" target="_blank" rel="noopener noreferrer">zenlm/zen4-max-pro</a></td>
                </tr>
                <tr>
                  <td><strong>Zen4 Coder Flash</strong></td>
                  <td>31B MoE</td>
                  <td>3B</td>
                  <td>GLM-4.7-Flash</td>
                  <td>131K</td>
                  <td>MIT</td>
                  <td><a href="https://huggingface.co/zenlm/zen4-coder-flash" target="_blank" rel="noopener noreferrer">zenlm/zen4-coder-flash</a></td>
                </tr>
                <tr className="flagship-row">
                  <td><strong>Zen4 Coder</strong></td>
                  <td>80B MoE</td>
                  <td>3B</td>
                  <td>Qwen3-Coder-Next</td>
                  <td>256K</td>
                  <td>Apache 2.0</td>
                  <td><a href="https://huggingface.co/zenlm/zen4-coder" target="_blank" rel="noopener noreferrer">zenlm/zen4-coder</a></td>
                </tr>
                <tr>
                  <td><strong>Zen4 Coder Pro</strong></td>
                  <td>355B</td>
                  <td>355B</td>
                  <td>GLM-4.7</td>
                  <td>200K</td>
                  <td>MIT</td>
                  <td><a href="https://huggingface.co/zenlm/zen4-coder-pro" target="_blank" rel="noopener noreferrer">zenlm/zen4-coder-pro</a></td>
                </tr>
                <tr>
                  <td><strong>Zen4 Ultra</strong></td>
                  <td>1.04T MoE</td>
                  <td>32B</td>
                  <td>Kimi K2.5</td>
                  <td>256K</td>
                  <td>MIT</td>
                  <td><a href="https://huggingface.co/zenlm/zen4-ultra" target="_blank" rel="noopener noreferrer">zenlm/zen4-ultra</a></td>
                </tr>
                <tr>
                  <td><strong>Zen4 Ultra Max</strong></td>
                  <td>1T+ MoE</td>
                  <td>TBD</td>
                  <td>DeepSeek V4</td>
                  <td>1M</td>
                  <td>-</td>
                  <td><span className="status-planned">Coming Soon</span></td>
                </tr>
              </tbody>
            </table>
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
  flagship?: boolean;
  frontier?: boolean;
  hfLink?: string;
  githubLink?: string;
  docsLink?: string;
  paperLink?: string;
}

function ModelCard({ name, status, specs, description, formats, flagship, frontier, hfLink, githubLink, docsLink, paperLink }: ModelCardProps) {
  const badgeClass = status === 'Coming Soon' ? 'badge badge-planned' :
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
      <div className="model-formats">
        {formats.map((format, idx) => (
          <span key={idx} className="format-tag">{format}</span>
        ))}
      </div>
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
        {paperLink && (
          <a href={paperLink} className="btn btn-sm btn-secondary" target="_blank" rel="noopener noreferrer">
            Paper
          </a>
        )}
      </div>
    </div>
  );
}
