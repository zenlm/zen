import Link from 'next/link';

export default function Home() {
  return (
    <main>
      <section className="hero">
        <div className="container">
          <h2 className="hero-title">Open models for agentic coding and marketing work</h2>
          <p className="hero-subtitle">
            Zen LM is the open model family of Zoo Labs Foundation, a 501(c)(3) non-profit.
          </p>
          <p className="hero-description">
            It is chosen for two jobs: agentic coding that runs on your own machine, and marketing work.
            Zen 6 and Zen 6 Flash are available now: download the weights, or call <code>zen6</code> and{' '}
            <code>zen6-flash</code> on api.hanzo.ai.
          </p>
          <div className="hero-cta">
            <Link href="/models#zen6" className="btn btn-primary">Meet Zen 6</Link>
            <a href="https://api.hanzo.ai" className="btn btn-secondary" target="_blank" rel="noopener noreferrer">Get API Key</a>
            <Link href="/models" className="btn btn-outline">All Models</Link>
          </div>
        </div>
      </section>

      <section id="jobs" className="architecture-section">
        <div className="container">
          <h2 className="section-title">Two Jobs</h2>
          <div className="arch-grid">
            <div className="arch-card">
              <h3>Agentic coding on your own machine</h3>
              <p>
                Long context, tool use and fast decoding. Zen 6 reads 262,144 tokens natively and 1,048,576 with
                YaRN, and its bundled drafter took code completion from 62.4 to 141.2 tokens per second on one DGX
                Spark. Zen 6 Flash fits an Apple Silicon laptop.
              </p>
            </div>
            <div className="arch-card">
              <h3>Marketing work</h3>
              <p>
                Content, campaigns and brand voice, with the images and video behind them. Zen 6 reads images and
                video beside text, so the brief, the brand assets and the draft sit in one context.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section id="zen6" className="featured-section">
        <div className="container">
          <h2 className="section-title">Zen 6 — Available Now</h2>
          <p className="section-subtitle">Two builds of one model. Open weights under Apache-2.0, and hosted on api.hanzo.ai.</p>
          <div className="model-lineup">
            <table className="models-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Weights</th>
                  <th>Reads</th>
                  <th>Context</th>
                  <th>API id</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><strong>Zen 6</strong> <span className="status-flagship">AVAILABLE</span></td>
                  <td>27B dense, NVFP4</td>
                  <td>Text, images, video</td>
                  <td>262,144 native; 1,048,576 with YaRN</td>
                  <td><code>zen6</code></td>
                </tr>
                <tr>
                  <td><strong>Zen 6 Flash</strong> <span className="status-flagship">AVAILABLE</span></td>
                  <td>27B ternary, 1.77 bits per weight (5.95 GB)</td>
                  <td>Text, images</td>
                  <td>262,144 native</td>
                  <td><code>zen6-flash</code></td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <section id="zen7" className="featured-section">
        <div className="container">
          <h2 className="section-title">Zen 7 — Research Preview</h2>
          <p className="section-subtitle">
            The next open-weight generation after Zen 6. It has no weights yet and cannot be called.
          </p>
          <div className="hero-cta">
            <a href="https://hanzo.ai/research-access" className="btn btn-primary" target="_blank" rel="noopener noreferrer">Request access</a>
          </div>
        </div>
      </section>

      <section id="earlier" className="architecture-section">
        <div className="container">
          <h2 className="section-title">Earlier Generations</h2>
          <p className="section-subtitle">The generations before Zen 6, catalogued with their weights.</p>
          <div className="arch-grid">
            <div className="arch-card">
              <h3>Zen 5</h3>
              <p>A chat ladder from Zen5 Nano to Zen5 Max, a coder and three embedding models.</p>
              <Link href="/models#zen5">Zen 5 models</Link>
            </div>
            <div className="arch-card">
              <h3>Zen 4</h3>
              <p>Mixture-of-experts chat, thinking and coder models.</p>
              <Link href="/models#zen4">Zen 4 models</Link>
            </div>
            <div className="arch-card">
              <h3>Zen 3</h3>
              <p>Vision, audio, image, safety, embeddings and rerankers.</p>
              <Link href="/models#zen3">Zen 3 models</Link>
            </div>
          </div>
        </div>
      </section>

      <section id="dataset" className="dataset-section">
        <div className="container">
          <h2 className="section-title">Zen Agentic Dataset</h2>
          <p className="section-subtitle">10B+ tokens of real-world tool use and multi-step reasoning</p>
          <div className="dataset-cta" style={{ textAlign: 'center', marginTop: '2rem' }}>
            <p>Available for research and commercial licensing.</p>
            <Link href="/datasets" className="btn btn-primary">About the Dataset</Link>
            <a href="https://huggingface.co/datasets/hanzoai/zen-agentic-dataset" className="btn btn-secondary" target="_blank" rel="noopener noreferrer">View on HuggingFace</a>
          </div>
        </div>
      </section>

      <section id="downloads" className="downloads-section">
        <div className="container">
          <h2 className="section-title">Get Started</h2>
          <div className="download-grid">
            <div className="download-card">
              <h3>Zen API</h3>
              <p>One key, every Zen model. OpenAI- and Anthropic-compatible.</p>
              <a href="https://api.hanzo.ai" className="btn btn-primary" target="_blank" rel="noopener noreferrer">
                Get API Key
              </a>
            </div>
            <div className="download-card">
              <h3>Zen 6</h3>
              <p>27B dense, text, images and video, 1M context with YaRN.</p>
              <a href="https://huggingface.co/zenlm/zen6" className="btn btn-primary" target="_blank" rel="noopener noreferrer">
                Download
              </a>
            </div>
            <div className="download-card">
              <h3>Zen 6 Flash</h3>
              <p>Ternary 27B in 5.95 GB. Reads images. Runs on a laptop.</p>
              <a href="https://huggingface.co/zenlm/zen6-flash" className="btn btn-primary" target="_blank" rel="noopener noreferrer">
                Download
              </a>
            </div>
            <div className="download-card">
              <h3>Browse All Models</h3>
              <p>Every Zen model on HuggingFace.</p>
              <a href="https://huggingface.co/zenlm" className="btn btn-primary" target="_blank" rel="noopener noreferrer">
                HuggingFace Hub
              </a>
            </div>
          </div>
        </div>
      </section>

      <section id="license" className="architecture-section" data-upstream="">
        <div className="container">
          <h2 className="section-title">License &amp; attribution</h2>
          <p className="section-subtitle">
            Zen 6: Apache-2.0, built from Qwen3.8-27B by the Qwen team. Zen 6 Flash: Apache-2.0, Ternary Bonsai 2
            27B by prism-ml. Each model card on Hugging Face names its license and base models.
          </p>
        </div>
      </section>
    </main>
  );
}
