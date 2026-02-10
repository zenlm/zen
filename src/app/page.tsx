import Link from 'next/link';

export default function Home() {
  return (
    <main>
      <section className="hero">
        <div className="container">
          <h2 className="hero-title">Zen4 - Open Foundation Models for Agentic AI</h2>
          <p className="hero-subtitle">The Zen4 model family: 4B to 1T+ parameters built on Qwen3 and frontier MoE architectures</p>
          <p className="hero-description">
            Zen4 delivers production-ready AI models across Consumer, Coder, and Ultra tiers.
            From the 4B Zen4 Mini for edge devices to the flagship Zen4 Pro Max (80B MoE) and Zen4 Coder for agentic programming,
            every model is trained on 8.47 billion tokens of real-world development data.
          </p>
          <div className="hero-cta">
            <Link href="/models" className="btn btn-primary">Explore Models</Link>
            <Link href="/datasets" className="btn btn-secondary">Training Data</Link>
            <Link href="/research" className="btn btn-outline">Research Papers</Link>
          </div>
        </div>
      </section>

      {/* Zen4 Model Family */}
      <section id="zen4" className="featured-section">
        <div className="container">
          <h2 className="section-title">Zen4 Model Family</h2>
          <p className="section-subtitle">Consumer, Coder, and Ultra tiers built on Qwen3 and frontier MoE architectures</p>

          <h3 className="section-title" style={{fontSize: '1.4rem', marginTop: '2rem'}}>Consumer Line</h3>
          <div className="model-lineup">
            <table className="models-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Size</th>
                  <th>Base</th>
                  <th>Architecture</th>
                  <th>Context</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><strong>Zen4 Mini</strong></td>
                  <td>4B</td>
                  <td>Qwen3-4B</td>
                  <td>Dense</td>
                  <td>32K</td>
                  <td><span className="status-trained">Available</span></td>
                </tr>
                <tr>
                  <td><strong>Zen4</strong></td>
                  <td>8B</td>
                  <td>Qwen3-8B</td>
                  <td>Dense</td>
                  <td>32K</td>
                  <td><span className="status-trained">Available</span></td>
                </tr>
                <tr>
                  <td><strong>Zen4 Pro</strong></td>
                  <td>14B</td>
                  <td>Qwen3-14B</td>
                  <td>Dense</td>
                  <td>32K</td>
                  <td><span className="status-trained">Available</span></td>
                </tr>
                <tr>
                  <td><strong>Zen4 Max</strong></td>
                  <td>30B MoE (3B active)</td>
                  <td>Qwen3-30B-A3B</td>
                  <td>MoE</td>
                  <td>256K</td>
                  <td><span className="status-trained">Available</span></td>
                </tr>
                <tr>
                  <td><strong>Zen4 Pro Max</strong> <span className="status-flagship">FLAGSHIP</span></td>
                  <td>80B MoE (3B active)</td>
                  <td>Qwen3-Next-80B-A3B</td>
                  <td>MoE</td>
                  <td>256K</td>
                  <td><span className="status-trained">Available</span></td>
                </tr>
              </tbody>
            </table>
          </div>

          <h3 className="section-title" style={{fontSize: '1.4rem', marginTop: '2rem'}}>Coder Line</h3>
          <div className="model-lineup">
            <table className="models-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Size</th>
                  <th>Base</th>
                  <th>Architecture</th>
                  <th>Context</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><strong>Zen4 Coder Flash</strong></td>
                  <td>31B MoE (3B active)</td>
                  <td>GLM-4.7-Flash</td>
                  <td>MoE</td>
                  <td>131K</td>
                  <td><span className="status-trained">Available</span></td>
                </tr>
                <tr>
                  <td><strong>Zen4 Coder</strong> <span className="status-flagship">FLAGSHIP CODE</span></td>
                  <td>80B MoE (3B active)</td>
                  <td>Qwen3-Coder-Next</td>
                  <td>MoE</td>
                  <td>256K</td>
                  <td><span className="status-trained">Available</span></td>
                </tr>
                <tr>
                  <td><strong>Zen4 Coder Pro</strong></td>
                  <td>355B</td>
                  <td>GLM-4.7</td>
                  <td>Dense</td>
                  <td>200K</td>
                  <td><span className="status-cloud">Cloud Only</span></td>
                </tr>
              </tbody>
            </table>
          </div>

          <h3 className="section-title" style={{fontSize: '1.4rem', marginTop: '2rem'}}>Ultra Line</h3>
          <div className="model-lineup">
            <table className="models-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Size</th>
                  <th>Base</th>
                  <th>Architecture</th>
                  <th>Context</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><strong>Zen4 Ultra</strong></td>
                  <td>1.04T MoE (32B active)</td>
                  <td>Kimi K2.5</td>
                  <td>MoE</td>
                  <td>256K</td>
                  <td><span className="status-cloud">Cloud Only</span></td>
                </tr>
                <tr>
                  <td><strong>Zen4 Ultra Max</strong></td>
                  <td>1T+ MoE</td>
                  <td>DeepSeek V4</td>
                  <td>MoE</td>
                  <td>-</td>
                  <td><span className="status-planned">Coming Soon</span></td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="coder-features arch-grid">
            <div className="arch-card">
              <h3>Real Agentic Data</h3>
              <p>Trained on actual agentic debug sessions - not synthetic data. Real debugging workflows, multi-file refactoring, and tool use patterns.</p>
            </div>
            <div className="arch-card">
              <h3>Production Code</h3>
              <p>15 years of professional development across AI, Web3, cryptography, and modern software engineering from 1,452 repositories.</p>
            </div>
            <div className="arch-card">
              <h3>Open Training</h3>
              <p>Use <a href="https://github.com/zenlm/zen-trainer">zen-trainer</a> to fine-tune on your own data. Supports MLX (Apple Silicon), Unsloth, and DeepSpeed.</p>
            </div>
          </div>
        </div>
      </section>

      <section id="overview" className="architecture-section">
        <div className="container">
          <h2 className="section-title">Zen4 Architecture</h2>
          <div className="arch-grid">
            <div className="arch-card">
              <div className="arch-icon">🧠</div>
              <h3>Consumer Line</h3>
              <p>5 models from 4B to 80B MoE. Dense models for edge and desktop, MoE flagships for frontier reasoning with only 3B active parameters.</p>
            </div>
            <div className="arch-card">
              <div className="arch-icon">💻</div>
              <h3>Coder Line</h3>
              <p>2 MoE coding models trained on 8.47B tokens of agentic programming data. Zen4 Coder Flash for speed, Zen4 Coder for state-of-the-art.</p>
            </div>
            <div className="arch-card">
              <div className="arch-icon">🚀</div>
              <h3>Ultra Line</h3>
              <p>Trillion-parameter MoE models for cloud deployment. Zen4 Ultra (1.04T, Kimi K2.5) and Zen4 Ultra Max (DeepSeek V4, coming soon).</p>
            </div>
            <div className="arch-card">
              <div className="arch-icon">⚡</div>
              <h3>Efficient MoE</h3>
              <p>Mixture-of-Experts architecture delivers frontier performance with only 3B active parameters - runs on consumer hardware.</p>
            </div>
            <div className="arch-card">
              <div className="arch-icon">📐</div>
              <h3>Long Context</h3>
              <p>Up to 256K context window on MoE models. Dense models support 32K context for efficient local inference.</p>
            </div>
            <div className="arch-card">
              <div className="arch-icon">🔬</div>
              <h3>Qwen3 Foundation</h3>
              <p>Built on the Qwen3 model family for proven quality. Consumer and Coder lines leverage Qwen3 dense and MoE checkpoints.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Dataset Section */}
      <section id="dataset" className="dataset-section">
        <div className="container">
          <h2 className="section-title">Zen Agentic Dataset</h2>
          <p className="section-subtitle">8.47 Billion Tokens of Real-World Agentic Programming</p>

          <div className="arch-grid">
            <div className="arch-card">
              <div className="arch-icon">8.47B</div>
              <h3>Tokens</h3>
              <p>Total training tokens across all data sources</p>
            </div>
            <div className="arch-card">
              <div className="arch-icon">3.35M</div>
              <h3>Samples</h3>
              <p>Training samples with conversation context</p>
            </div>
            <div className="arch-card">
              <div className="arch-icon">1,452</div>
              <h3>Repositories</h3>
              <p>Open source and private codebases</p>
            </div>
            <div className="arch-card">
              <div className="arch-icon">15yr</div>
              <h3>History</h3>
              <p>Years of development history (2010-2025)</p>
            </div>
          </div>

          <div className="dataset-cta" style={{textAlign: 'center', marginTop: '2rem'}}>
            <p>Available for research and commercial licensing.</p>
            <a href="mailto:z@hanzo.ai" className="btn btn-primary">Request Access</a>
            <a href="https://huggingface.co/datasets/hanzoai/zen-agentic-dataset" className="btn btn-secondary" target="_blank" rel="noopener noreferrer">View on HuggingFace</a>
          </div>
        </div>
      </section>

      <section id="downloads" className="downloads-section">
        <div className="container">
          <h2 className="section-title">Get Started</h2>
          <div className="download-grid">
            <div className="download-card">
              <h3>Download Zen4 Models</h3>
              <p>All Zen4 models available on HuggingFace Hub</p>
              <a href="https://huggingface.co/zenlm" className="btn btn-primary" target="_blank" rel="noopener noreferrer">
                Browse All Models
              </a>
            </div>
            <div className="download-card">
              <h3>Zen4 Pro Max (Flagship)</h3>
              <p>80B MoE, 3B active - runs on consumer hardware</p>
              <a href="https://huggingface.co/zenlm/zen4-pro-max" className="btn btn-primary" target="_blank" rel="noopener noreferrer">
                Download
              </a>
            </div>
            <div className="download-card">
              <h3>Zen4 Coder (Flagship Code)</h3>
              <p>80B MoE, 3B active - agentic programming</p>
              <a href="https://huggingface.co/zenlm/zen4-coder" className="btn btn-primary" target="_blank" rel="noopener noreferrer">
                Download
              </a>
            </div>
            <div className="download-card">
              <h3>zen-trainer</h3>
              <p>Fine-tune Zen4 models on your own data</p>
              <pre><code>pip install zen-trainer</code></pre>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}
