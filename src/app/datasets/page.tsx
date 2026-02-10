import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Zen Agentic Dataset - 8.47B Tokens of Real AI Programming',
  description: 'Training dataset combining agentic programming sessions with 15 years of git history from 1,452 repositories',
};

export default function DatasetsPage() {
  return (
    <main>
      <section className="hero">
        <div className="container">
          <h2 className="hero-title">Zen Agentic Dataset</h2>
          <p className="hero-subtitle">8.47 Billion Tokens of Real-World Agentic Programming</p>
          <p className="hero-description">
            A comprehensive training dataset combining agentic programming interactions with full git history
            from 1,400+ repositories spanning 15 years of professional development across AI, Web3,
            cryptography, and modern software engineering.
          </p>
        </div>
      </section>

      <section className="dataset-stats architecture-section">
        <div className="container">
          <h2 className="section-title">Quick Stats</h2>
          <div className="arch-grid">
            <div className="arch-card">
              <div className="arch-icon">8.47B</div>
              <h3>Total Tokens</h3>
            </div>
            <div className="arch-card">
              <div className="arch-icon">3.35M</div>
              <h3>Training Samples</h3>
            </div>
            <div className="arch-card">
              <div className="arch-icon">100K</div>
              <h3>Validation Samples</h3>
            </div>
            <div className="arch-card">
              <div className="arch-icon">27 GB</div>
              <h3>Total Size</h3>
            </div>
            <div className="arch-card">
              <div className="arch-icon">1,452</div>
              <h3>Repositories</h3>
            </div>
            <div className="arch-card">
              <div className="arch-icon">15 yr</div>
              <h3>Time Span (2010-2025)</h3>
            </div>
          </div>
        </div>
      </section>

      <section className="data-composition architecture-section">
        <div className="container">
          <h2 className="section-title">Data Composition</h2>
          <table className="models-table">
            <thead>
              <tr>
                <th>Component</th>
                <th>Tokens</th>
                <th>Percentage</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Git History</td>
                <td>4.03B</td>
                <td>48%</td>
              </tr>
              <tr>
                <td>Agentic Debug Sessions</td>
                <td>2.42B</td>
                <td>29%</td>
              </tr>
              <tr>
                <td>Architecture Discussions</td>
                <td>1.14B</td>
                <td>13%</td>
              </tr>
              <tr>
                <td>Code Review Sessions</td>
                <td>0.86B</td>
                <td>10%</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section className="domain-coverage architecture-section">
        <div className="container">
          <h2 className="section-title">Domain Coverage</h2>
          <div className="arch-grid">
            <div className="arch-card">
              <div className="arch-icon">🤖</div>
              <h3>Agentic AI & LLM Infrastructure</h3>
              <ul>
                <li>Model Context Protocol (MCP) - 260+ tools</li>
                <li>Multi-agent orchestration</li>
                <li>Agent frameworks - planning, memory, reflection</li>
                <li>LLM Gateway - 100+ provider proxy</li>
              </ul>
            </div>
            <div className="arch-card">
              <div className="arch-icon">⛓️</div>
              <h3>Web3 & Blockchain</h3>
              <ul>
                <li>Smart contracts - Solidity, Vyper</li>
                <li>Consensus engines - Snow family, BFT, DAG</li>
                <li>Cross-chain bridges</li>
                <li>DeFi protocols - AMMs, lending, staking</li>
              </ul>
            </div>
            <div className="arch-card">
              <div className="arch-icon">🔐</div>
              <h3>Cryptography & Security</h3>
              <ul>
                <li>Post-quantum - Kyber, Dilithium, SPHINCS+</li>
                <li>Threshold cryptography - MPC, DKG</li>
                <li>Zero-knowledge proofs</li>
                <li>Key management - HD wallets</li>
              </ul>
            </div>
            <div className="arch-card">
              <div className="arch-icon">💻</div>
              <h3>Modern Development</h3>
              <ul>
                <li>Full-stack TypeScript - Next.js, React</li>
                <li>Systems - Rust, Go, Python, C/C++</li>
                <li>DevOps - Docker, Kubernetes, CI/CD</li>
                <li>Real-time systems - Event sourcing, CQRS</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section className="languages-section architecture-section">
        <div className="container">
          <h2 className="section-title">Languages</h2>
          <table className="models-table">
            <thead>
              <tr>
                <th>Tier 1 (Core)</th>
                <th>Tier 2 (Infrastructure)</th>
                <th>Tier 3 (Specialized)</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Python</td>
                <td>SQL</td>
                <td>Solidity</td>
              </tr>
              <tr>
                <td>TypeScript</td>
                <td>Bash/Shell</td>
                <td>C/C++</td>
              </tr>
              <tr>
                <td>JavaScript</td>
                <td>YAML/TOML</td>
                <td>Protobuf</td>
              </tr>
              <tr>
                <td>Rust</td>
                <td>Dockerfile</td>
                <td>GraphQL</td>
              </tr>
              <tr>
                <td>Go</td>
                <td>Makefile</td>
                <td>Move</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section className="models-trained architecture-section">
        <div className="container">
          <h2 className="section-title">Models Trained on This Dataset</h2>
          <table className="models-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Size</th>
                <th>Architecture</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Zen Coder 4B</strong></td>
                <td>4B</td>
                <td>Qwen3</td>
                <td><span className="status-trained">Trained</span></td>
              </tr>
              <tr>
                <td><strong>Zen Coder 24B</strong></td>
                <td>24B</td>
                <td>Devstral Small 2</td>
                <td><span className="status-trained">Trained</span></td>
              </tr>
              <tr>
                <td><strong>Zen Coder 123B</strong></td>
                <td>123B</td>
                <td>Devstral 2</td>
                <td><span className="status-training">Training</span></td>
              </tr>
              <tr>
                <td><strong>Zen Coder Max</strong></td>
                <td>358B</td>
                <td>GLM-4.7 (MoE)</td>
                <td><span className="status-planned">Planned</span></td>
              </tr>
              <tr>
                <td><strong>Zen Coder Ultra</strong></td>
                <td>1T</td>
                <td>Kimi K2 (MoE)</td>
                <td><span className="status-planned">Planned</span></td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section className="unique-section architecture-section">
        <div className="container">
          <h2 className="section-title">What Makes This Dataset Unique</h2>
          <div className="arch-grid">
            <div className="arch-card">
              <h3>Real Agentic Programming</h3>
              <p>Unlike synthetic datasets, this contains <strong>actual agentic programming sessions</strong> showing real debugging workflows, multi-file refactoring decisions, architecture discussions, tool use patterns, and error recovery.</p>
            </div>
            <div className="arch-card">
              <h3>Production Code Quality</h3>
              <p>Code that shipped to production systems. Security-audited smart contracts. Performance-optimized infrastructure. Battle-tested patterns from real deployments.</p>
            </div>
          </div>
        </div>
      </section>

      <section className="access-section architecture-section">
        <div className="container">
          <h2 className="section-title">Access & Licensing</h2>
          <p style={{textAlign: 'center', marginBottom: '2rem'}}>
            This dataset is available for <strong>research and commercial licensing</strong>.
          </p>

          <div className="arch-grid">
            <div className="arch-card">
              <h3>For Developers & Researchers</h3>
              <p>We award grants to individuals and teams building:</p>
              <ul>
                <li>Models for specific blockchain ecosystems</li>
                <li>Open-source AI tools using OpenAI-compatible protocols</li>
                <li>Research advancing agentic AI capabilities</li>
                <li>Infrastructure for decentralized AI training</li>
              </ul>
            </div>
            <div className="arch-card">
              <h3>Request Access</h3>
              <p><strong>Email:</strong> <a href="mailto:z@hanzo.ai">z@hanzo.ai</a></p>
              <p>Please include:</p>
              <ul>
                <li>Intended use case (training, research, evaluation)</li>
                <li>Organization/affiliation</li>
                <li>Target ecosystem (if applicable)</li>
                <li>Licensing requirements</li>
              </ul>
            </div>
          </div>

          <div style={{textAlign: 'center', marginTop: '2rem'}}>
            <a href="mailto:z@hanzo.ai" className="btn btn-primary">Request Access</a>
            <a href="https://huggingface.co/datasets/hanzoai/zen-agentic-dataset" className="btn btn-secondary" target="_blank" rel="noopener noreferrer">View on HuggingFace</a>
          </div>
        </div>
      </section>

      <section className="training-section architecture-section">
        <div className="container">
          <h2 className="section-title">Training Framework</h2>
          <p style={{textAlign: 'center', marginBottom: '2rem'}}>
            Use <a href="https://github.com/zenlm/zen-trainer">zen-trainer</a> for fine-tuning:
          </p>
          <pre style={{background: '#1a1a1a', padding: '1.5rem', borderRadius: '8px', overflow: 'auto'}}>
            <code>{`from zen_trainer import ZenTrainer

trainer = ZenTrainer(
    model_key="qwen3-4b",
    dataset_path="hanzoai/zen-agentic-dataset-private",  # Requires access
    output_dir="./output/my-model",
)
trainer.train()`}</code>
          </pre>
          <div style={{textAlign: 'center', marginTop: '2rem'}}>
            <a href="https://github.com/zenlm/zen-trainer" className="btn btn-primary" target="_blank" rel="noopener noreferrer">View zen-trainer</a>
          </div>
        </div>
      </section>

      <section className="orgs-section architecture-section">
        <div className="container">
          <h2 className="section-title">Supported Organizations</h2>
          <table className="models-table">
            <thead>
              <tr>
                <th>Organization</th>
                <th>Focus</th>
                <th>Role</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><a href="https://hanzo.ai" target="_blank" rel="noopener noreferrer">Hanzo AI</a></td>
                <td>AI infrastructure</td>
                <td>Primary maintainer</td>
              </tr>
              <tr>
                <td><a href="https://zenlm.org" target="_blank" rel="noopener noreferrer">Zen LM</a></td>
                <td>Open model research</td>
                <td>Model training</td>
              </tr>
              <tr>
                <td><a href="https://zoo.ngo" target="_blank" rel="noopener noreferrer">Zoo Labs</a></td>
                <td>Decentralized AI</td>
                <td>Research grants</td>
              </tr>
              <tr>
                <td><a href="https://lux.network" target="_blank" rel="noopener noreferrer">Lux Network</a></td>
                <td>AI compute settlement</td>
                <td>Infrastructure</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
    </main>
  );
}
