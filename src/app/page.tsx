import Link from 'next/link';

export default function Home() {
  return (
    <main>
      <section className="hero">
        <div className="container">
          <h2 className="hero-title">Zen LM - Open Foundation Models for Agentic AI</h2>
          <p className="hero-subtitle">95 open Zen models across Zen3, Zen4, and Zen5. Chat, code, vision, audio, image, embeddings, rerankers, and safety.</p>
          <p className="hero-description">
            From edge-class Zen5 Nano (0.8B) to the Zen5 Max frontier MoE. 8K to 1M context windows.
            OpenAI- and Anthropic-compatible API. One key, one billing surface — the Zen API, resold natively.
          </p>
          <div className="hero-cta">
            <Link href="/models" className="btn btn-primary">Explore Models</Link>
            <a href="https://api.hanzo.ai" className="btn btn-secondary" target="_blank" rel="noopener noreferrer">Get API Key</a>
            <Link href="/research" className="btn btn-outline">Research Papers</Link>
          </div>
        </div>
      </section>

      {/* Zen5 Generation */}
      <section id="zen5" className="featured-section">
        <div className="container">
          <h2 className="section-title">Zen 5 - Next-Generation Agentic Models</h2>
          <p className="section-subtitle">Native agentic training, chain-of-thought reasoning, and Zen Embedding for retrieval.</p>

          <h3 className="section-title" style={{ fontSize: '1.4rem', marginTop: '2rem' }}>Zen5 Chat Ladder</h3>
          <div className="model-lineup">
            <table className="models-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Parameters</th>
                  <th>Architecture</th>
                  <th>Context</th>
                  <th>Tier</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><strong>Zen5 Nano 0.8B</strong></td>
                  <td>0.9B dense</td>
                  <td>Multimodal dense</td>
                  <td>32K</td>
                  <td><span className="status-trained">Edge</span></td>
                </tr>
                <tr>
                  <td><strong>Zen5 Nano 2B</strong></td>
                  <td>2B dense</td>
                  <td>Multimodal dense</td>
                  <td>32K</td>
                  <td><span className="status-trained">Starter</span></td>
                </tr>
                <tr>
                  <td><strong>Zen5 Nano 4B</strong></td>
                  <td>5B dense</td>
                  <td>Multimodal dense</td>
                  <td>32K</td>
                  <td><span className="status-trained">Starter</span></td>
                </tr>
                <tr>
                  <td><strong>Zen5 Nano 9B</strong></td>
                  <td>10B dense</td>
                  <td>Multimodal dense</td>
                  <td>32K</td>
                  <td><span className="status-trained">Pro</span></td>
                </tr>
                <tr>
                  <td><strong>Zen5 Flash</strong></td>
                  <td>4B dense</td>
                  <td>Zen dense</td>
                  <td>32K</td>
                  <td><span className="status-trained">Starter</span></td>
                </tr>
                <tr>
                  <td><strong>Zen5 Mini</strong></td>
                  <td>230B (10B active)</td>
                  <td>Zen agentic MoE</td>
                  <td>192K</td>
                  <td><span className="status-trained">Pro</span></td>
                </tr>
                <tr>
                  <td><strong>Zen5</strong> <span className="status-flagship">DEFAULT</span></td>
                  <td>35B (3B active)</td>
                  <td>Zen frontier MoE</td>
                  <td>256K</td>
                  <td><span className="status-trained">Pro</span></td>
                </tr>
                <tr>
                  <td><strong>Zen5 Coder</strong></td>
                  <td>80B sparse</td>
                  <td>Zen Coder MoE</td>
                  <td>256K</td>
                  <td><span className="status-trained">Pro</span></td>
                </tr>
                <tr>
                  <td><strong>Zen5 Pro</strong></td>
                  <td>284B (37B active)</td>
                  <td>Zen Flash MoE</td>
                  <td>1M</td>
                  <td><span className="status-trained">Ultra</span></td>
                </tr>
                <tr>
                  <td><strong>Zen5 Max</strong> <span className="status-flagship">FRONTIER</span></td>
                  <td>Full Zen Pro weights</td>
                  <td>Zen Pro MoE</td>
                  <td>1M+</td>
                  <td><span className="status-trained">Ultra Max</span></td>
                </tr>
              </tbody>
            </table>
          </div>

          <h3 className="section-title" style={{ fontSize: '1.4rem', marginTop: '2rem' }}>Zen5 Embedding</h3>
          <div className="model-lineup">
            <table className="models-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Parameters</th>
                  <th>Dimensions</th>
                  <th>Context</th>
                  <th>Endpoint</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><strong>Zen5 Embedding 0.6B</strong></td>
                  <td>0.6B</td>
                  <td>1024</td>
                  <td>32K</td>
                  <td><code>/v1/embeddings</code></td>
                </tr>
                <tr>
                  <td><strong>Zen5 Embedding 4B</strong></td>
                  <td>4B</td>
                  <td>2560</td>
                  <td>32K</td>
                  <td><code>/v1/embeddings</code></td>
                </tr>
                <tr>
                  <td><strong>Zen5 Embedding 8B</strong></td>
                  <td>8B</td>
                  <td>4096</td>
                  <td>32K</td>
                  <td><code>/v1/embeddings</code></td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* Zen4 Generation */}
      <section id="zen4" className="featured-section">
        <div className="container">
          <h2 className="section-title">Zen 4 - Production Chat &amp; Code</h2>
          <p className="section-subtitle">The everyday Zen production family. MoE flagships, thinking models, and a dedicated coder line.</p>

          <h3 className="section-title" style={{ fontSize: '1.4rem', marginTop: '2rem' }}>Zen4 Chat</h3>
          <div className="model-lineup">
            <table className="models-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Parameters</th>
                  <th>Architecture</th>
                  <th>Context</th>
                  <th>Tier</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><strong>Zen4 Mini</strong></td>
                  <td>Dense</td>
                  <td>Dense</td>
                  <td>128K</td>
                  <td><span className="status-trained">Starter</span></td>
                </tr>
                <tr>
                  <td><strong>Zen4</strong></td>
                  <td>744B (40B active)</td>
                  <td>MoE</td>
                  <td>202K</td>
                  <td><span className="status-trained">Ultra Max</span></td>
                </tr>
                <tr>
                  <td><strong>Zen4 Pro</strong></td>
                  <td>80B (3B active)</td>
                  <td>MoE</td>
                  <td>131K</td>
                  <td><span className="status-trained">Ultra</span></td>
                </tr>
                <tr>
                  <td><strong>Zen4 Thinking</strong></td>
                  <td>80B (3B active)</td>
                  <td>MoE + CoT</td>
                  <td>131K</td>
                  <td><span className="status-trained">Pro Max</span></td>
                </tr>
                <tr>
                  <td><strong>Zen4 Ultra</strong></td>
                  <td>744B (40B active)</td>
                  <td>MoE + CoT</td>
                  <td>262K</td>
                  <td><span className="status-trained">Ultra Max</span></td>
                </tr>
                <tr>
                  <td><strong>Zen4.1</strong></td>
                  <td>Dense</td>
                  <td>Dense</td>
                  <td>1M</td>
                  <td><span className="status-trained">Ultra</span></td>
                </tr>
                <tr>
                  <td><strong>Zen4 Max</strong> <span className="status-flagship">FLAGSHIP</span></td>
                  <td>Dense</td>
                  <td>Dense</td>
                  <td>1M</td>
                  <td><span className="status-trained">Ultra Max</span></td>
                </tr>
              </tbody>
            </table>
          </div>

          <h3 className="section-title" style={{ fontSize: '1.4rem', marginTop: '2rem' }}>Zen4 Coder</h3>
          <div className="model-lineup">
            <table className="models-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Parameters</th>
                  <th>Architecture</th>
                  <th>Context</th>
                  <th>Tier</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><strong>Zen4 Coder Flash</strong></td>
                  <td>30B (3B active)</td>
                  <td>MoE</td>
                  <td>262K</td>
                  <td><span className="status-trained">Pro Max</span></td>
                </tr>
                <tr>
                  <td><strong>Zen4 Coder</strong></td>
                  <td>480B (35B active)</td>
                  <td>MoE</td>
                  <td>163K</td>
                  <td><span className="status-trained">Ultra</span></td>
                </tr>
                <tr>
                  <td><strong>Zen4 Coder Pro</strong></td>
                  <td>480B</td>
                  <td>Dense BF16</td>
                  <td>131K</td>
                  <td><span className="status-trained">Ultra Max</span></td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* Zen3 Multimodal */}
      <section id="zen3" className="featured-section">
        <div className="container">
          <h2 className="section-title">Zen 3 - Multimodal &amp; Specialty</h2>
          <p className="section-subtitle">Vision, audio, web, safety, embeddings, image generation, and edge.</p>

          <h3 className="section-title" style={{ fontSize: '1.4rem', marginTop: '2rem' }}>Zen3 Chat &amp; Vision</h3>
          <div className="model-lineup">
            <table className="models-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Parameters</th>
                  <th>Modalities</th>
                  <th>Context</th>
                  <th>Tier</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><strong>Zen3 Omni</strong></td>
                  <td>~200B dense</td>
                  <td>Text + Vision + Audio</td>
                  <td>202K</td>
                  <td><span className="status-trained">Pro Max</span></td>
                </tr>
                <tr>
                  <td><strong>Zen3 VL</strong> (default 30B-A3B)</td>
                  <td>30B (3B active)</td>
                  <td>Vision + Language</td>
                  <td>262K</td>
                  <td><span className="status-trained">Pro Max</span></td>
                </tr>
                <tr>
                  <td><strong>Zen3 VL 2B / 8B / 32B</strong></td>
                  <td>2B / 9B / 33B</td>
                  <td>Vision + Language</td>
                  <td>32K - 128K</td>
                  <td><span className="status-trained">Starter - Pro Max</span></td>
                </tr>
                <tr>
                  <td><strong>Zen3 VL 235B-A22B</strong> <span className="status-flagship">FRONTIER VL</span></td>
                  <td>235B (22B active)</td>
                  <td>Vision + Language (MoE)</td>
                  <td>256K</td>
                  <td><span className="status-trained">Ultra Max</span></td>
                </tr>
                <tr>
                  <td><strong>Zen3 Web 8B / 14B / 32B</strong></td>
                  <td>8B / 15B / 32B</td>
                  <td>Web agentic / browser</td>
                  <td>32K</td>
                  <td><span className="status-trained">Starter - Pro Max</span></td>
                </tr>
                <tr>
                  <td><strong>Zen3 Nano</strong></td>
                  <td>8B dense</td>
                  <td>Edge chat</td>
                  <td>128K</td>
                  <td><span className="status-trained">Starter</span></td>
                </tr>
                <tr>
                  <td><strong>Zen3 Guard</strong></td>
                  <td>4B dense</td>
                  <td>Safety classifier</td>
                  <td>65K</td>
                  <td><span className="status-trained">Pro</span></td>
                </tr>
              </tbody>
            </table>
          </div>

          <h3 className="section-title" style={{ fontSize: '1.4rem', marginTop: '2rem' }}>Zen3 Embedding &amp; Reranker</h3>
          <div className="model-lineup">
            <table className="models-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Type</th>
                  <th>Parameters</th>
                  <th>Context</th>
                  <th>Endpoint</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><strong>Zen3 Embedding (small / medium / default)</strong></td>
                  <td>Text embedding</td>
                  <td>0.6B / 4B / N/A</td>
                  <td>8K - 40K</td>
                  <td><code>/v1/embeddings</code></td>
                </tr>
                <tr>
                  <td><strong>Zen3 Reranker (small / medium / default)</strong></td>
                  <td>Reranker</td>
                  <td>0.6B / 4B / 8B</td>
                  <td>40K</td>
                  <td><code>/v1/rerank</code></td>
                </tr>
                <tr>
                  <td><strong>Zen3 VL Embedding 2B / 8B</strong></td>
                  <td>Multimodal embedding</td>
                  <td>2B / 8B</td>
                  <td>32K</td>
                  <td><code>/v1/embeddings</code></td>
                </tr>
                <tr>
                  <td><strong>Zen3 VL Reranker 2B / 8B</strong></td>
                  <td>Multimodal reranker</td>
                  <td>2B / 9B</td>
                  <td>32K</td>
                  <td><code>/v1/rerank</code></td>
                </tr>
              </tbody>
            </table>
          </div>

          <h3 className="section-title" style={{ fontSize: '1.4rem', marginTop: '2rem' }}>Zen3 Image, Audio &amp; Speech</h3>
          <div className="model-lineup">
            <table className="models-table">
              <thead>
                <tr>
                  <th>Family</th>
                  <th>Models</th>
                  <th>Endpoint</th>
                  <th>Pricing</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><strong>Zen3 Image</strong></td>
                  <td>image, image-max, image-dev, image-fast, image-sdxl, image-playground, image-ssd, image-jp</td>
                  <td><code>/v1/images/generations</code></td>
                  <td>from $0.00013/step</td>
                </tr>
                <tr>
                  <td><strong>Zen3 Audio</strong> (STT)</td>
                  <td>audio, audio-fast, asr, asr-0.6B, asr-aligner, asr-v1</td>
                  <td><code>/v1/audio/transcriptions</code></td>
                  <td>from $0.002/min</td>
                </tr>
                <tr>
                  <td><strong>Zen3 TTS</strong></td>
                  <td>tts, tts-hd, tts-fast, tts-0.6B, tts-voice-design, tts-custom-voice</td>
                  <td><code>/v1/audio/speech</code></td>
                  <td>from $2/1M chars</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* Why Zen */}
      <section id="overview" className="architecture-section">
        <div className="container">
          <h2 className="section-title">Why Zen</h2>
          <div className="arch-grid">
            <div className="arch-card">
              <h3>One API, Three Generations</h3>
              <p>Zen3, Zen4, and Zen5 share one OpenAI- and Anthropic-compatible API. Migrate generations without changing client code.</p>
            </div>
            <div className="arch-card">
              <h3>Frontier Agentic</h3>
              <p>Zen5 Mini hits 80.2% SWE-Bench Verified and 76.3% BrowseComp. Zen5 ladder is trained on 200K+ real-world environments via large-scale RL.</p>
            </div>
            <div className="arch-card">
              <h3>Edge to Frontier</h3>
              <p>From Zen5 Nano 0.8B (phone / browser WASM class) to Zen5 Max (1M+ context, Mac Studio Ultra class). Same family, same identity layer.</p>
            </div>
            <div className="arch-card">
              <h3>Full Modality Coverage</h3>
              <p>Chat, code, vision-language, web agentic, embeddings, rerankers, image generation, streaming ASR, and TTS - all under the Zen brand.</p>
            </div>
            <div className="arch-card">
              <h3>Long Context</h3>
              <p>Up to 1M tokens on Zen5 Pro and Zen5 Max. 256K standard on Zen5 default and Zen5 Coder. 8K-262K across the rest of the catalog.</p>
            </div>
            <div className="arch-card">
              <h3>Open Weights Where It Counts</h3>
              <p>Zen5 default is Apache-2.0. Zen5 Pro ships as 81 GB IQ2_XXS-imatrix GGUF on HuggingFace. Run frontier models on a single 128 GB box.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Dataset Section */}
      <section id="dataset" className="dataset-section">
        <div className="container">
          <h2 className="section-title">Zen Agentic Dataset</h2>
          <p className="section-subtitle">10B+ tokens of real-world tool use and multi-step reasoning</p>

          <div className="arch-grid">
            <div className="arch-card">
              <div className="arch-icon">10B+</div>
              <h3>Tokens</h3>
              <p>Real agentic training data across the Zen5 ladder</p>
            </div>
            <div className="arch-card">
              <div className="arch-icon">200K+</div>
              <h3>Environments</h3>
              <p>Real-world environments used in large-scale RL</p>
            </div>
            <div className="arch-card">
              <div className="arch-icon">80.2%</div>
              <h3>SWE-Bench Verified</h3>
              <p>Zen5 Mini on the SWE-Bench Verified benchmark</p>
            </div>
            <div className="arch-card">
              <div className="arch-icon">76.3%</div>
              <h3>BrowseComp</h3>
              <p>Zen5 Mini on the BrowseComp web-agent benchmark</p>
            </div>
          </div>

          <div className="dataset-cta" style={{ textAlign: 'center', marginTop: '2rem' }}>
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
              <h3>Zen API</h3>
              <p>One key, every Zen model. OpenAI- and Anthropic-compatible.</p>
              <a href="https://api.hanzo.ai" className="btn btn-primary" target="_blank" rel="noopener noreferrer">
                Get API Key
              </a>
            </div>
            <div className="download-card">
              <h3>Zen5 (Default)</h3>
              <p>35B MoE, 3B active, 256K context, Apache-2.0.</p>
              <a href="https://huggingface.co/zenlm/zen-5" className="btn btn-primary" target="_blank" rel="noopener noreferrer">
                Download
              </a>
            </div>
            <div className="download-card">
              <h3>Zen5 Pro</h3>
              <p>284B MoE (37B active), 1M context, 81 GB IQ2_XXS-imatrix.</p>
              <a href="https://huggingface.co/zenlm/zen-5-pro-gguf" className="btn btn-primary" target="_blank" rel="noopener noreferrer">
                Download GGUF
              </a>
            </div>
            <div className="download-card">
              <h3>Browse All Models</h3>
              <p>All 95 Zen models on HuggingFace.</p>
              <a href="https://huggingface.co/zenlm" className="btn btn-primary" target="_blank" rel="noopener noreferrer">
                HuggingFace Hub
              </a>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}
