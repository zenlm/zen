import Link from 'next/link';

export default function Home() {
  return (
    <main>
      <section className="hero">
        <div className="container">
          <h2 className="hero-title">Real-Time Hyper-Modal AI for XR/VR/Robotics</h2>
          <p className="hero-subtitle">Ultra-low latency models with spatial awareness for immersive experiences</p>
          <p className="hero-description">
            Zen LM powers next-generation XR/VR applications and robotic systems with real-time multimodal understanding.
            From 3D scene generation to spatial audio, from gesture recognition to embodied navigation - our models deliver
            sub-10ms latency for seamless human-AI interaction in extended reality and physical environments.
          </p>
          <div className="hero-cta">
            <Link href="/models" className="btn btn-primary">Explore Models</Link>
            <Link href="/research" className="btn btn-secondary">Read Research</Link>
          </div>
        </div>
      </section>

      <section id="overview" className="architecture-section">
        <div className="container">
          <h2 className="section-title">Complete AI Stack for Immersive Computing</h2>
          <div className="arch-grid">
            <div className="arch-card">
              <div className="arch-icon">🧠</div>
              <h3>Core Language Models</h3>
              <p>6 models from 0.6B to 1T+ parameters for edge to cloud deployment. Optimized for real-time instruction following and reasoning in XR environments.</p>
            </div>
            <div className="arch-card">
              <div className="arch-icon">👁️</div>
              <h3>Multimodal Models</h3>
              <p>10 specialized models for vision, audio, video, 3D generation, and spatial understanding. Built for seamless integration with XR/VR platforms.</p>
            </div>
            <div className="arch-card">
              <div className="arch-icon">🤖</div>
              <h3>Specialized Systems</h3>
              <p>Agent frameworks, safety guardrails, embeddings, and IDE tools. Production-ready infrastructure for embodied AI and robotic applications.</p>
            </div>
            <div className="arch-card">
              <div className="arch-icon">⚡</div>
              <h3>High-Performance Infrastructure</h3>
              <p>Rust-based inference engine with GGUF quantization, training frameworks, and deployment tools for real-time performance.</p>
            </div>
          </div>
        </div>
      </section>

      <section id="why-zen" className="architecture-section">
        <div className="container">
          <h2 className="section-title">Why Zen for XR/VR/Robotics?</h2>
          <div className="arch-grid">
            <div className="arch-card">
              <div className="arch-icon">⚡</div>
              <h3>Real-Time Performance</h3>
              <p>Sub-10ms latency with optimized quantization and edge deployment. Seamless integration with XR headsets and robotic control systems.</p>
            </div>
            <div className="arch-card">
              <div className="arch-icon">🌐</div>
              <h3>Spatial Awareness</h3>
              <p>Native 3D understanding, scene generation, and spatial audio processing. Built for immersive environments and physical world interaction.</p>
            </div>
            <div className="arch-card">
              <div className="arch-icon">🎯</div>
              <h3>Multimodal Fusion</h3>
              <p>Unified understanding across vision, language, audio, and 3D. Real-time gesture recognition, voice commands, and environmental awareness.</p>
            </div>
            <div className="arch-card">
              <div className="arch-icon">🔒</div>
              <h3>Open Source & Transparent</h3>
              <p>Fully open models, training code, and infrastructure. Complete control for customization, fine-tuning, and deployment on any platform.</p>
            </div>
          </div>
        </div>
      </section>

      <section id="downloads" className="downloads-section">
        <div className="container">
          <h2 className="section-title">Get Started</h2>
          <div className="download-grid">
            <div className="download-card">
              <h3>🤗 HuggingFace</h3>
              <p>Access all 24+ models via HuggingFace Hub with easy integration</p>
              <a href="https://huggingface.co/zenlm" className="btn btn-primary" target="_blank" rel="noopener noreferrer">
                Visit HuggingFace
              </a>
            </div>
            <div className="download-card">
              <h3>💻 GitHub</h3>
              <p>Training code, datasets, documentation, and complete source</p>
              <a href="https://github.com/zenlm" className="btn btn-primary" target="_blank" rel="noopener noreferrer">
                View on GitHub
              </a>
            </div>
            <div className="download-card">
              <h3>📚 Documentation</h3>
              <p>Comprehensive guides, papers, and tutorials for all models</p>
              <a href="https://github.com/zenlm/zen-blog" className="btn btn-primary" target="_blank" rel="noopener noreferrer">
                Read Docs
              </a>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}
