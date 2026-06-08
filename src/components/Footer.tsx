import Link from 'next/link';

export default function Footer() {
  return (
    <footer>
      <div className="container">
        <div className="footer-content">
          <div className="footer-section">
            <h4>Zen LM</h4>
            <p>95 open Zen models across Zen3, Zen4, and Zen5. Chat, code, vision, audio, image, embeddings, rerankers, and safety. OpenAI- and Anthropic-compatible API.</p>
          </div>
          <div className="footer-section">
            <h4>Zen 5</h4>
            <ul>
              <li><Link href="/models#zen5">Zen5 Nano (0.8B - 9B)</Link></li>
              <li><Link href="/models#zen5">Zen5 Flash</Link></li>
              <li><Link href="/models#zen5">Zen5 Mini</Link></li>
              <li><Link href="/models#zen5">Zen5 (default)</Link></li>
              <li><Link href="/models#zen5">Zen5 Coder</Link></li>
              <li><Link href="/models#zen5">Zen5 Pro</Link></li>
              <li><Link href="/models#zen5">Zen5 Max</Link></li>
            </ul>
          </div>
          <div className="footer-section">
            <h4>Zen 4</h4>
            <ul>
              <li><Link href="/models#zen4">Zen4 / Zen4.1</Link></li>
              <li><Link href="/models#zen4">Zen4 Ultra / Max / Pro</Link></li>
              <li><Link href="/models#zen4">Zen4 Mini / Thinking</Link></li>
              <li><Link href="/models#zen4">Zen4 Coder / Pro / Flash</Link></li>
            </ul>
          </div>
          <div className="footer-section">
            <h4>Zen 3 Multimodal</h4>
            <ul>
              <li><Link href="/models#zen3">Zen3 Omni / VL / Web</Link></li>
              <li><Link href="/models#zen3">Zen3 Nano / Guard</Link></li>
              <li><Link href="/models#zen3">Zen3 Embedding / Reranker</Link></li>
              <li><Link href="/models#zen3">Zen3 Image / ASR / TTS</Link></li>
            </ul>
          </div>
          <div className="footer-section">
            <h4>Resources</h4>
            <ul>
              <li><Link href="/datasets">Training Data</Link></li>
              <li><a href="https://huggingface.co/zenlm" target="_blank" rel="noopener noreferrer">HuggingFace</a></li>
              <li><a href="https://github.com/zenlm" target="_blank" rel="noopener noreferrer">GitHub</a></li>
              <li><Link href="/research">Research Papers</Link></li>
              <li><a href="https://api.hanzo.ai" target="_blank" rel="noopener noreferrer">Zen API</a></li>
            </ul>
          </div>
        </div>
        <div className="footer-bottom">
          <p>&copy; {new Date().getFullYear()} Zen Authors. Open foundation models. Served on the Zen API.</p>
        </div>
      </div>
    </footer>
  );
}
