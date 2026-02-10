import Link from 'next/link';

export default function Footer() {
  return (
    <footer>
      <div className="container">
        <div className="footer-content">
          <div className="footer-section">
            <h4>Zen LM</h4>
            <p>Open foundation models for agentic AI. 30+ models from 0.6B to 1T parameters.</p>
          </div>
          <div className="footer-section">
            <h4>Zen Coder</h4>
            <ul>
              <li><Link href="/models#zen-coder">Zen Coder 4B</Link></li>
              <li><Link href="/models#zen-coder">Zen Coder 24B</Link></li>
              <li><Link href="/models#zen-coder">Zen Coder 123B</Link></li>
              <li><Link href="/models#zen-coder">Zen Coder Max (358B)</Link></li>
              <li><Link href="/models#zen-coder">Zen Coder Ultra (1T)</Link></li>
            </ul>
          </div>
          <div className="footer-section">
            <h4>Model Family</h4>
            <ul>
              <li><Link href="/models#core-models">zen-nano (0.6B)</Link></li>
              <li><Link href="/models#core-models">zen-eco (4B)</Link></li>
              <li><Link href="/models#core-models">zen-omni (7B)</Link></li>
              <li><Link href="/models#multimodal">zen-vl (Vision)</Link></li>
              <li><Link href="/models#multimodal">zen-3d (3D Gen)</Link></li>
            </ul>
          </div>
          <div className="footer-section">
            <h4>Resources</h4>
            <ul>
              <li><Link href="/datasets">Training Data</Link></li>
              <li><a href="https://huggingface.co/zenlm" target="_blank" rel="noopener noreferrer">HuggingFace</a></li>
              <li><a href="https://github.com/zenlm" target="_blank" rel="noopener noreferrer">GitHub</a></li>
              <li><Link href="/research">Research Papers</Link></li>
              <li><a href="https://github.com/zenlm/zen-trainer" target="_blank" rel="noopener noreferrer">zen-trainer</a></li>
            </ul>
          </div>
        </div>
        <div className="footer-bottom">
          <p>&copy; 2025 Zen Authors. All rights reserved. Built with clarity and purpose.</p>
        </div>
      </div>
    </footer>
  );
}
