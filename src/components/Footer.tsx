import Link from 'next/link';

export default function Footer() {
  return (
    <footer>
      <div className="container">
        <div className="footer-content">
          <div className="footer-section">
            <h4>Zen LM</h4>
            <p>Real-time hyper-modal AI for XR/VR/Robotics. From language to multimodal, nano to next-gen.</p>
          </div>
          <div className="footer-section">
            <h4>Core Models</h4>
            <ul>
              <li><Link href="/models#core-models">zen-nano (0.6B)</Link></li>
              <li><Link href="/models#core-models">zen-eco (4B)</Link></li>
              <li><Link href="/models#core-models">zen-omni (7B)</Link></li>
              <li><Link href="/models#core-models">zen-coder (14B)</Link></li>
              <li><Link href="/models#core-models">zen-next (32B)</Link></li>
            </ul>
          </div>
          <div className="footer-section">
            <h4>Multimodal</h4>
            <ul>
              <li><Link href="/models#multimodal">zen-designer (Vision)</Link></li>
              <li><Link href="/models#multimodal">zen-artist (Image)</Link></li>
              <li><Link href="/models#multimodal">zen-video (Video)</Link></li>
              <li><Link href="/models#multimodal">zen-3d (3D Gen)</Link></li>
              <li><Link href="/models#multimodal">zen-musician (Music)</Link></li>
            </ul>
          </div>
          <div className="footer-section">
            <h4>Resources</h4>
            <ul>
              <li><a href="https://huggingface.co/zenlm" target="_blank" rel="noopener noreferrer">HuggingFace</a></li>
              <li><a href="https://github.com/zenlm" target="_blank" rel="noopener noreferrer">GitHub</a></li>
              <li><Link href="/research">Research Papers</Link></li>
              <li><a href="https://github.com/QwenLM/Qwen3" target="_blank" rel="noopener noreferrer">Qwen3</a></li>
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
