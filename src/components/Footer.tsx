import Link from 'next/link';

export default function Footer() {
  return (
    <footer>
      <div className="container">
        <div className="footer-content">
          <div className="footer-section">
            <h4>Zen LM</h4>
            <p>
              Open models from Zoo Labs Foundation, a 501(c)(3) non-profit, for agentic coding on your own machine
              and marketing work. Served on api.hanzo.ai.
            </p>
          </div>
          <div className="footer-section">
            <h4>Zen 6</h4>
            <ul>
              <li><Link href="/models#zen6">Zen 6</Link></li>
              <li><Link href="/models#zen6">Zen 6 Flash</Link></li>
              <li><a href="https://hanzo.ai/research-access" target="_blank" rel="noopener noreferrer">Zen 7 research preview</a></li>
            </ul>
          </div>
          <div className="footer-section">
            <h4>Earlier Generations</h4>
            <ul>
              <li><Link href="/models#zen5">Zen 5</Link></li>
              <li><Link href="/models#zen4">Zen 4</Link></li>
              <li><Link href="/models#zen3">Zen 3</Link></li>
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
          <p>&copy; {new Date().getFullYear()} Zoo Labs Foundation, a 501(c)(3) non-profit. Zen LM open models.</p>
        </div>
      </div>
    </footer>
  );
}
