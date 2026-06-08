'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import Enso from './Enso';

type MenuItem = {
  label: string;
  description: string;
  href: string;
  tag?: string;
  external?: boolean;
};

type NavMenu = {
  label: string;
  href: string;
  items: MenuItem[];
};

const NAV_MENUS: NavMenu[] = [
  {
    label: 'Models',
    href: '/models',
    items: [
      { label: 'Zen5', tag: 'Latest', description: 'Frontier agentic — 256K–1M context, native tool use', href: '/models#zen5' },
      { label: 'Zen4', tag: 'Stable', description: 'Production chat & code — MoE flagships and coders', href: '/models#zen4' },
      { label: 'Zen3', description: 'Multimodal & specialty — vision, audio, image, safety', href: '/models#zen3' },
      { label: 'All models', description: 'Browse the full catalog — 95 open models', href: '/models' },
    ],
  },
  {
    label: 'Datasets',
    href: '/datasets',
    items: [
      { label: 'Zen Agentic Dataset', tag: 'New', description: '10B+ tokens of real tool use and reasoning', href: '/datasets' },
      { label: 'Training Data', description: 'Curation, sources, and methodology', href: '/datasets' },
      { label: 'On HuggingFace', description: 'Download and license the datasets', href: 'https://huggingface.co/datasets/hanzoai/zen-agentic-dataset', external: true },
    ],
  },
  {
    label: 'Research',
    href: '/research',
    items: [
      { label: 'Research Papers', description: 'Technical reports across the Zen family', href: '/research' },
      { label: 'Papers Archive', description: 'Full paper archive — papers.zenlm.org', href: 'https://papers.zenlm.org', external: true },
      { label: 'Blog', description: 'Releases and deep dives — blog.zenlm.org', href: 'https://blog.zenlm.org', external: true },
    ],
  },
];

const TRY_MENU: MenuItem[] = [
  { label: 'Zen5', tag: 'Latest', description: 'Chat with the frontier model — live now', href: 'https://hanzo.chat/?model=zen5', external: true },
  { label: 'Zen4', tag: 'Stable', description: 'Chat with the production family', href: 'https://hanzo.chat/?model=zen4', external: true },
  { label: 'Zen API', description: 'OpenAI- & Anthropic-compatible. One key.', href: 'https://api.hanzo.ai', external: true },
  { label: 'Zen Chat', description: 'Open the full chat experience', href: 'https://hanzo.chat', external: true },
];

const ArrowIcon = () => (
  <svg className="dd-arrow" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M5 12h14M13 6l6 6-6 6" />
  </svg>
);

function DropdownItem({ item }: { item: MenuItem }) {
  const inner = (
    <>
      <div className="dd-item-text">
        <span className="dd-item-label">
          {item.label}
          {item.tag && <span className="dd-tag">{item.tag}</span>}
        </span>
        <span className="dd-item-desc">{item.description}</span>
      </div>
      <ArrowIcon />
    </>
  );
  return item.external ? (
    <a className="dd-item" href={item.href} target="_blank" rel="noopener noreferrer">{inner}</a>
  ) : (
    <Link className="dd-item" href={item.href}>{inner}</Link>
  );
}

export default function Header() {
  const pathname = usePathname();

  const isActive = (path: string) => {
    if (path === '/') return pathname === '/';
    return pathname.startsWith(path);
  };

  return (
    <header>
      <nav className="navbar">
        <div className="container">
          <div className="logo">
            <Link href="/">
              <Enso size={34} className="logo-img" />
              <div className="logo-text">
                <h1>Zen LM</h1>
              </div>
            </Link>
          </div>
          <div className="header-right">
            <div className="nav-links">
              <Link href="/" className={isActive('/') && pathname === '/' ? 'active' : ''}>
                Home
              </Link>
              {NAV_MENUS.map((menu) => (
                <div className="nav-item" key={menu.label}>
                  <Link href={menu.href} className={isActive(menu.href) ? 'active' : ''}>
                    {menu.label}
                    <svg className="nav-caret" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                      <path d="M6 9l6 6 6-6" />
                    </svg>
                  </Link>
                  <div className="nav-dropdown">
                    <div className="nav-dropdown-inner">
                      {menu.items.map((item) => (
                        <DropdownItem key={item.label + item.href} item={item} />
                      ))}
                    </div>
                  </div>
                </div>
              ))}
            </div>
            <div className="logo-links">
              <a href="https://github.com/zenlm" target="_blank" rel="noopener noreferrer" className="icon-link" title="GitHub">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                </svg>
              </a>
              <a href="https://huggingface.co/zenlm" target="_blank" rel="noopener noreferrer" className="icon-link" title="HuggingFace">
                <svg width="28" height="28" viewBox="0 0 32 32" fill="currentColor" aria-hidden="true">
                  <circle cx="16" cy="16" r="13" fill="#d4d4d8"/>
                  <circle cx="11" cy="14" r="2" fill="#18181b"/>
                  <circle cx="21" cy="14" r="2" fill="#18181b"/>
                  <path d="M11 19 Q16 24 21 19" stroke="#18181b" strokeWidth="1.6" fill="none" strokeLinecap="round"/>
                </svg>
              </a>
            </div>
            <div className="nav-item try-item">
              <a
                href="https://hanzo.chat/?model=zen5"
                target="_blank"
                rel="noopener noreferrer"
                className="btn-try"
              >
                Try Zen
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <path d="M5 12h14M13 6l6 6-6 6" />
                </svg>
              </a>
              <div className="nav-dropdown nav-dropdown-right">
                <div className="nav-dropdown-inner">
                  {TRY_MENU.map((item) => (
                    <DropdownItem key={item.label + item.href} item={item} />
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </nav>
    </header>
  );
}
