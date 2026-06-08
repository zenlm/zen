'use client';

import { useEffect } from 'react';

/**
 * Progressive enhancement: fade-up content as it scrolls into view.
 * Adds `js-reveal` to <html> so the hiding CSS only applies when JS runs
 * (no flash-of-hidden-content if JS is disabled or fails). The hero has its
 * own load entrance, so it's excluded here.
 */
export default function ScrollReveal() {
  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const root = document.documentElement;

    const targets = Array.from(
      document.querySelectorAll<HTMLElement>(
        [
          '.section-title',
          '.section-subtitle',
          '.arch-card',
          '.download-card',
          '.model-card',
          '.cat-group',
          '.blog-card',
          '.model-lineup',
          '.paper-grid',
        ].join(',')
      )
    );
    if (!targets.length) return;

    root.classList.add('js-reveal');

    targets.forEach((el) => {
      el.classList.add('reveal-item');
      // Stagger by position among same-typed siblings so rows cascade in.
      const siblings = el.parentElement ? Array.from(el.parentElement.children).filter((c) => c.className === el.className) : [];
      const idx = Math.max(0, siblings.indexOf(el));
      el.style.transitionDelay = `${Math.min(idx, 5) * 55}ms`;
    });

    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) {
            e.target.classList.add('in-view');
            io.unobserve(e.target);
          }
        }
      },
      { rootMargin: '0px 0px -8% 0px', threshold: 0.08 }
    );
    targets.forEach((el) => io.observe(el));

    return () => io.disconnect();
  }, []);

  return null;
}
