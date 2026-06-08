'use client';

import { useId } from 'react';

interface EnsoProps {
  size?: number;
  className?: string;
}

/**
 * Zen ensō (円相) — a hand-brushed open circle that draws itself in on mount
 * via a stroke-dash animation. The single tapered gap sits at the upper right,
 * the traditional lift point of the brush.
 *
 * The gradient id is generated with useId() so the component can be rendered
 * any number of times on a page (nav, sidebar, hero, mobile menu…) without
 * colliding ids — a collision where the first def lands in a display:none
 * container makes the stroke vanish in Chrome.
 */
export default function Enso({ size = 40, className = '' }: EnsoProps) {
  const gradientId = useId();
  return (
    <svg
      className={`enso ${className}`.trim()}
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 100 100"
      width={size}
      height={size}
      fill="none"
      role="img"
      aria-label="Zen ensō"
    >
      <defs>
        <linearGradient id={gradientId} x1="12%" y1="0%" x2="88%" y2="100%">
          <stop offset="0%" stopColor="#ffffff" />
          <stop offset="50%" stopColor="#d4d4d8" />
          <stop offset="100%" stopColor="#7c7c82" />
        </linearGradient>
      </defs>
      <path
        className="enso-stroke"
        pathLength={1}
        d="M58 13 C38 9 17 22 13 44 C9 68 26 89 50 90 C74 91 91 74 89 50 C88 41 87 35 84 30"
        stroke={`url(#${gradientId})`}
        strokeWidth={7}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
