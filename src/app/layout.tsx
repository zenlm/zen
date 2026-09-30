import type { Metadata } from 'next';
import Header from '@/components/Header';
import Footer from '@/components/Footer';
import ScrollReveal from '@/components/ScrollReveal';
import './globals.css';

export const metadata: Metadata = {
  title: 'Zen LM - Open Foundation Models for Agentic AI',
  description: 'Open models from Zoo Labs Foundation, a 501(c)(3) non-profit, for agentic coding on your own machine and marketing work. Zen 6 and Zen 6 Flash are available now.',
  keywords: 'AI, LLM, Agentic AI, Code Generation, Zen Coder, Multimodal, Open Source, Machine Learning',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link rel="icon" type="image/svg+xml" href="/favicon.svg" />
        <link rel="alternate icon" href="/favicon.png" />
      </head>
      <body>
        <Header />
        {children}
        <Footer />
        <ScrollReveal />
        <script src="/assets/js/main.js" async></script>
      </body>
    </html>
  );
}
