import type { Metadata } from 'next';
import Header from '@/components/Header';
import Footer from '@/components/Footer';
import './globals.css';

export const metadata: Metadata = {
  title: 'Zen LM - Real-Time AI for XR/VR/Robotics',
  description: 'Real-Time Hyper-Modal AI for XR/VR/Robotics',
  keywords: 'AI, XR, VR, Robotics, Language Models, Multimodal AI, 3D, Spatial Computing',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link rel="icon" type="image/png" href="/favicon.png" />
      </head>
      <body>
        <Header />
        {children}
        <Footer />
        <script src="/assets/js/main.js" async></script>
      </body>
    </html>
  );
}
