import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Video Translator',
  description: 'Translate and dub English IT videos into Chinese',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors">
        {children}
      </body>
    </html>
  );
}
