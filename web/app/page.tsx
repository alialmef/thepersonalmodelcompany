import type { Metadata } from 'next';
import {
  Hero,
  HowItWorks,
  Folder,
  Privacy,
  Closer,
  Footer,
} from '@/components/landing/landing-sections';

export const metadata: Metadata = {
  title: 'The Personal Model Company',
  description: 'Your personal model. Trained on you. Yours to keep.',
};

export default function LandingPage() {
  return (
    <main>
      <Hero />
      <HowItWorks />
      <Folder />
      <Privacy />
      <Closer />
      <Footer />
    </main>
  );
}
