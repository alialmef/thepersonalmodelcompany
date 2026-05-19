'use client';

import {
  Mail,
  Notebook,
  FileText,
  MessageSquare,
  MessagesSquare,
  Search,
  Code,
  Wrench,
} from 'lucide-react';
import { useInView } from '@/hooks/use-in-view';

// ---------------------------------------------------------------------------
// Letter cascade — used wherever text "arrives" word-by-word, matching the
// model's token-streaming behavior. The hero headline uses this to materialize
// in sync with the brand mark's fusion event.
// ---------------------------------------------------------------------------
function LetterCascade({
  text,
  startMs = 0,
  perLetterMs = 35,
}: {
  text: string;
  startMs?: number;
  perLetterMs?: number;
}) {
  return (
    <>
      {[...text].map((char, i) => (
        <span
          key={i}
          className="pmc-letter"
          style={{ animationDelay: `${startMs + i * perLetterMs}ms` }}
        >
          {char === ' ' ? '\u00A0' : char}
        </span>
      ))}
    </>
  );
}

// ---------------------------------------------------------------------------
// Hero — dark, choreographed on page load.
//
// Timing (all CSS keyframes, see globals.css):
//   0.1s  nav fades in
//   0.2s  mark circles fade in (still separated)
//   0.8s  drift toward center begins (2.0s, cubic-bezier(0.65,0,0.35,1))
//   2.5s  flash pulse fires (0.75s)
//   2.8s  fusion complete; headline letters begin cascade (35ms stagger)
//   4.0s  sub-line fades in
//   4.5s  CTA fades in
//
// Production note: replace the SVG fusion choreography with the iridescent
// brand video once available. The SVG below remains as a lightweight
// pre-buffer fallback.
// ---------------------------------------------------------------------------
export function Hero() {
  return (
    <section className="bg-white text-black min-h-screen flex flex-col">
      <nav
        className="flex items-center justify-between px-7 py-[18px] pmc-anim-fade"
        style={{ animationDelay: '0.1s' }}
      >
        <div className="text-[13px] font-medium tracking-tight">
          The Personal Model Company
        </div>
        <div className="flex gap-[22px] text-[12px] text-neutral-500">
          <a href="/privacy">Privacy</a>
          <a href="/sign-in">Sign in</a>
        </div>
      </nav>

      <div className="flex-1 flex flex-col items-center justify-center px-7 text-center">
        <div className="mb-2 flex justify-center">
          <svg viewBox="0 0 280 180" width="640" height="420" overflow="visible" aria-hidden="true">
            <circle
              cx="140"
              cy="90"
              r="48"
              fill="#000000"
              className="pmc-hero-flash"
            />
            <circle
              cx="140"
              cy="90"
              r="48"
              fill="none"
              stroke="#000000"
              strokeWidth="0.75"
              className="pmc-hero-c1"
            />
            <circle
              cx="140"
              cy="90"
              r="48"
              fill="none"
              stroke="#000000"
              strokeWidth="0.75"
              className="pmc-hero-c2"
            />
            {/* Bullseye dot — appears after fusion settles */}
            <circle
              cx="140"
              cy="90"
              r="6"
              fill="#DC2626"
              className="pmc-hero-dot"
            />
          </svg>
        </div>

        <h1 className="max-w-[680px] text-[56px] md:text-[72px] font-medium leading-[1.04] tracking-[-0.035em] text-black mb-5">
          <LetterCascade text="build your ai model" startMs={2800} />
        </h1>

        <p
          className="mb-12 text-[19px] md:text-[21px] text-neutral-500 pmc-anim-fade"
          style={{ animationDelay: '4.0s' }}
        >
          made from your data. owned by you.
        </p>

        <div className="pmc-anim-fade" style={{ animationDelay: '4.5s' }}>
          <a
            href="/downloads/PersonalModelCompany.dmg"
            download="PersonalModelCompany.dmg"
            className="inline-block rounded-full bg-black px-[28px] py-[14px] text-[15px] font-medium text-white"
          >
            Download for Mac
          </a>
        </div>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// How it works — dark, choreographed when scrolled into view.
//
// Timing (relative to in-view trigger):
//   0.2s  title fades in
//   0.7s  left icons stagger in (100ms apart)
//   1.1s  left lines draw in via stroke-dashoffset (1.0s)
//   1.8s  center mark grows in (0.6s)
//   2.1s  left-to-center particles fire (SMIL, 1.0s, freeze at end)
//   2.4s  mark label fades in
//   3.1s  right lines draw in
//   3.3s  right icons stagger in
//   3.7s  center-to-right particles fire
//   5.0s  tagline fades in
//
// The mark in this section is a SINGLE circle — fusion already happened in
// the hero; here the model exists and acts.
// ---------------------------------------------------------------------------
export function HowItWorks() {
  const [ref, inView] = useInView<HTMLDivElement>(0.15);

  return (
    <section
      ref={ref}
      className={`bg-black min-h-screen flex flex-col items-center justify-center px-7 py-20 ${
        inView ? 'pmc-in-view' : ''
      }`}
    >
      <h2 className="mb-14 text-center text-[24px] font-medium tracking-[-0.02em] text-[#F5F5F7] pmc-hiw-title">
        How it works.
      </h2>

      <div className="relative mx-auto h-[440px] w-full max-w-[920px]">
        <svg
          viewBox="0 0 620 280"
          className="absolute inset-0 h-full w-full"
          aria-hidden="true"
        >
          {/* Left ingest lines — drawn one at a time (cascading) */}
          <path
            id="hiw-lp1"
            d="M 90 50 C 180 50, 250 140, 310 140"
            fill="none"
            stroke="#5a5a5c"
            strokeWidth="0.75"
            className="pmc-hiw-l-line pmc-hiw-l-line-1"
          />
          <path
            id="hiw-lp2"
            d="M 90 100 C 200 100, 270 140, 310 140"
            fill="none"
            stroke="#5a5a5c"
            strokeWidth="0.75"
            className="pmc-hiw-l-line pmc-hiw-l-line-2"
          />
          <path
            id="hiw-lp3"
            d="M 90 180 C 200 180, 270 140, 310 140"
            fill="none"
            stroke="#5a5a5c"
            strokeWidth="0.75"
            className="pmc-hiw-l-line pmc-hiw-l-line-3"
          />
          <path
            id="hiw-lp4"
            d="M 90 230 C 180 230, 250 140, 310 140"
            fill="none"
            stroke="#5a5a5c"
            strokeWidth="0.75"
            className="pmc-hiw-l-line pmc-hiw-l-line-4"
          />
          {/* Right connector lines */}
          <path
            id="hiw-rp1"
            d="M 310 140 C 370 140, 440 50, 530 50"
            fill="none"
            stroke="#5a5a5c"
            strokeWidth="0.75"
            className="pmc-hiw-r-line"
          />
          <path
            id="hiw-rp2"
            d="M 310 140 C 350 140, 420 100, 530 100"
            fill="none"
            stroke="#5a5a5c"
            strokeWidth="0.75"
            className="pmc-hiw-r-line"
          />
          <path
            id="hiw-rp3"
            d="M 310 140 C 350 140, 420 180, 530 180"
            fill="none"
            stroke="#5a5a5c"
            strokeWidth="0.75"
            className="pmc-hiw-r-line"
          />
          <path
            id="hiw-rp4"
            d="M 310 140 C 370 140, 440 230, 530 230"
            fill="none"
            stroke="#5a5a5c"
            strokeWidth="0.75"
            className="pmc-hiw-r-line"
          />

          {/*
            Particles: SMIL animateMotion. Modern browsers support this fine
            but it's deprecated long-term. Optional upgrade: replace with
            CSS offset-path or a Framer Motion / GSAP implementation.

            The `begin` attribute uses indefinite + a trigger from JS to make
            these scroll-triggered. See the useEffect at the bottom that
            calls beginElement() when inView becomes true.
          */}
          {inView &&
            ['hiw-lp1', 'hiw-lp2', 'hiw-lp3', 'hiw-lp4'].map((id, i) => {
              // Each particle fires the moment its own line finishes drawing.
              // Line i starts at (1.0 + i*0.7)s and draws for 0.5s, so the
              // particle begins at (1.5 + i*0.7)s. 0.7s cadence gives the
              // viewer a beat to register each source label.
              const begin = `${(1.5 + i * 0.7).toFixed(2)}s`;
              return (
                <circle key={id} r="1.5" fill="#F5F5F7" opacity="0">
                  <set attributeName="opacity" to="1" begin={begin} />
                  <animateMotion dur="1.0s" begin={begin} fill="freeze">
                    <mpath href={`#${id}`} />
                  </animateMotion>
                </circle>
              );
            })}
          {inView &&
            ['hiw-rp1', 'hiw-rp2', 'hiw-rp3', 'hiw-rp4'].map((id) => (
              <circle key={id} r="1.5" fill="#F5F5F7" opacity="0">
                <set attributeName="opacity" to="1" begin="6.1s" />
                <animateMotion
                  dur="1.0s"
                  begin="6.1s"
                  fill="freeze"
                >
                  <mpath href={`#${id}`} />
                </animateMotion>
              </circle>
            ))}

          {/* Center mark — single circle, post-fusion */}
          <circle
            cx="310"
            cy="140"
            r="18"
            fill="none"
            stroke="#F5F5F7"
            strokeWidth="0.75"
            className="pmc-hiw-mark"
          />
          {/* Bullseye dot — settles after the center mark grows in */}
          <circle
            cx="310"
            cy="140"
            r="2.5"
            fill="#DC2626"
            className="pmc-hiw-dot"
          />
          <text
            x="310"
            y="190"
            textAnchor="middle"
            fill="#86868B"
            fontSize="11"
            className="pmc-hiw-mark-label"
          >
            your model
          </text>
        </svg>

        {/* Left sources cascade in one at a time — each icon arrives, then
            its line draws, then its particle flows. 0.7s cadence between
            sources gives the viewer time to register each label. */}
        <div className="pmc-hiw-l-node absolute left-2 top-[15%]" style={{ animationDelay: '0.8s' }}>
          <Node icon={<MessageSquare />} label="texts" />
        </div>
        <div className="pmc-hiw-l-node absolute left-2 top-[33%]" style={{ animationDelay: '1.5s' }}>
          <Node icon={<Notebook />} label="notes" />
        </div>
        <div className="pmc-hiw-l-node absolute left-2 top-[61%]" style={{ animationDelay: '2.2s' }}>
          <Node icon={<FileText />} label="documents" />
        </div>
        <div className="pmc-hiw-l-node absolute left-2 top-[79%]" style={{ animationDelay: '2.9s' }}>
          <Node icon={<Mail />} label="mail" />
        </div>

        {/* Right destinations all arrive together — the model emits to every
            surface at once, not one at a time. */}
        <div className="pmc-hiw-r-node absolute right-2 top-[15%]" style={{ animationDelay: '5.4s' }}>
          <Node icon={<MessagesSquare />} label="chat" reverse />
        </div>
        <div className="pmc-hiw-r-node absolute right-2 top-[33%]" style={{ animationDelay: '5.4s' }}>
          <Node icon={<Search />} label="search" reverse />
        </div>
        <div className="pmc-hiw-r-node absolute right-2 top-[61%]" style={{ animationDelay: '5.4s' }}>
          <Node icon={<Code />} label="code" reverse />
        </div>
        <div className="pmc-hiw-r-node absolute right-2 top-[79%]" style={{ animationDelay: '5.4s' }}>
          <Node icon={<Wrench />} label="build" reverse />
        </div>
      </div>

      <p className="mt-14 text-center text-[14px] font-medium text-[#F5F5F7] pmc-hiw-tag">
        Trained on what you write, tuned for what you do.
      </p>
    </section>
  );
}

function Node({
  icon,
  label,
  reverse = false,
}: {
  icon: React.ReactNode;
  label: string;
  reverse?: boolean;
}) {
  return (
    <div
      className={`flex items-center gap-2.5 text-[14px] text-[#86868B] transition-colors hover:text-[#F5F5F7] ${
        reverse ? 'flex-row-reverse' : ''
      }`}
    >
      <span className="[&>svg]:h-[18px] [&>svg]:w-[18px] [&>svg]:stroke-[1.5]">
        {icon}
      </span>
      <span>{label}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Folder — light section. The lab metaphor's payload made literal.
// File list fades in line-by-line on scroll-into-view.
// ---------------------------------------------------------------------------
export function Folder() {
  const [ref, inView] = useInView<HTMLDivElement>(0.15);
  const files = [
    'adapter.safetensors',
    'style_profile.json',
    'manifest.toml',
    'eval_report.md',
    'audit.log',
  ];

  return (
    <section
      ref={ref}
      className={`bg-white min-h-screen flex flex-col items-center justify-center px-7 py-20 ${
        inView ? 'pmc-in-view' : ''
      }`}
    >
      <h2 className="mb-2 text-center text-[40px] md:text-[52px] font-medium tracking-[-0.03em] text-neutral-900">
        A lab in a folder.
      </h2>
      <p className="mb-14 text-center text-[19px] md:text-[21px] text-neutral-500">
        Yours to export and run anywhere.
      </p>

      <div className="w-full max-w-[460px] rounded-lg border-[0.5px] border-neutral-200 px-[26px] py-[22px] font-mono text-[14px]">
        <div className="mb-2 text-neutral-500">your_model/</div>
        <div className="flex flex-col gap-[5px] pl-[18px] text-neutral-900">
          {files.map((file, i) => (
            <div
              key={file}
              className="pmc-folder-line"
              style={{ animationDelay: `${0.2 + i * 0.1}s` }}
            >
              {file}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Privacy — light tertiary. Single declarative sentence + supporting line.
// ---------------------------------------------------------------------------
export function Privacy() {
  const [ref, inView] = useInView<HTMLDivElement>(0.15);

  return (
    <section
      ref={ref}
      className={`bg-black min-h-screen flex flex-col items-center justify-center px-7 py-20 text-center ${
        inView ? 'pmc-in-view' : ''
      }`}
    >
      <h2 className="max-w-[800px] text-[40px] md:text-[60px] font-medium leading-[1.05] tracking-[-0.03em] text-white pmc-fade-up">
        Your writing never leaves your Mac.
      </h2>
      <p
        className="mx-auto mt-6 max-w-[520px] text-[18px] md:text-[20px] leading-relaxed text-neutral-400 pmc-fade-up"
        style={{ animationDelay: '0.2s' }}
      >
        Read locally and trained privately, with every byte in the audit log.
      </p>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Closer — light. Static single-circle mark + final CTA.
// ---------------------------------------------------------------------------
export function Closer() {
  const [ref, inView] = useInView<HTMLDivElement>(0.15);

  return (
    <section
      ref={ref}
      className={`bg-white min-h-screen flex flex-col items-center justify-center px-7 py-20 text-center ${
        inView ? 'pmc-in-view' : ''
      }`}
    >
      <svg
        viewBox="0 0 80 80"
        width="72"
        height="72"
        className="mx-auto mb-10 block pmc-closer-mark"
        aria-hidden="true"
      >
        <circle
          cx="40"
          cy="40"
          r="30"
          fill="none"
          stroke="currentColor"
          strokeWidth="0.75"
          className="text-black"
        />
        {/* Bullseye dot — settles after the mark fades in */}
        <circle
          cx="40"
          cy="40"
          r="4"
          fill="#DC2626"
          className="pmc-closer-dot"
        />
      </svg>
      <h2
        className="mb-10 text-[44px] md:text-[60px] font-medium leading-[1.04] tracking-[-0.03em] text-black pmc-fade-up"
        style={{ animationDelay: '0.15s' }}
      >
        Make one of your own.
      </h2>
      <a
        href="/downloads/PersonalModelCompany.dmg"
            download="PersonalModelCompany.dmg"
        className="inline-block rounded-full bg-black px-[28px] py-[14px] text-[15px] font-medium text-white pmc-fade-up"
        style={{ animationDelay: '0.3s' }}
      >
        Download for Mac
      </a>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Footer — minimal.
// ---------------------------------------------------------------------------
export function Footer() {
  return (
    <footer className="flex items-center justify-between bg-black px-7 py-[22px] text-[11px] text-neutral-500">
      <div>© The Personal Model Company</div>
      <div className="flex gap-[14px]">
        <a href="/privacy" className="hover:text-white transition-colors">Privacy</a>
        <a href="/terms" className="hover:text-white transition-colors">Terms</a>
        <a href="/contact" className="hover:text-white transition-colors">Contact</a>
      </div>
    </footer>
  );
}
