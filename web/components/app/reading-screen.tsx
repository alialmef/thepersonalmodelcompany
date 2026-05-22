"use client";

import { useEffect, useState } from "react";

import { BrandMark } from "@/components/shared/brand-mark";

/**
 * /reading — the screen after /connect where the system shows its work
 * piece by piece. Two headers anchor the page ("Constructing your
 * voice." / "Structuring your memory."). Both headers appear
 * immediately so the user has something to read while the backend
 * spins up. Under each, a soft pulse indicates work in progress;
 * items type in as `reading_source_found` audit events arrive.
 *
 * Design principles:
 *   * Headers visible from t=0, even before backend events.
 *   * Counts shown are *grounded* — "the people you actually talk
 *     to", not "every handle the system found." Backend filters
 *     before emitting.
 *   * No progress bars, no scoreboards.
 *   * Lots of white space, centered, typed prose.
 */

export interface ReadingItem {
  bucket: "voice" | "memory";
  kind: string;
  count: number;
  phrase: string;
}

export interface ReadingScreenProps {
  items: ReadingItem[];
  ready: boolean;
  onContinue: () => void;
}

const HEADINGS = {
  voice: "Constructing your voice.",
  memory: "Structuring your memory.",
} as const;

function countLabel(n: number, compact = false): string {
  // Voice items get K-style compact format: 27,474 → "27k".
  // Memory items stay numeric since the grounded counts are small
  // and exact precision matters there ("23 people" not "0.02k").
  if (compact && n >= 1000) {
    const k = n / 1000;
    return k >= 100 ? `${Math.round(k)}k` : `${k.toFixed(k >= 10 ? 0 : 1)}k`;
  }
  if (n >= 1000) return n.toLocaleString();
  return `${n}`;
}

function PulseDot() {
  return (
    <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-foreground/40" />
  );
}

function ReadingItemRow({ item, delay }: { item: ReadingItem; delay: number }) {
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setVisible(true), delay);
    return () => clearTimeout(t);
  }, [delay]);

  // Voice: "27k messages" — terse.
  // Memory: "163 people you actually talk to: Dad, Aisha, Ben." —
  //         the count + grounded phrase with names baked in by the
  //         backend so the user sees real recognizable signals
  //         rather than abstract aggregates.
  const isVoice = item.bucket === "voice";
  return (
    <div
      className={`transition-opacity duration-700 ease-out ${visible ? "opacity-100" : "opacity-0"}`}
    >
      <div className="text-[0.95rem] leading-relaxed text-foreground/85">
        <span className="font-medium tabular-nums">
          {countLabel(item.count, isVoice)}
        </span>
        <span className="text-foreground/70"> {item.phrase}</span>
      </div>
    </div>
  );
}

function Bucket({
  bucket,
  items,
  headerDelay,
  itemBaseDelay,
}: {
  bucket: "voice" | "memory";
  items: ReadingItem[];
  headerDelay: number;
  itemBaseDelay: number;
}) {
  const [headerVisible, setHeaderVisible] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setHeaderVisible(true), headerDelay);
    return () => clearTimeout(t);
  }, [headerDelay]);
  // Once the FIRST item for this bucket arrives, stop pulsing.
  const [pulseGone, setPulseGone] = useState(false);
  useEffect(() => {
    if (items.length > 0) {
      const t = setTimeout(() => setPulseGone(true), itemBaseDelay + 400);
      return () => clearTimeout(t);
    }
  }, [items.length, itemBaseDelay]);

  return (
    <div className="space-y-7">
      <div
        className={`text-xl font-semibold text-foreground transition-opacity duration-700 ${
          headerVisible ? "opacity-100" : "opacity-0"
        }`}
      >
        {HEADINGS[bucket]}
      </div>
      <div className="space-y-5 pl-1">
        {items.map((item, i) => (
          <ReadingItemRow
            key={`${bucket}-${item.kind}`}
            item={item}
            delay={itemBaseDelay + i * 700}
          />
        ))}
        {/* Pulse stays until first item arrives, then fades out */}
        {headerVisible && !pulseGone && (
          <div
            className={`pt-1 transition-opacity duration-700 ${
              items.length > 0 ? "opacity-0" : "opacity-100"
            }`}
          >
            <PulseDot />
          </div>
        )}
      </div>
    </div>
  );
}

export default function ReadingScreen({
  items,
  ready,
  onContinue,
}: ReadingScreenProps) {
  const voice = items.filter((i) => i.bucket === "voice");
  const memory = items.filter((i) => i.bucket === "memory");

  return (
    <main className="min-h-screen w-full bg-background text-foreground">
      <div className="mx-auto flex min-h-screen max-w-2xl flex-col px-8 pb-32 pt-16">
        <div className="mb-20">
          <BrandMark />
        </div>

        <div className="space-y-16">
          {/* Both headers visible from the start. Voice first (immediate),
              memory ~1.5s later so the reader's eye lands on voice. */}
          <Bucket
            bucket="voice"
            items={voice}
            headerDelay={300}
            itemBaseDelay={900}
          />
          <Bucket
            bucket="memory"
            items={memory}
            headerDelay={1800}
            itemBaseDelay={2400}
          />
        </div>

        <div className="mt-auto pt-24">
          <button
            type="button"
            onClick={onContinue}
            disabled={!ready}
            className={`text-base transition-opacity duration-700 ${
              ready
                ? "cursor-pointer text-foreground/80 hover:text-foreground opacity-100"
                : "cursor-default text-foreground/30 opacity-50"
            }`}
          >
            {ready ? "Continue" : "Still reading…"}
          </button>
        </div>
      </div>
    </main>
  );
}
