import Link from "next/link";

export default function OtherPlatformsPage() {
  return (
    <main className="min-h-screen flex items-center justify-center px-7">
      <div className="max-w-md text-center">
        <h1 className="text-[26px] font-medium tracking-[-0.02em] text-neutral-900 mb-3">
          Mac first.
        </h1>
        <p className="text-[14px] text-neutral-500 leading-relaxed mb-9 max-w-[420px] mx-auto">
          PMC reads your iMessage, Apple Notes, and Apple Mail directly from
          your machine. That only works on macOS today. Windows and Linux are
          on the roadmap once the Mac app is stable.
        </p>
        <Link
          href="/"
          className="text-[13px] text-neutral-500 hover:text-neutral-900"
        >
          ← Back
        </Link>
      </div>
    </main>
  );
}
