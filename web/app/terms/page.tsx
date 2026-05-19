import Link from "next/link";

export default function TermsPage() {
  return (
    <main className="min-h-screen px-7 py-16 max-w-[640px] mx-auto">
      <Link
        href="/"
        className="text-[12px] text-neutral-500 hover:text-neutral-900"
      >
        ← Back
      </Link>
      <h1 className="text-[32px] font-medium tracking-[-0.02em] text-neutral-900 mt-10 mb-6">
        Terms.
      </h1>
      <p className="text-[15px] leading-relaxed text-neutral-700">
        Formal terms will live here before launch.
      </p>
    </main>
  );
}
