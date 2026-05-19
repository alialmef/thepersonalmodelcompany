import Link from "next/link";

export default function PrivacyPage() {
  return (
    <main className="min-h-screen px-7 py-16 max-w-[640px] mx-auto">
      <Link
        href="/"
        className="text-[12px] text-neutral-500 hover:text-neutral-900"
      >
        ← Back
      </Link>

      <h1 className="text-[32px] font-medium tracking-[-0.02em] text-neutral-900 mt-10 mb-6">
        Privacy.
      </h1>

      <div className="space-y-6 text-[15px] leading-relaxed text-neutral-700">
        <p>
          Your writing never leaves your Mac unless you train.
        </p>
        <p>
          Connecting a source (iMessage, Apple Notes, Apple Mail, WhatsApp,
          documents) reads the local data on your machine. Nothing is sent to
          our servers until you press <span className="text-neutral-900">Train</span>.
        </p>
        <p>
          When you do train, we send the curated examples (de-duplicated,
          PII-flagged, with your writing context) to our training infrastructure.
          The resulting LoRA adapter and a manifest of what was used to make it
          are stored against your account.
        </p>
        <p>
          We never train base models on your data. We never share it. We never
          read it ourselves — the pipeline is automated end to end.
        </p>
        <p>
          You can delete a source, delete a training run, or delete everything
          at any time. Deletion is immediate and your model retrains from what
          remains. The audit log shows every byte that touched our systems.
        </p>
        <p className="text-neutral-500 text-[13px] pt-8 border-t border-neutral-200">
          A formal policy will live here before launch.
        </p>
      </div>
    </main>
  );
}
