import Link from "next/link";

export default function ContactPage() {
  return (
    <main className="min-h-screen flex items-center justify-center px-7">
      <div className="max-w-md text-center">
        <h1 className="text-[26px] font-medium tracking-[-0.02em] text-neutral-900 mb-3">
          Contact.
        </h1>
        <p className="text-[14px] text-neutral-500 leading-relaxed mb-6">
          <a
            href="mailto:hello@thepersonalmodelcompany.com"
            className="text-neutral-900 underline"
          >
            hello@thepersonalmodelcompany.com
          </a>
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
