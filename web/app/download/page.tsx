"use client";

import { useEffect } from "react";
import Link from "next/link";

const DMG_PATH = "/downloads/PersonalModelCompany.dmg";

export default function DownloadPage() {
  // Auto-trigger the download on mount. Most browsers respect the navigation
  // and start the download without leaving the page.
  useEffect(() => {
    const a = document.createElement("a");
    a.href = DMG_PATH;
    a.download = "PersonalModelCompany.dmg";
    a.rel = "noopener noreferrer";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }, []);

  return (
    <main className="min-h-screen flex items-center justify-center bg-white text-black px-7">
      <div className="max-w-md text-center">
        <svg
          viewBox="0 0 80 80"
          width="64"
          height="64"
          className="mx-auto mb-8"
          aria-hidden="true"
        >
          <circle
            cx="40"
            cy="40"
            r="30"
            fill="none"
            stroke="currentColor"
            strokeWidth="0.75"
          />
          <circle cx="40" cy="40" r="3" fill="#DC2626" />
        </svg>

        <h1 className="text-[32px] font-medium tracking-[-0.02em] mb-4">
          Your download is starting.
        </h1>
        <p className="text-[15px] text-neutral-500 leading-relaxed mb-10">
          If nothing happens,{" "}
          <a
            href={DMG_PATH}
            download="PersonalModelCompany.dmg"
            className="underline text-black"
          >
            click here to download
          </a>
          .
        </p>

        <div className="text-left max-w-sm mx-auto bg-neutral-50 rounded-lg p-5 text-[13px] text-neutral-600 leading-relaxed">
          <p className="font-medium text-neutral-900 mb-2">After downloading</p>
          <p>
            Open the .dmg, drag <span className="text-neutral-900">Personal Model Company</span>{" "}
            to Applications, then launch it from there.
          </p>
          <p className="mt-3 text-neutral-500">
            macOS may warn that the app is unsigned — this is a local dev build.
            Right-click the app → Open → Open to bypass.
          </p>
        </div>

        <div className="mt-10">
          <Link
            href="/"
            className="text-[13px] text-neutral-500 hover:text-black transition-colors"
          >
            ← Back to home
          </Link>
        </div>
      </div>
    </main>
  );
}
