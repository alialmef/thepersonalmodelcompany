import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "The Personal Model Company",
  description:
    "Train an AI model on your own writing. You own it. Host it. Take it anywhere.",
  metadataBase: new URL(
    process.env.NEXT_PUBLIC_SITE_URL ?? "https://thepersonalmodelcompany.com",
  ),
  openGraph: {
    title: "The Personal Model Company",
    description:
      "Train an AI model on your own writing. You own it. Host it. Take it anywhere.",
    type: "website",
    siteName: "The Personal Model Company",
  },
  twitter: {
    card: "summary_large_image",
    title: "The Personal Model Company",
    description:
      "Train an AI model on your own writing. You own it. Host it. Take it anywhere.",
  },
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#ffffff" },
    { media: "(prefers-color-scheme: dark)", color: "#000000" },
  ],
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
