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
    images: [
      {
        url: "/icon-1024.png",
        width: 1024,
        height: 1024,
        alt: "The Personal Model Company",
      },
    ],
  },
  twitter: {
    card: "summary",
    title: "The Personal Model Company",
    description:
      "Train an AI model on your own writing. You own it. Host it. Take it anywhere.",
    images: ["/icon-1024.png"],
  },
  icons: {
    icon: [
      { url: "/icon.png", sizes: "256x256", type: "image/png" },
      { url: "/icon.svg", type: "image/svg+xml" },
    ],
    apple: "/icon.png",
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
