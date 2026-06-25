// src/app/layout.tsx
import type { Metadata } from "next";
import { Archivo, DM_Sans, JetBrains_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const display = Archivo({
  subsets: ["latin"],
  weight: ["600", "700", "800", "900"],
  variable: "--font-display",
});
const body = DM_Sans({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-body",
});
const mono = JetBrains_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "700"],
  variable: "--font-mono",
});

export const metadata: Metadata = {
  title: "Churn War Room",
  description: "Multi-agent churn intelligence — XGBoost · RAG · MCP",
};

const NAV = [
  { href: "/", label: "Dashboard" },
  { href: "/logs", label: "Retention Log" },
  { href: "/metrics", label: "Metrics" },
];

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={`${display.variable} ${body.variable} ${mono.variable}`}>
        <header
          className="sticky top-0 z-50 flex items-center justify-between px-6 h-14 border-b backdrop-blur"
          style={{
            borderColor: "var(--border)",
            background: "rgba(10,14,20,0.8)",
          }}
        >
          <Link href="/" className="flex items-center gap-2.5 no-underline">
            <span
              className="inline-block w-2.5 h-2.5 rounded-full pulse"
              style={{ background: "var(--risk)", boxShadow: "var(--glow-risk)" }}
            />
            <span
              className="font-display font-extrabold tracking-tight text-[15px]"
              style={{ color: "var(--text)" }}
            >
              CHURN WAR ROOM
            </span>
          </Link>

          <nav className="flex items-center gap-1">
            {NAV.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className="px-3 py-1.5 rounded-md text-[13px] font-medium no-underline transition-colors"
                style={{ color: "var(--text-dim)" }}
              >
                {item.label}
              </Link>
            ))}
          </nav>
        </header>

        <main className="max-w-[1280px] mx-auto px-6 py-8">{children}</main>
      </body>
    </html>
  );
}