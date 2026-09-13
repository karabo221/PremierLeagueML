import type { Metadata } from "next";
import { Archivo, JetBrains_Mono } from "next/font/google";
import Link from "next/link";

import { META } from "@/lib/frozen.generated";
import { shortHash } from "@/lib/format";
import "./globals.css";
import "./parts.css";

const archivo = Archivo({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700", "800"],
  variable: "--font-archivo",
  display: "swap",
});

const mono = JetBrains_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "700"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: {
    default: "Premier League pre-kickoff predictions",
    template: "%s · PremierLeagueML",
  },
  description:
    "A frozen Dixon-Coles model's pre-kickoff probabilities for the 2026-27 " +
    "Premier League, written before each matchweek's first fixture and logged " +
    "where the timestamp cannot be backdated. Plus the development evidence: " +
    "0.99036 log loss against a market benchmark of 0.96057 on 1,520 matches.",
  robots: { index: true, follow: true },
};

const NAV = [
  { href: "/", label: "Fixtures" },
  { href: "/evidence", label: "Evidence" },
  { href: "/log", label: "The log" },
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en-GB" className={`${archivo.variable} ${mono.variable}`}>
      <body>
        <header className="site-hd">
          <div className="shell site-hd-in">
            <Link href="/" className="brand">
              <span className="brand-mark" aria-hidden="true" />
              <span className="brand-tx">
                <strong>PremierLeagueML</strong>
                <em>Pre-kickoff prediction log</em>
              </span>
            </Link>

            <nav className="site-nav" aria-label="Sections">
              {NAV.map((item) => (
                <Link key={item.href} href={item.href}>
                  {item.label}
                </Link>
              ))}
            </nav>
          </div>
        </header>

        <main>{children}</main>

        <footer className="site-foot">
          <div className="shell foot-grid">
            <div>
              <p className="eyebrow">The governing documents</p>
              <dl className="foot-dl mono">
                <dt>freeze</dt>
                <dd>{shortHash(META.freezeSha)}</dd>
                <dt>pin</dt>
                <dd>{shortHash(META.pinSha)}</dd>
                <dt>protocol</dt>
                <dd>{shortHash(META.protocolSha)}</dd>
                <dt>cutoff</dt>
                <dd>{META.cutoffDate}</dd>
              </dl>
            </div>

            <div className="foot-note">
              <p className="eyebrow">What this site is not</p>
              <p>
                Every running figure here is <strong>informal</strong>. The official
                2026-27 result is <span className="mono">phase6_score_holdout.py</span>,
                run once after the final fixture against the pinned cutoff. A live-log
                figure is a different instrument with a strictly smaller information
                set &mdash; {META.logRefitsPerSeason} refits a season against the frozen
                model&apos;s {META.frozenRefitsRange} &mdash; and is not comparable to it.
              </p>
              <p>
                Nothing observed in this log may change the model, a feature, a
                hyperparameter, a rating, the cutoff or the de-vig. Any such change marks
                the holdout compromised.
              </p>
            </div>
          </div>
        </footer>
      </body>
    </html>
  );
}
