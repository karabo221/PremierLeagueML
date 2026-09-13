import type { Metadata } from "next";
import { Archivo, IBM_Plex_Mono, Newsreader } from "next/font/google";
import Link from "next/link";

import { META } from "@/lib/frozen.generated";
import "./globals.css";
import "./parts.css";

/**
 * Three faces, three jobs. Archivo sets every heading, name and figure -
 * anything scanned. Newsreader sets running text, because the site now has a
 * page somebody is meant to READ rather than consult. The mono is for hashes,
 * dates and the few places where digits have to line up in a column.
 */
const archivo = Archivo({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700", "800"],
  variable: "--font-archivo",
  display: "swap",
});

const newsreader = Newsreader({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  style: ["normal", "italic"],
  variable: "--font-newsreader",
  display: "swap",
  // Next has no built-in metrics for Newsreader, so its automatic fallback
  // adjustment fails and prints as an error during the build. Declaring the
  // fallback stack by hand and turning the adjustment off is the documented
  // way out: the face still loads, and the build log stops carrying a red
  // line that is not a problem.
  adjustFontFallback: false,
  fallback: ["Georgia", "Times New Roman", "serif"],
});

const mono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: {
    default: "Premier League predictions, written before kickoff",
    template: "%s · PremierLeagueML",
  },
  description:
    "A sealed model's pre-kickoff chances for every 2026-27 Premier League " +
    "match, written down where the timestamp cannot be moved afterwards — " +
    "plus how it was built, in plain English, and the evidence behind it.",
  robots: { index: true, follow: true },
};

const NAV = [
  { href: "/", label: "Fixtures" },
  { href: "/how-it-works", label: "How it works" },
  { href: "/evidence", label: "The evidence" },
  { href: "/log", label: "The record" },
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html
      lang="en-GB"
      className={`${archivo.variable} ${newsreader.variable} ${mono.variable}`}
    >
      <body>
        <header className="site-hd">
          <div className="shell site-hd-in">
            <Link href="/" className="brand">
              Premier League<em>ML</em>
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

        {/*
          The footer used to print three 64-character hashes and restate the
          informal notice a fourth time. Nobody has ever verified a hash by
          reading it off a footer, and a warning given four times reads as
          anxiety rather than care. One line each, and the full hashes live on
          the evidence page where someone checking them is already standing.
        */}
        <footer className="site-foot">
          <div className="shell foot-grid">
            <div className="foot-note">
              <p className="eyebrow">What this site is</p>
              <p>
                A prediction written before each match, and the working behind it.
                The model was sealed on {META.cutoffDate} and has not been touched
                since &mdash; that is the whole point of it.
              </p>
            </div>

            <div className="foot-note">
              <p className="eyebrow">What it is not</p>
              <p>
                Not betting advice, and not a result yet. The season&apos;s answer is
                one figure, scored once after the final fixture against rules fixed
                in advance. Everything running on this site is a tally kept while we
                wait, and <strong>nothing seen in it may change the model</strong>.
              </p>
              <p className="mono">
                Sealed {META.cutoffDate} &middot; {META.cutoffFixture} onward &middot;{" "}
                <Link href="/evidence" className="inline-link">
                  full hashes and sources
                </Link>
              </p>
            </div>
          </div>
        </footer>
      </body>
    </html>
  );
}
