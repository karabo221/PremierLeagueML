import type { Metadata } from "next";
import Link from "next/link";

import { DECOMPOSITION, LADDER, META } from "@/lib/frozen.generated";
import { fixed, pct } from "@/lib/format";

export const metadata: Metadata = {
  title: "How it works",
  description:
    "How a computer that has never watched a football match ends up pricing " +
    "one — the whole project in plain English, with no equations: the question, " +
    "the rules written down first, six attempts, and what beat us.",
};

const blindGuess = LADDER.find((r) => r.kind === "baseline")!;
const frozen = LADDER.find((r) => r.kind === "frozen")!;
const market = LADDER.find((r) => r.kind === "market")!;

/** Of the distance from a blind guess to the market, how much we closed. */
const roadCovered =
  (blindGuess.logLoss! - frozen.logLoss!) / (blindGuess.logLoss! - market.logLoss!);

/** The first rung's share of the whole improvement - the page's first surprise. */
const firstStepShare = DECOMPOSITION.parts[0].share;

/** "2026-2027" is how the pin spells it; "2026–27" is how a reader reads it. */
const SEASON = `${META.holdoutSeason.slice(0, 4)}–${META.holdoutSeason.slice(-2)}`;

/**
 * THE PAGE FOR SOMEONE WHO HAS NEVER MET ANY OF THIS.
 *
 * Rules it is written under, because they are easy to lose a paragraph at a
 * time: no equations, no model names, no file names, one idea per paragraph,
 * and every number carries what it means in the same sentence. The figures are
 * still read from the frozen artefacts rather than typed in, so this page
 * cannot drift away from the evidence page - it just says it differently.
 *
 * The diagram is the spine. Step 2 is the one the rest of the site never
 * showed and the one the whole argument rests on.
 */
export default function HowItWorksPage() {
  return (
    <>
      <section className="hero">
        <div className="shell">
          <p className="eyebrow">No maths required &middot; about six minutes</p>
          <h1 className="hero-h display">
            A computer that has never{" "}
            <span className="hero-em">watched a football match</span>
          </h1>
        </div>
      </section>

      <div className="shell stack stack-lg page-body">
        <div className="article">
          <p>
            Every week, before a ball is kicked, a computer programme writes down
            what it thinks will happen in each Premier League match. Not a
            scoreline &mdash; a set of chances.{" "}
            <strong>Arsenal 61%, draw 23%, Everton 16%.</strong> Then the matches
            are played, and we find out.
          </p>
          <p>
            The programme has never watched a football match. It has never heard of
            a manager, an injury or a transfer. All it has is a list of results:
            who played whom, where, and how it finished. The whole project is a
            test of one question &mdash;{" "}
            <strong>how much of football is already written in the results?</strong>
          </p>
          <p className="pull">
            The yardstick is the bookmakers. They are the hardest opponent
            available, because they lose money when they are wrong.
          </p>
        </div>

        <div className="figwrap">
          <figure>
            <svg
              viewBox="0 0 700 520"
              role="img"
              aria-label="The six steps of the project in order: ask a question you can lose, write the rules down before looking at any data, gather 1,900 matches, try six ideas with the hardest test each time, hold the result up against the bookmakers, then seal the model and predict in public before each kickoff."
            >
              <line
                x1="54" y1="46" x2="54" y2="456"
                stroke="currentColor" strokeWidth="2" opacity="0.22"
              />
              <g fontFamily="var(--sans)">
                {/* 1 */}
                <circle cx="54" cy="46" r="17" fill="none" stroke="currentColor" strokeWidth="2" />
                <text x="54" y="51" fontSize="14" fontWeight="700" fill="currentColor" textAnchor="middle">1</text>
                <text x="92" y="42" fontSize="16" fontWeight="700" fill="currentColor">Ask a question you can lose</text>
                <text x="92" y="62" fontSize="13.5" fill="currentColor" opacity="0.72">Can a computer price a football match as well as a bookmaker?</text>
                <text x="92" y="79" fontSize="13.5" fill="currentColor" opacity="0.72">Bookmakers are the yardstick because they are paid to be right.</text>

                {/* 2 - the one the rest of the site never showed */}
                <circle cx="54" cy="128" r="17" fill="none" stroke="#96263b" strokeWidth="2.5" />
                <text x="54" y="133" fontSize="14" fontWeight="700" fill="#96263b" textAnchor="middle">2</text>
                <text x="92" y="124" fontSize="16" fontWeight="700" fill="currentColor">Write the rules down first</text>
                <text x="92" y="144" fontSize="13.5" fill="currentColor" opacity="0.72">Before looking at a single match: what counts as better, which</text>
                <text x="92" y="161" fontSize="13.5" fill="currentColor" opacity="0.72">seasons we test on, and what would make us abandon an idea.</text>

                {/* 3 */}
                <circle cx="54" cy="210" r="17" fill="none" stroke="currentColor" strokeWidth="2" />
                <text x="54" y="215" fontSize="14" fontWeight="700" fill="currentColor" textAnchor="middle">3</text>
                <text x="92" y="206" fontSize="16" fontWeight="700" fill="currentColor">Gather the matches</text>
                <text x="92" y="226" fontSize="13.5" fill="currentColor" opacity="0.72">1,900 Premier League games across five seasons, 2021&ndash;22 to</text>
                <text x="92" y="243" fontSize="13.5" fill="currentColor" opacity="0.72">2025&ndash;26: who played, where, the score, the odds on offer.</text>

                {/* 4 */}
                <circle cx="54" cy="292" r="17" fill="none" stroke="currentColor" strokeWidth="2" />
                <text x="54" y="297" fontSize="14" fontWeight="700" fill="currentColor" textAnchor="middle">4</text>
                <text x="92" y="288" fontSize="16" fontWeight="700" fill="currentColor">Try six ideas, hardest test each time</text>
                <text x="92" y="308" fontSize="13.5" fill="currentColor" opacity="0.72">Each one only ever judged on seasons it had never seen. Two</text>
                <text x="92" y="325" fontSize="13.5" fill="currentColor" opacity="0.72">surprises: the first idea did most of the work, and a single</text>
                <text x="92" y="342" fontSize="13.5" fill="currentColor" opacity="0.72">strength rating beat 139 hand-built statistics.</text>

                {/* 5 */}
                <circle cx="54" cy="374" r="17" fill="none" stroke="currentColor" strokeWidth="2" />
                <text x="54" y="379" fontSize="14" fontWeight="700" fill="currentColor" textAnchor="middle">5</text>
                <text x="92" y="370" fontSize="16" fontWeight="700" fill="currentColor">Hold it up against the bookmakers</text>
                <text x="92" y="390" fontSize="13.5" fill="currentColor" opacity="0.72">They win, by a small but real margin. We tried four explanations</text>
                <text x="92" y="407" fontSize="13.5" fill="currentColor" opacity="0.72">for why &mdash; all four failed, and we published all four.</text>

                {/* 6 */}
                <circle cx="54" cy="456" r="17" fill="#96263b" stroke="#96263b" strokeWidth="2" />
                <text x="54" y="461" fontSize="14" fontWeight="700" fill="#f5e7e3" textAnchor="middle">6</text>
                <text x="92" y="452" fontSize="16" fontWeight="700" fill="currentColor">Seal it, then predict in public</text>
                <text x="92" y="472" fontSize="13.5" fill="currentColor" opacity="0.72">The model is frozen &mdash; no more tinkering. Every week it writes</text>
                <text x="92" y="489" fontSize="13.5" fill="currentColor" opacity="0.72">its predictions down before kickoff, where the date can&apos;t be faked.</text>
              </g>
            </svg>
            <figcaption>
              <strong>Step 2 is the one that matters.</strong> If you write the rules
              after seeing the data, you can always find a story that fits &mdash;
              which is why almost every impressive prediction result on the internet
              is worth nothing. Ours were written down, in full, before any model
              existed, and they have not moved since.
            </figcaption>
          </figure>
        </div>

        <div className="article">
          <h2 className="article-h">The first surprise: almost all of it is one step</h2>
          <p>
            We tried six increasingly elaborate ideas. The first was almost
            embarrassingly simple: work out how well each team has done{" "}
            <em>this season</em>, and lean on that. It turned out to be worth more
            than everything that came after it combined &mdash; about{" "}
            <strong>{pct(firstStepShare, 0)} of every improvement we ever made</strong>.
          </p>
          <p>
            After that, things got strange. We built 139 detailed statistics from
            the previous season &mdash; shots, possession, the lot &mdash; and they
            were worth almost nothing. Meanwhile a single number per team, worked
            out the way chess players are ranked, quietly beat all 139 of them.{" "}
            <strong>More detail did not mean more knowledge.</strong>
          </p>

          <h2 className="article-h">The second surprise: we could not close the gap</h2>
          <p>
            The best version we built lands a little short of the bookmakers. In the
            units we score in, we reach {fixed(frozen.logLoss!, 3)} where they reach{" "}
            {fixed(market.logLoss!, 3)} &mdash; lower is better, and the difference
            is small but real, not noise. In plainer terms:{" "}
            <strong>
              we covered about {Math.round(roadCovered * 100)}% of the road from a
              blind guess to a professional bookmaker
            </strong>
            , and then stopped.
          </p>
          <p>
            We spent a long time trying to explain the last stretch and failed four
            different ways. Was it certain kinds of match? No &mdash; we lose evenly
            across all of them. A statistic we had missed? No &mdash; we searched 128
            columns and nothing lined up. Were we being too cautious? No &mdash; we
            are bolder than the bookmakers, just bolder in the wrong direction. Team
            news, then? No &mdash; that evidence came out backwards.
          </p>
          <p>
            All four failures are written up on this site in full, because the
            alternative &mdash; quietly reporting the one test that flattered us
            &mdash; is the most common way results like this turn out to be
            worthless.{" "}
            <Link href="/evidence" className="inline-link">
              The full working is on the evidence page
            </Link>
            .
          </p>

          <h2 className="article-h">What happens now</h2>
          <p>
            The model is <strong>sealed</strong>. Nothing about it can change &mdash;
            not a setting, not a rating, not the date it is allowed to look back to.
            Each week it publishes its chances before kickoff, in a place where the
            date cannot be altered afterwards, and at the end of the {SEASON} season
            we score it once &mdash; whatever it says.
          </p>
          <p>
            That last part is the only reason any of this counts for anything. A
            prediction published after the match is not a prediction, and a model
            that gets adjusted whenever it looks wrong is not being tested. So we
            keep a running tally on the front page for interest, and we are not
            allowed to act on it.
          </p>
        </div>

        <div className="informal">
          <span className="informal-t">One honest caveat</span>
          <p>
            The comparison above is <strong>tilted in our favour</strong>, and it is
            worth saying so. Our figures come from seasons the project could look at
            while it was being built; the bookmakers&apos; figures were never fitted
            to anything. We lose anyway. How much of that tilt there is, is exactly
            what the sealed {SEASON} season is for.
          </p>
        </div>
      </div>
    </>
  );
}
