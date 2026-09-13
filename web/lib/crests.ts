/**
 * Team name -> crest file, plus the monogram fallback.
 *
 * The crests are the 16 that exist in the quiz-shorts library. FOUR OF THE
 * TWENTY 2026-27 SIDES ARE NOT IN IT - Nottingham Forest, Hull City, Ipswich
 * Town and Coventry City - so those render as monograms. A missing crest has to
 * look deliberate rather than broken, which is why the fallback is a designed
 * element and not a 404'd <img>.
 *
 * Keys are spelled exactly as the live log spells them, because the log's
 * spelling is what the pin declares (P7.2) and a lookup miss here would be
 * silent. V6b in this project's record is a misspelled side that scored as
 * exactly average with no exception raised; hasCrest() is checked, not assumed.
 */

const CREST_FILES: Record<string, string> = {
  "Arsenal": "arsenal",
  "Aston Villa": "aston-villa",
  "Bournemouth": "bournemouth",
  "Brentford": "brentford",
  "Brighton": "brighton",
  "Chelsea": "chelsea",
  "Crystal Palace": "crystal-palace",
  "Everton": "everton",
  "Fulham": "fulham",
  "Leeds United": "leeds-united",
  "Liverpool": "liverpool",
  "Manchester City": "manchester-city",
  "Manchester Utd": "manchester-utd",
  "Newcastle": "newcastle",
  "Sunderland": "sunderland",
  "Tottenham": "tottenham",
};

/** The four with no crest in the library, named so the gap is inspectable. */
export const MISSING_CRESTS = [
  "Nottingham",
  "Hull",
  "Ipswich Town",
  "Coventry",
] as const;

export function crestUrl(team: string): string | null {
  const file = CREST_FILES[team];
  return file ? `/crests/${file}.png` : null;
}

export function hasCrest(team: string): boolean {
  return team in CREST_FILES;
}

/** Up to three letters, one per word. "Manchester Utd" -> "MU". */
export function monogram(team: string): string {
  return team
    .split(/[\s-]+/)
    .map((word) => word[0] ?? "")
    .join("")
    .slice(0, 3)
    .toUpperCase();
}

/**
 * Short display names. The log's spellings are canonical for data but some are
 * awkward on a card ("Nottingham" for Forest, "Manchester Utd"), so the UI gets
 * its own labels and the data keeps its own.
 */
const DISPLAY: Record<string, string> = {
  "Nottingham": "Nott'm Forest",
  "Manchester Utd": "Man Utd",
  "Manchester City": "Man City",
  "Brighton": "Brighton",
  "Hull": "Hull City",
  "Coventry": "Coventry City",
  "Bournemouth": "Bournemouth",
};

export function displayName(team: string): string {
  return DISPLAY[team] ?? team;
}
