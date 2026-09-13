import { crestUrl, monogram } from "@/lib/crests";

/**
 * A crest, or a monogram where the library has none. Four of the twenty
 * 2026-27 sides have no crest, so the fallback is a designed element rather
 * than a broken image - it has to read as "no crest for this side", not as a
 * loading failure.
 */
export function Crest({ team, size = 28 }: { team: string; size?: number }) {
  const url = crestUrl(team);
  const box = { width: size, height: size };

  if (!url) {
    return (
      <span
        className="crest crest-mono"
        style={{ ...box, fontSize: Math.round(size * 0.36) }}
        title={`${team} — no crest available`}
        aria-hidden="true"
      >
        {monogram(team)}
      </span>
    );
  }

  return (
    // eslint-disable-next-line @next/next/no-img-element -- pre-sized PNGs in
    // public/, so next/image would add a Vercel dependency and no benefit.
    <img
      className="crest"
      src={url}
      alt=""
      width={size}
      height={size}
      style={box}
      loading="lazy"
      decoding="async"
    />
  );
}
