"""Seasonal color palette definitions.

The classic 4-season model decomposes personal coloring along three
axes and assigns a season to each corner:

              warm            cool
  light     Spring          Summer
  deep      Autumn          Winter

  Spring = warm + light + clear (high chroma)
  Summer = cool + light + soft  (low chroma)
  Autumn = warm + deep  + soft  (low chroma)
  Winter = cool + deep  + clear (high chroma)

This is the standard simplification used by most personal-color
consultants before subdividing into 12 or 16 tones. Swatches below are a
small curated recommendation set per season, not derived from the model.
"""

from __future__ import annotations

from dataclasses import dataclass

SEASONS = ("Spring", "Summer", "Autumn", "Winter")


@dataclass(frozen=True)
class SeasonProfile:
    name: str
    tagline: str
    swatches: tuple[str, ...]


PALETTES: dict[str, SeasonProfile] = {
    "Spring": SeasonProfile(
        name="Spring",
        tagline="Warm undertone, light value, clear/high-chroma coloring",
        swatches=("#FFCA3A", "#FF7A5C", "#8AC926", "#FFB4A2", "#52B788", "#FFD670"),
    ),
    "Summer": SeasonProfile(
        name="Summer",
        tagline="Cool undertone, light value, soft/muted coloring",
        swatches=("#A7C6DA", "#C6A4C5", "#9BB0C1", "#D8B4D8", "#7FA8A3", "#B8C9D9"),
    ),
    "Autumn": SeasonProfile(
        name="Autumn",
        tagline="Warm undertone, deep value, soft/muted coloring",
        swatches=("#A9642A", "#7A5230", "#B08B2E", "#7C3626", "#6B7A3A", "#C97B3D"),
    ),
    "Winter": SeasonProfile(
        name="Winter",
        tagline="Cool undertone, deep value, clear/high-chroma coloring",
        swatches=("#1D3557", "#7B2D8B", "#0B132B", "#E63946", "#003049", "#3A0CA3"),
    ),
}


def get_palette(season: str) -> SeasonProfile:
    """Look up a season's profile; raises ValueError (not KeyError) for an unknown name."""
    try:
        return PALETTES[season]
    except KeyError as exc:
        raise ValueError(f"Unknown season: {season}") from exc
