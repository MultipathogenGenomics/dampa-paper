"""Shared program colors sampled from Figure 4 panels B and C."""

from __future__ import annotations

import matplotlib as mpl
import numpy as np


PROGRAM_COLORS = {
    "dampa": "#2278B5",
    "probetools": "#F57F20",
    "catch": "#2FA148",
    "catch_large": "#C03D3E",
}

FALLBACK_COLORS = (
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#BCBD22",
    "#17BECF",
)


def program_color(label: str, fallback_index: int = 0) -> str:
    """Return the manuscript color for a program label."""
    normalized = label.lower().replace("-", "_").replace(" ", "_")
    if "dampa" in normalized:
        return PROGRAM_COLORS["dampa"]
    if "probetools" in normalized:
        return PROGRAM_COLORS["probetools"]
    if "catch" in normalized and "large" in normalized:
        return PROGRAM_COLORS["catch_large"]
    if "catch" in normalized:
        return PROGRAM_COLORS["catch"]
    return FALLBACK_COLORS[fallback_index % len(FALLBACK_COLORS)]


def distinct_program_colors(labels: list[str]) -> list[str]:
    """Assign a distinct color to every label while preserving program colors.

    The first occurrence of a recognized program keeps its standard manuscript
    color. Additional labels for that program receive unused categorical
    colors instead.
    """
    preferred_colors = [program_color(label, index) for index, label in enumerate(labels)]
    reserved_colors = {mpl.colors.to_hex(color).lower() for color in preferred_colors}
    candidate_colors = list(FALLBACK_COLORS) + [
        mpl.colors.to_hex(color)
        for color in mpl.colormaps["tab20"].colors
    ]
    if len(labels) > len(candidate_colors):
        candidate_colors.extend(
            mpl.colors.to_hex(mpl.colormaps["hsv"](value))
            for value in np.linspace(0, 1, len(labels), endpoint=False)
        )

    colors = []
    used_colors: set[str] = set()
    for preferred_color in preferred_colors:
        normalized = mpl.colors.to_hex(preferred_color).lower()
        if normalized not in used_colors:
            color = preferred_color
        else:
            color = next(
                candidate
                for candidate in candidate_colors
                if candidate.lower() not in used_colors
                and candidate.lower() not in reserved_colors
            )
        colors.append(color)
        used_colors.add(mpl.colors.to_hex(color).lower())
    return colors


def program_palette(labels: list[str]) -> dict[str, str]:
    """Build a label-to-color mapping while retaining colors for unknown labels."""
    return {
        label: program_color(label, fallback_index=index)
        for index, label in enumerate(labels)
    }
