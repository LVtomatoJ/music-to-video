"""One theme is one product skin: projectile, hit feel, objects, effects, palette.

The ball's path is still planned so it touches a surface on the beat. Feel
parameters only drive the deformation spring, the trail, and the impact hold.
"""

from __future__ import annotations

from dataclasses import dataclass

from music2video.analysis import Analysis


@dataclass(frozen=True)
class Feel:
    mass: float
    squash: float
    stretch: float
    stiffness: float
    damping: float
    pause: float
    shake: float
    tumble: float
    trail: str
    trail_length: int


@dataclass(frozen=True)
class Theme:
    name: str
    label: str
    objects: tuple[str, ...]
    effects: tuple[str, ...]
    projectile: str
    feel: Feel
    hue: float
    hue_step: float
    object_sat: float
    object_light: float
    bg_light: float
    bg_sat: float
    background: str
    intensity: float
    flash: float
    show_spectrum: bool
    # Same-family shapes. A neon video never grows pegs; a pastel video never grows cracks.
    variants: tuple[tuple[str, tuple[str, ...]], ...] = ()
    flourish: str = ""


@dataclass(frozen=True)
class Palette:
    """Screen colors for one skin. Values are linear 0–1 RGB, chosen as a set."""

    bg_top: tuple[float, float, float]
    bg_bottom: tuple[float, float, float]
    accent: tuple[float, float, float]
    accent2: tuple[float, float, float]
    ink: tuple[float, float, float]
    rim: tuple[float, float, float]
    shadow: tuple[float, float, float]
    vignette: float


PALETTES: dict[str, Palette] = {
    "neon": Palette(
        (0.043, 0.067, 0.133),
        (0.012, 0.016, 0.043),
        (0.361, 0.902, 0.886),
        (0.965, 0.286, 0.541),
        (0.055, 0.094, 0.149),
        (0.780, 1.000, 0.965),
        (0.000, 0.000, 0.000),
        0.46,
    ),
    "arcade": Palette(
        (0.110, 0.051, 0.133),
        (0.035, 0.012, 0.055),
        (1.000, 0.784, 0.341),
        (1.000, 0.302, 0.545),
        (0.220, 0.118, 0.275),
        (1.000, 0.910, 0.655),
        (0.000, 0.000, 0.000),
        0.42,
    ),
    "piano": Palette(
        (0.125, 0.098, 0.078),
        (0.047, 0.035, 0.027),
        (0.949, 0.902, 0.816),
        (0.776, 0.631, 0.357),
        (0.965, 0.941, 0.902),
        (1.000, 0.996, 0.980),
        (0.040, 0.024, 0.016),
        0.38,
    ),
    "shatter": Palette(
        (0.110, 0.086, 0.078),
        (0.035, 0.027, 0.024),
        (0.949, 0.910, 0.867),
        (0.886, 0.345, 0.259),
        (0.890, 0.855, 0.816),
        (1.000, 0.980, 0.960),
        (0.000, 0.000, 0.000),
        0.50,
    ),
    "pastel": Palette(
        (0.965, 0.937, 0.910),
        (0.855, 0.765, 0.835),
        (0.761, 0.337, 0.478),
        (0.365, 0.612, 0.569),
        (0.945, 0.835, 0.870),
        (1.000, 0.980, 0.973),
        (0.620, 0.310, 0.420),
        0.22,
    ),
    # Forms (--scene city / piano / stars) use their own night pictures.
    "city": Palette(
        (0.016, 0.027, 0.071),
        (0.090, 0.063, 0.102),
        (1.000, 0.780, 0.420),
        (0.490, 0.855, 0.965),
        (0.075, 0.090, 0.125),
        (0.820, 0.880, 0.980),
        (0.000, 0.000, 0.020),
        0.36,
    ),
    "waterfall": Palette(
        (0.020, 0.024, 0.043),
        (0.008, 0.010, 0.020),
        (0.945, 0.955, 0.980),
        (0.478, 0.780, 0.980),
        (0.110, 0.125, 0.165),
        (1.000, 1.000, 1.000),
        (0.000, 0.000, 0.000),
        0.48,
    ),
    "rain": Palette(
        (0.030, 0.060, 0.100),
        (0.006, 0.022, 0.036),
        (0.620, 0.780, 0.900),
        (0.420, 0.700, 0.780),
        (0.030, 0.060, 0.090),
        (0.900, 0.960, 1.000),
        (0.000, 0.000, 0.010),
        0.38,
    ),
    "stars": Palette(
        (0.05, 0.055, 0.14),
        (0.08, 0.05, 0.14),
        (0.855, 0.910, 1.000),
        (0.965, 0.820, 0.620),
        (0.040, 0.050, 0.090),
        (1.000, 0.980, 0.940),
        (0.000, 0.000, 0.000),
        0.30,
    ),
}


def palette_of(theme: Theme | None) -> Palette:
    if theme is None:
        return PALETTES["neon"]
    return PALETTES.get(theme.name, PALETTES["neon"])


THEMES: dict[str, Theme] = {}


def register_theme(theme: Theme) -> Theme:
    THEMES[theme.name] = theme
    return theme


def pick_theme(analysis: Analysis) -> str:
    """Choose one skin from the song's overall energy. Never mixes skins."""
    if not analysis.sections:
        return "neon"
    energy = sum(section.energy for section in analysis.sections) / len(analysis.sections)
    strong_ratio = len(analysis.strong_beats) / max(1, len(analysis.beats))
    if energy < 0.27:
        return "pastel"
    if strong_ratio < 0.28 and energy < 0.34:
        return "piano"
    if strong_ratio >= 0.48 or energy >= 0.40:
        return "shatter"
    if energy >= 0.32:
        return "arcade"
    return "neon"


register_theme(
    Theme(
        name="neon",
        label="霓虹",
        objects=("wall",),
        effects=("shockwave",),
        projectile="arrow",
        feel=Feel(
            mass=0.32,
            squash=0.22,
            stretch=0.12,
            stiffness=150.0,
            damping=24.0,
            pause=0.0,
            shake=0.0,
            tumble=0.0,
            trail="streak",
            trail_length=12,
        ),
        hue=168.0,
        hue_step=0.0,
        object_sat=0.88,
        object_light=0.52,
        bg_light=0.035,
        bg_sat=0.72,
        background="neon",
        intensity=0.8,
        flash=0.1,
        show_spectrum=True,
        variants=(("wall", ("short", "long", "corner", "double")),),
        flourish="ring",
    )
)

register_theme(
    Theme(
        name="arcade",
        label="街机",
        objects=("peg", "pad", "spinner"),
        effects=("sparks",),
        projectile="pinball",
        feel=Feel(
            mass=0.48,
            squash=0.9,
            stretch=0.22,
            stiffness=130.0,
            damping=26.0,
            pause=0.02,
            shake=0.0,
            tumble=0.0,
            trail="afterimage",
            trail_length=5,
        ),
        hue=48.0,
        hue_step=108.0,
        object_sat=0.9,
        object_light=0.56,
        bg_light=0.07,
        bg_sat=0.62,
        background="arcade",
        intensity=0.95,
        flash=0.22,
        show_spectrum=True,
        variants=(
            ("peg", ("single", "pair")),
            ("pad", ("short", "long")),
            ("spinner", ("short", "long")),
        ),
        flourish="burst",
    )
)

register_theme(
    Theme(
        name="piano",
        label="琴键",
        objects=("piano", "platform"),
        effects=("glow",),
        projectile="note",
        feel=Feel(
            mass=0.85,
            squash=0.42,
            stretch=0.95,
            stiffness=34.0,
            damping=3.0,
            pause=0.04,
            shake=0.0,
            tumble=0.0,
            trail="ribbon",
            trail_length=18,
        ),
        hue=32.0,
        hue_step=16.0,
        object_sat=0.42,
        object_light=0.58,
        bg_light=0.06,
        bg_sat=0.4,
        background="piano",
        intensity=0.55,
        flash=0.06,
        show_spectrum=False,
        variants=(
            ("piano", ("single", "chord", "pattern")),
            ("platform", ("short", "long")),
        ),
        flourish="arpeggio",
    )
)

register_theme(
    Theme(
        name="shatter",
        label="冲击",
        objects=("breakable", "wall"),
        effects=("shatter", "shake"),
        projectile="cube",
        feel=Feel(
            mass=2.1,
            squash=1.05,
            stretch=0.12,
            stiffness=16.0,
            damping=14.0,
            pause=0.14,
            shake=16.0,
            tumble=11.0,
            trail="none",
            trail_length=0,
        ),
        hue=8.0,
        hue_step=20.0,
        object_sat=0.08,
        object_light=0.84,
        bg_light=0.02,
        bg_sat=0.15,
        background="shatter",
        intensity=1.0,
        flash=0.34,
        show_spectrum=False,
        variants=(
            ("breakable", ("crack", "block", "stack")),
            ("wall", ("short", "long")),
        ),
        flourish="break",
    )
)

register_theme(
    Theme(
        name="pastel",
        label="柔彩",
        objects=("wall", "platform"),
        effects=("glow",),
        projectile="bubble",
        feel=Feel(
            mass=1.35,
            squash=0.32,
            stretch=0.28,
            stiffness=11.0,
            damping=4.8,
            pause=0.06,
            shake=0.0,
            tumble=0.0,
            trail="bubbles",
            trail_length=14,
        ),
        hue=210.0,
        hue_step=28.0,
        object_sat=0.28,
        object_light=0.72,
        bg_light=0.16,
        bg_sat=0.22,
        background="pastel",
        intensity=0.4,
        flash=0.05,
        show_spectrum=False,
        variants=(
            ("platform", ("short", "long", "arch")),
            ("wall", ("short", "long")),
        ),
        flourish="split",
    )
)
