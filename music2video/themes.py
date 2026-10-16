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
