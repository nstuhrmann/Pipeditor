"""
Every colour and font the editor uses, in one place.

Change a value here and it takes effect everywhere — nothing else in the
codebase should contain a literal colour or font family.

Fonts
-----
FONT_FAMILIES is a preference list, tried in order. Qt silently
substitutes a default when a family is missing, so a build on a machine
without the intended font looks subtly different with no error at all —
resolve_font() checks explicitly instead and reports what it actually
got, once, at startup.

A licensed font (Helvetica Neue LT among them) has to be either
installed on the target machine or bundled with the application and
registered at runtime:

    QFontDatabase.addApplicationFont("path/to/HelveticaNeueLTStd-Roman.otf")

Bundling redistributes the font file, so check that your licence allows
it before shipping the .otf inside a Nuitka build. If it doesn't, keep
the family first in the list and let the fallbacks cover machines that
lack it.
"""
from PySide6.QtGui import QColor, QFont, QFontDatabase

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

COLORS = {
    # canvas
    "background":          "#222222",
    # node fills
    "node_body":           "#575756",
    "header_source":       "#13a256",
    "header_sink":         "#ef7d00",
    "header_metric":       "#2b82bd",
    "header_step":         "#888888",
    # node outline
    "node_outline":        "#222222",   # unselected: reads as a gap
    "node_outline_selected": "#ffe48b",
    # ports
    "port_in":             "#85c9f0",
    "port_out":            "#95c994",
    "port_control":        "#afca0b",
    # edges
    "edge":                "#dedfe0",
    "edge_selected":       "#e40521",
    "edge_control":        "#afca0b",
    "edge_temporary":      "#888888",   # while being dragged
    # text on the canvas
    "text_params":         "#dddddd",
    "text_metric_value":   "#ffdd44",
    "text_timing":         "#8a8a8a",
    "text_meta":           "#4dd0e1",   # metadata: forward, with the frame
    "text_control":        "#afca0b",   # control: backward, next frame
    "text_outline":        "#141414",   # halo behind canvas text
    # menus / popups
    "menu_bg":             "#2b2b2b",
    "menu_fg":             "#eeeeee",
    "menu_border":         "#444444",
    "menu_selected_bg":    "#3a6ea5",
    "menu_selected_fg":    "#ffffff",
    "menu_separator":      "#444444",
}


def color(name: str) -> QColor:
    return QColor(COLORS[name])


def readable_on(bg: QColor) -> QColor:
    """Dark or light text, whichever contrasts better with `bg`.

    Used for header labels: white on the lighter headers is only ~2.8:1,
    and hard-coding one or the other breaks the moment a header colour is
    retuned here."""
    def _lin(c):
        c /= 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    lum = (0.2126 * _lin(bg.red()) + 0.7152 * _lin(bg.green())
           + 0.0722 * _lin(bg.blue()))
    on_dark_text = (lum + 0.05) / 0.05      # contrast against near-black
    on_light_text = 1.05 / (lum + 0.05)     # contrast against white
    return (color("background") if on_dark_text > on_light_text
            else QColor("#ffffff"))


def menu_style() -> str:
    return f"""
    QMenu {{
        background-color: {COLORS['menu_bg']};
        color: {COLORS['menu_fg']};
        border: 1px solid {COLORS['menu_border']};
    }}
    QMenu::item {{ padding: 4px 22px 4px 22px; }}
    QMenu::item:selected {{
        background-color: {COLORS['menu_selected_bg']};
        color: {COLORS['menu_selected_fg']};
    }}
    QMenu::separator {{
        height: 1px;
        background: {COLORS['menu_separator']};
        margin: 4px 0;
    }}
    """


# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------

#: Tried in order; the first family actually present is used.
FONT_FAMILIES = [
    "Helvetica Neue LT Std",
    "HelveticaNeueLT Std",
    "Helvetica Neue LT Pro",
    "Helvetica Neue",
    "Helvetica",
    "Arial",            # metric-compatible, present on every Windows box
    "Segoe UI",
]

FONT_SIZES = {
    "node_title":   9,
    "node_params":  7,
    "node_metric":  9,
    "port_label":   7,
    "side_channel": 7,
    "timing":       6,
}

_resolved: str | None = None


def resolve_family() -> str:
    """First available family from FONT_FAMILIES, or Qt's default.

    Resolved once and cached. Qt would silently substitute rather than
    fail, which is exactly the kind of difference that shows up only as
    'the deployed build looks a bit off'."""
    global _resolved
    if _resolved is None:
        available = set(QFontDatabase.families())
        _resolved = next((f for f in FONT_FAMILIES if f in available), "")
    return _resolved


def font(role: str = "node_params", bold: bool = False) -> QFont:
    f = QFont()
    family = resolve_family()
    if family:
        f.setFamily(family)
    f.setPointSize(FONT_SIZES.get(role, 9))
    f.setBold(bold)
    return f


def font_report() -> str:
    """One line describing what was actually resolved — printed at
    startup so a missing font is visible instead of silent."""
    family = resolve_family()
    if not family:
        return (f"font: none of {FONT_FAMILIES[:3]}… available, "
                f"using Qt's default")
    if family != FONT_FAMILIES[0]:
        return (f"font: '{FONT_FAMILIES[0]}' not installed, "
                f"using '{family}'")
    return f"font: '{family}'"
