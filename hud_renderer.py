"""Big Brother HUD overlay renderer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from emotion_mapper import compute_loyalty
from face_tracker import TrackedFace

PALETTES = {
    "green": {
        "primary": (65, 255, 0),  # BGR #00FF41
        "loyalty": (40, 195, 35),  # darker green than primary
        "dim": (40, 140, 0),
        "loyalty_dim": (28, 115, 22),
        "rec": (0, 0, 255),
    },
    "amber": {
        "primary": (0, 176, 255),  # BGR #FFB000
        "loyalty": (0, 145, 210),  # darker gold than primary
        "dim": (0, 100, 160),
        "loyalty_dim": (0, 95, 140),
        "rec": (0, 0, 255),
    },
}

METRIC_ORDER = ("JOY", "HAPPINESS", "FEAR", "FOCUS", "DROWSY", "LOYALTY")
METRIC_LABELS: dict[str, str] = {
    "JOY": "JOY",
    "HAPPINESS": "HAPPINESS",
    "FEAR": "FEAR",
    "FOCUS": "FOCUS",
    "DROWSY": "DROWSY",
}
_LOYALTY_LABEL_LINES = ("LOYALTY", "TO BIG", "BROTHER")
# Global scale factor for the per-subject metrics panel ("hub").
_HUD_SCALE = 1.6
_BAR_W = int(150 * _HUD_SCALE)
_BAR_H = int(16 * _HUD_SCALE)
_ROW_GAP = int(22 * _HUD_SCALE)
_LABEL_COL = int(92 * _HUD_SCALE)
_LABEL_SCALE = 0.42 * _HUD_SCALE
_VALUE_SCALE = 0.44 * _HUD_SCALE
_AGE_LINE_H = int(20 * _HUD_SCALE)
_LOYALTY_LINE_H = int(14 * _HUD_SCALE)
_LOYALTY_BLOCK_H = len(_LOYALTY_LABEL_LINES) * _LOYALTY_LINE_H
_LOYALTY_ROW_H = max(_ROW_GAP, _LOYALTY_BLOCK_H + 4)
_LOYALTY_TOP_GAP = int(12 * _HUD_SCALE)
_PANEL_WIDTH = _LABEL_COL + _BAR_W + int(44 * _HUD_SCALE)
_PANEL_HEIGHT = (
    _AGE_LINE_H
    + (len(METRIC_ORDER) - 1) * _ROW_GAP
    + _LOYALTY_TOP_GAP
    + _LOYALTY_ROW_H
)


@dataclass
class HudRenderer:
    """Draw surveillance HUD on BGR frames."""

    color_mode: str = "green"
    font_path: Path | None = None
    _font_scale: float = 0.45
    _thickness: int = 1
    _freetype = None

    def __post_init__(self) -> None:
        self._load_font()

    def _load_font(self) -> None:
        if self.font_path and self.font_path.exists():
            try:
                self._freetype = cv2.freetype.createFreeType2()
                self._freetype.loadFontData(str(self.font_path), id=0)
                return
            except (cv2.error, AttributeError):
                self._freetype = None

    def set_color_mode(self, mode: str) -> None:
        if mode in PALETTES:
            self.color_mode = mode

    def _palette(self) -> dict:
        return PALETTES.get(self.color_mode, PALETTES["green"])

    def _text(
        self,
        img: np.ndarray,
        text: str,
        pos: tuple[int, int],
        scale: float | None = None,
        color: tuple[int, int, int] | None = None,
        thickness: int = 1,
        bold: bool = False,
    ) -> None:
        pal = self._palette()
        c = color if color is not None else pal["primary"]
        sc = scale if scale is not None else self._font_scale
        x, y = pos
        stroke = max(thickness, 2) if bold else thickness
        if self._freetype is not None:
            height = int(18 * sc / 0.45)
            self._freetype.putText(
                img,
                text,
                (x, y),
                fontHeight=height,
                color=c,
                thickness=stroke,
                line_type=cv2.LINE_AA,
                bottomLeftOrigin=False,
            )
            if bold:
                self._freetype.putText(
                    img,
                    text,
                    (x + 1, y),
                    fontHeight=height,
                    color=c,
                    thickness=stroke,
                    line_type=cv2.LINE_AA,
                    bottomLeftOrigin=False,
                )
        else:
            cv2.putText(
                img,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                sc,
                c,
                stroke,
                cv2.LINE_AA,
            )

    def draw_brackets(
        self,
        img: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        arm: int = 18,
    ) -> None:
        pal = self._palette()
        c = pal["primary"]
        t = 2
        # Top-left
        cv2.line(img, (x, y), (x + arm, y), c, t)
        cv2.line(img, (x, y), (x, y + arm), c, t)
        # Top-right
        cv2.line(img, (x + w, y), (x + w - arm, y), c, t)
        cv2.line(img, (x + w, y), (x + w, y + arm), c, t)
        # Bottom-left
        cv2.line(img, (x, y + h), (x + arm, y + h), c, t)
        cv2.line(img, (x, y + h), (x, y + h - arm), c, t)
        # Bottom-right
        cv2.line(img, (x + w, y + h), (x + w - arm, y + h), c, t)
        cv2.line(img, (x + w, y + h), (x + w, y + h - arm), c, t)

    def draw_crosshair(
        self, img: np.ndarray, cx: int, cy: int, size: int = 12
    ) -> None:
        pal = self._palette()
        c = pal["primary"]
        cv2.line(img, (cx - size, cy), (cx + size, cy), c, 1)
        cv2.line(img, (cx, cy - size), (cx, cy + size), c, 1)
        cv2.circle(img, (cx, cy), 3, c, 1)

    def _draw_gauge_bar(
        self,
        img: np.ndarray,
        bx: int,
        by: int,
        val: float,
        bar_w: int,
        bar_h: int,
        bar_color: tuple[int, int, int] | None = None,
        frame_color: tuple[int, int, int] | None = None,
        text_color: tuple[int, int, int] | None = None,
        value_bold: bool = False,
    ) -> None:
        pal = self._palette()
        primary = bar_color if bar_color is not None else pal["primary"]
        dim = frame_color if frame_color is not None else pal["dim"]
        cv2.rectangle(img, (bx, by), (bx + bar_w, by + bar_h), dim, 1)
        fill = int(bar_w * val / 100.0)
        if fill > 0:
            cv2.rectangle(
                img,
                (bx + 1, by + 1),
                (bx + fill, by + bar_h - 1),
                primary,
                -1,
            )
        self._text(
            img,
            f"{int(val):03d}",
            (bx + bar_w + 6, by + bar_h - 1),
            scale=_VALUE_SCALE,
            color=text_color,
            bold=value_bold,
        )

    def _draw_label_lines(
        self,
        img: np.ndarray,
        x: int,
        y: int,
        lines: tuple[str, ...],
        line_h: int,
        color: tuple[int, int, int] | None = None,
        bold: bool = False,
    ) -> int:
        """Draw stacked label lines; return total block height."""
        for i, line in enumerate(lines):
            self._text(
                img,
                line,
                (x, y + (i + 1) * line_h - 3),
                scale=_LABEL_SCALE,
                color=color,
                bold=bold,
            )
        return len(lines) * line_h

    def draw_gauge_bars(
        self,
        img: np.ndarray,
        x: int,
        y: int,
        metrics: dict[str, float],
        bar_w: int = _BAR_W,
        bar_h: int = _BAR_H,
        gap: int = _ROW_GAP,
    ) -> None:
        cursor_y = y
        for key in METRIC_ORDER:
            val = metrics.get(key, 0.0)
            bx = x + _LABEL_COL

            if key == "LOYALTY":
                pal = self._palette()
                loyalty_c = pal["loyalty"]
                loyalty_dim = pal["loyalty_dim"]
                cursor_y += _LOYALTY_TOP_GAP
                self._draw_label_lines(
                    img,
                    x,
                    cursor_y,
                    _LOYALTY_LABEL_LINES,
                    _LOYALTY_LINE_H,
                    color=loyalty_c,
                )
                bar_y = cursor_y + (_LOYALTY_BLOCK_H - bar_h) // 2
                self._draw_gauge_bar(
                    img,
                    bx,
                    bar_y,
                    val,
                    bar_w,
                    bar_h,
                    bar_color=loyalty_c,
                    frame_color=loyalty_dim,
                    text_color=loyalty_c,
                )
                cursor_y += _LOYALTY_ROW_H
                continue

            label = METRIC_LABELS.get(key, key)
            self._text(
                img,
                label,
                (x, cursor_y + bar_h - 1),
                scale=_LABEL_SCALE,
            )
            self._draw_gauge_bar(img, bx, cursor_y, val, bar_w, bar_h)
            cursor_y += gap

    def draw_age_line(
        self,
        img: np.ndarray,
        x: int,
        y: int,
        track: TrackedFace,
    ) -> int:
        """Draw AGE header; return y offset for gauge bars below."""
        if track.species == "cat":
            age_text = "AGE: --"
        else:
            age_text = f"AGE: {int(track.age):03d}"
        self._text(img, age_text, (x, y + _AGE_LINE_H - 4), scale=_LABEL_SCALE)
        return y + _AGE_LINE_H

    def _panel_position(
        self,
        img: np.ndarray,
        bx: int,
        by: int,
        bw: int,
        bh: int,
        track_index: int,
        prefer_left: bool = False,
    ) -> tuple[int, int]:
        """Place gauge panel beside the subject bbox, not at a fixed screen column."""
        gh, gw = img.shape[:2]
        margin = 10
        right_x = bx + bw + margin
        left_x = bx - _PANEL_WIDTH - margin

        if prefer_left and left_x >= 0:
            panel_x = left_x
        elif right_x + _PANEL_WIDTH <= gw:
            panel_x = right_x
        elif left_x >= 0:
            panel_x = left_x
        else:
            panel_x = max(0, min(right_x, gw - _PANEL_WIDTH))

        panel_y = by + track_index * 12
        max_y = gh - _PANEL_HEIGHT - margin
        panel_y = max(margin, min(panel_y, max_y))
        return panel_x, panel_y

    def draw_face_hud(
        self,
        img: np.ndarray,
        track: TrackedFace,
        track_index: int = 0,
    ) -> None:
        x, y, w, h = track.bbox
        cx, cy = track.center
        pad = 12
        bx = max(0, x - pad)
        by = max(0, y - pad)
        bw = w + pad * 2
        bh = h + pad * 2

        self.draw_brackets(img, bx, by, bw, bh)
        self.draw_crosshair(img, cx, cy)

        # Rangefinder ticks
        pal = self._palette()
        cv2.line(img, (bx, cy), (bx + 20, cy), pal["dim"], 1)
        cv2.line(img, (bx + bw - 20, cy), (bx + bw, cy), pal["dim"], 1)

        # Subject label above box
        self._text(img, track.label(), (bx, max(18, by - 8)), scale=0.42)

        panel_x, panel_y = self._panel_position(
            img,
            bx,
            by,
            bw,
            bh,
            track_index,
            prefer_left=track.species == "cat",
        )
        metrics = track.metrics.as_dict()
        metrics["LOYALTY"] = compute_loyalty(
            track.metrics, species=track.species
        )
        bars_y = self.draw_age_line(img, panel_x, panel_y, track)
        self.draw_gauge_bars(img, panel_x, bars_y, metrics)

    def draw_global_hud(self, img: np.ndarray, target_count: int) -> None:
        pal = self._palette()
        h, w = img.shape[:2]
        now = datetime.now().strftime("%H:%M:%S")
        s = _HUD_SCALE

        # REC indicator
        cv2.circle(img, (int(18 * s), int(22 * s)), int(5 * s), pal["rec"], -1)
        self._text(
            img,
            f"REC  {now}",
            (int(32 * s), int(28 * s)),
            scale=0.5 * s,
        )

        self._text(
            img,
            "SYS:BIG_BROTHER_VISION",
            (int(12 * s), h - int(36 * s)),
            scale=0.42 * s,
        )
        targets_text = f"TARGETS: {target_count}"
        self._text(
            img,
            targets_text,
            (w - int(160 * s), int(28 * s)),
            scale=0.45 * s,
        )

        # Corner frame accents
        c = pal["dim"]
        arm = int(40 * s)
        line_t = max(1, int(round(s)))
        cv2.line(img, (0, 0), (arm, 0), c, line_t)
        cv2.line(img, (0, 0), (0, arm), c, line_t)
        cv2.line(img, (w - 1, 0), (w - arm, 0), c, line_t)
        cv2.line(img, (w - 1, 0), (w - 1, arm), c, line_t)

    def render(
        self,
        frame: np.ndarray,
        tracks: list[TrackedFace],
    ) -> np.ndarray:
        """Draw full HUD on a copy of the frame."""
        out = frame.copy()
        for index, track in enumerate(tracks):
            self.draw_face_hud(out, track, track_index=index)
        self.draw_global_hud(out, len(tracks))
        return out
