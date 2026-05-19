"""Fighter-jet style HUD overlay renderer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from face_tracker import TrackedFace

PALETTES = {
    "green": {
        "primary": (65, 255, 0),  # BGR #00FF41
        "dim": (40, 140, 0),
        "rec": (0, 0, 255),
    },
    "amber": {
        "primary": (0, 176, 255),  # BGR #FFB000
        "dim": (0, 100, 160),
        "rec": (0, 0, 255),
    },
}

METRIC_ORDER = ("JOY", "HAPPINESS", "FEAR", "FOCUS", "DROWSY")


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
    ) -> None:
        pal = self._palette()
        c = color if color is not None else pal["primary"]
        sc = scale if scale is not None else self._font_scale
        x, y = pos
        if self._freetype is not None:
            self._freetype.putText(
                img,
                text,
                (x, y),
                fontHeight=int(18 * sc / 0.45),
                color=c,
                thickness=thickness,
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
                thickness,
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

    def draw_gauge_bars(
        self,
        img: np.ndarray,
        x: int,
        y: int,
        metrics: dict[str, float],
        bar_w: int = 90,
        bar_h: int = 8,
        gap: int = 14,
    ) -> None:
        pal = self._palette()
        primary = pal["primary"]
        dim = pal["dim"]
        for i, key in enumerate(METRIC_ORDER):
            val = metrics.get(key, 0.0)
            by = y + i * gap
            self._text(img, key[:4], (x, by + bar_h - 1), scale=0.35)
            bx = x + 42
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
                scale=0.35,
            )

    def draw_face_hud(self, img: np.ndarray, track: TrackedFace) -> None:
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

        # Gauge panel to the right of face
        gh, gw = img.shape[:2]
        panel_x = min(bx + bw + 10, gw - 160)
        panel_y = max(10, by)
        self.draw_gauge_bars(img, panel_x, panel_y, track.metrics.as_dict())

    def draw_global_hud(self, img: np.ndarray, target_count: int) -> None:
        pal = self._palette()
        h, w = img.shape[:2]
        now = datetime.now().strftime("%H:%M:%S")

        # REC indicator
        cv2.circle(img, (18, 22), 5, pal["rec"], -1)
        self._text(img, f"REC  {now}", (32, 28), scale=0.5)

        self._text(img, "SYS:BIG_BROTHER_VISION", (12, h - 36), scale=0.42)
        self._text(
            img,
            f"TARGETS: {target_count}",
            (w - 160, 28),
            scale=0.45,
        )

        # Corner frame accents
        c = pal["dim"]
        arm = 40
        cv2.line(img, (0, 0), (arm, 0), c, 1)
        cv2.line(img, (0, 0), (0, arm), c, 1)
        cv2.line(img, (w - 1, 0), (w - arm, 0), c, 1)
        cv2.line(img, (w - 1, 0), (w - 1, arm), c, 1)

    def render(
        self,
        frame: np.ndarray,
        tracks: list[TrackedFace],
    ) -> np.ndarray:
        """Draw full HUD on a copy of the frame."""
        out = frame.copy()
        for track in tracks:
            self.draw_face_hud(out, track)
        self.draw_global_hud(out, len(tracks))
        return out
