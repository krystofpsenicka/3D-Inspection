"""Phase / PhaseController — step through scene phases inside an Isaac Sim app.

Each phase has its own ``/World/<sanitized_name>`` parent Xform; the controller deletes the
subtree on phase exit (unless ``persistent=True``). Phases advance on N/RIGHT or after
``duration`` seconds.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)


def _sanitize(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not s or not (s[0].isalpha() or s[0] == "_"):
        s = "_" + s
    return s


@dataclass
class Phase:
    """One phase. ``enter`` builds prims under ``parent_path``; default ``exit`` deletes the subtree
    unless ``persistent=True``. ``on_step(stage, parent_path, frame_idx, dt)`` runs every render tick.
    ``duration`` (seconds) auto-advances; ``None`` means manual keypress only."""

    name: str
    enter: Callable[[object, str], object | None]
    exit: Callable[[object, str], None] | None = None
    on_step: Callable[[object, str, int, float], None] | None = None
    duration: float | None = None
    persistent: bool = False
    metadata: dict = field(default_factory=dict)


class PhaseController:
    """Drive a list of :class:`Phase` objects. Keys: N/Right next, P/Left prev, Q/Esc quit."""

    def __init__(
        self,
        ctx,
        phases: list[Phase],
        advance_keys: tuple[str, ...] = ("N", "RIGHT"),
        prev_keys: tuple[str, ...] = ("P", "LEFT"),
        quit_keys: tuple[str, ...] = ("Q", "ESCAPE"),
    ):
        self.ctx = ctx
        self.phases = phases
        self.advance_keys = tuple(k.upper() for k in advance_keys)
        self.prev_keys = tuple(k.upper() for k in prev_keys)
        self.quit_keys = tuple(k.upper() for k in quit_keys)
        self._idx = -1
        self._frame_in_phase = 0
        self._enter_t = 0.0
        self._pending: str | None = None
        self._kb_sub = None
        self._kb_iface = None
        self._app_kb = None
        self._parent_paths: list[str] = []

    def _install_keyboard(self) -> None:
        try:
            import carb.input  # type: ignore
            import omni.appwindow  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on Isaac runtime
            logger.warning("[PhaseController] Keyboard unavailable: %s", exc)
            return

        self._kb_iface = carb.input.acquire_input_interface()
        appwin = omni.appwindow.get_default_app_window()
        self._app_kb = appwin.get_keyboard()
        # Current Isaac uses KEY_PRESS; older builds use KEY_PRESSED.
        KEY_PRESSED = getattr(
            carb.input.KeyboardEventType,
            "KEY_PRESS",
            getattr(carb.input.KeyboardEventType, "KEY_PRESSED", None),
        )

        def on_key(event, *_args):
            if event.type != KEY_PRESSED:
                return True
            name = str(event.input).rsplit(".", 1)[-1].upper()
            if name in self.advance_keys:
                self._pending = "next"
            elif name in self.prev_keys:
                self._pending = "prev"
            elif name in self.quit_keys:
                self._pending = "quit"
            return True

        self._kb_sub = self._kb_iface.subscribe_to_keyboard_events(self._app_kb, on_key)

    def _uninstall_keyboard(self) -> None:
        try:
            if self._kb_sub is not None and self._kb_iface is not None and self._app_kb is not None:
                self._kb_iface.unsubscribe_to_keyboard_events(self._app_kb, self._kb_sub)
        except Exception:
            pass
        self._kb_sub = None
        self._kb_iface = None
        self._app_kb = None

    def _phase_parent(self, phase: Phase) -> str:
        return f"{self.ctx.default_prim_path}/{_sanitize(phase.name)}"

    def _enter_phase(self, idx: int) -> None:
        phase = self.phases[idx]
        parent = self._phase_parent(phase)

        from pxr import UsdGeom

        UsdGeom.Xform.Define(self.ctx.stage, parent)

        logger.info("[PhaseController] [%d/%d] -> %s", idx + 1, len(self.phases), phase.name)
        phase.enter(self.ctx.stage, parent)
        self._parent_paths.append(parent)
        self._frame_in_phase = 0
        self._enter_t = time.perf_counter()

    def _exit_phase(self, idx: int) -> None:
        phase = self.phases[idx]
        parent = self._phase_parent(phase)
        if phase.exit is not None:
            phase.exit(self.ctx.stage, parent)
        elif not phase.persistent:
            try:
                self.ctx.stage.RemovePrim(parent)
            except Exception:
                pass

    def run(self) -> None:
        if not self.phases:
            return
        self._install_keyboard()
        try:
            self._idx = 0
            self._enter_phase(self._idx)
            # Tick Kit once so Hydra ingests the new prims before sampling is_running()
            # (Kit auto-shuts the app down if its main loop is starved during a slow enter()).
            self.ctx.update()
            while self.ctx.is_running():
                phase = self.phases[self._idx]
                if phase.on_step is not None:
                    parent = self._phase_parent(phase)
                    phase.on_step(self.ctx.stage, parent, self._frame_in_phase, time.perf_counter())
                self.ctx.update()
                self._frame_in_phase += 1

                pending = self._pending
                if pending is None and phase.duration is not None:
                    if (time.perf_counter() - self._enter_t) >= phase.duration:
                        pending = "next"

                if pending is None:
                    continue

                self._pending = None
                if pending == "quit":
                    self._exit_phase(self._idx)
                    break
                if pending == "next":
                    self._exit_phase(self._idx)
                    if self._idx + 1 >= len(self.phases):
                        break
                    self._idx += 1
                    self._enter_phase(self._idx)
                elif pending == "prev":
                    self._exit_phase(self._idx)
                    self._idx = max(0, self._idx - 1)
                    self._enter_phase(self._idx)
        finally:
            self._uninstall_keyboard()

    def headless_play(self, durations: list[float] | None = None) -> None:
        """Iterate phases without a keyboard loop. ``durations[i]`` overrides ``phases[i].duration``;
        defaults to 1.0s. Useful for CI smoke-tests."""
        for i, phase in enumerate(self.phases):
            self._idx = i
            self._enter_phase(i)
            d = (durations[i] if durations is not None and i < len(durations) else None) or (
                phase.duration if phase.duration is not None else 1.0
            )
            t_end = time.perf_counter() + d
            while self.ctx.is_running() and time.perf_counter() < t_end:
                if phase.on_step is not None:
                    parent = self._phase_parent(phase)
                    phase.on_step(self.ctx.stage, parent, self._frame_in_phase, time.perf_counter())
                self.ctx.update()
                self._frame_in_phase += 1
            self._exit_phase(i)
