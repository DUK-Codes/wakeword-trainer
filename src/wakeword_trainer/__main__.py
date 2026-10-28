"""``python -m wakeword_trainer`` launches the desktop app.

The package is a library first; this module exists so that a checkout or a
wheel can open the GUI without remembering the console-script name.
"""

from __future__ import annotations


def main() -> None:
    from .gui.app import main as gui_main

    gui_main()


if __name__ == "__main__":
    main()
