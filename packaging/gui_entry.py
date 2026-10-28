"""Entry point for the standalone Windows executable.

The same executable runs the GUI and the background training worker. The GUI
re-invokes itself with ``--worker`` (see ``gui.runner``), which avoids needing
a separate Python interpreter in the packaged build.
"""

from __future__ import annotations

import sys


def main() -> int:
    if sys.argv[1:2] == ["--worker"]:
        from wakeword_trainer.gui.worker import main as worker_main

        return worker_main(sys.argv[2:])

    from wakeword_trainer.gui.app import main as gui_main

    gui_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
