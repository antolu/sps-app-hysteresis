"""
This script is used to generate the UI files for the application.
"""
from os import path

import pyqt5ac

HERE = path.split(path.abspath(__file__))[0]
PACKAGE_DIR = "sps_apps/btrain_data"


pyqt5ac.main(
    uicOptions="--from-imports",
    force=False,
    initPackage=True,
    ioPaths=[
        [
            path.join(HERE, "resources/ui/*.ui"),
            path.join(HERE, PACKAGE_DIR, "generated/%%FILENAME%%_ui.py"),
        ],
    ],
)
