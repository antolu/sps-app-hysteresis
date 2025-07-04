"""
This script is used to generate the UI files for the application.
"""

import glob
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
PACKAGE_DIR = "sps_app_hysteresis"
UI_DIR = HERE / "resources" / "ui"
GENERATED_DIR = HERE / PACKAGE_DIR / "generated"


def generate_ui_files():
    """Generate Python UI files from Qt Designer .ui files using PyQt6."""

    # Ensure the generated directory exists
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    # Create __init__.py in generated directory
    init_file = GENERATED_DIR / "__init__.py"
    init_file.touch()

    # Find all .ui files
    ui_files = glob.glob(str(UI_DIR / "*.ui"))

    for ui_file in ui_files:
        ui_path = Path(ui_file)
        output_name = f"{ui_path.stem}_ui.py"
        output_path = GENERATED_DIR / output_name

        print(f"Generating {output_name} from {ui_path.name}")

        # Use PyQt6's uic tool to convert .ui to .py
        try:
            subprocess.run(
                [
                    "python",
                    "-m",
                    "PyQt6.uic.pyuic",
                    str(ui_path),
                    "-o",
                    str(output_path),
                ],
                check=True,
            )
            print(f"Successfully generated {output_name}")
        except subprocess.CalledProcessError as e:
            print(f"Error generating {output_name}: {e}")
        except FileNotFoundError:
            print("PyQt6 uic not found. Please install PyQt6-tools")
            return False

    return True


if __name__ == "__main__":
    success = generate_ui_files()
    if success:
        print("UI generation completed successfully")
    else:
        print("UI generation failed")
