# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyQt5-based GUI application for predicting magnetic field hysteresis in the CERN SPS (Super Proton Synchrotron) accelerator. The application provides real-time monitoring and prediction of magnetic field compensation for different magnetic elements (MBI main dipoles, QF focusing quadrupoles, QD defocusing quadrupoles).

## Development Commands

### Installation and Setup
```bash
# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e .[dev,test]
```

### UI Development
```bash
# Generate Python UI files from Qt Designer .ui files
python build_ui.py
```

### Code Quality
```bash
# Run linting
ruff check .

# Run type checking
mypy sps_apps/hysteresis_prediction/

# Run tests
pytest tests/
```

### Running the Application
```bash
# Run with required device parameter
app-hysteresis -d MBI --logdir /tmp/logs

# Run in online mode (no local predictions)
app-hysteresis -d QF --online --logdir /tmp/logs

# Run with different LSA server
app-hysteresis -d QD --lsa-server sps --logdir /tmp/logs
```

## Architecture Overview

### Core Components

1. **Application Entry Point**: `application.py` - Main application setup, argument parsing, and Qt application initialization
2. **Main Window**: `main_window.py` - Primary GUI interface built with PyQt5
3. **Data Flow**: `flow/` directory - Handles real-time data acquisition and processing
4. **Context System**: `contexts/` directory - Device-specific configurations and parameter management
5. **Widgets**: `widgets/` directory - Custom Qt widgets for different UI components

### Key Architectural Patterns

- **MVC Pattern**: Models, views, and controllers are separated in widget implementations
- **Context-Based Configuration**: Device-specific settings are managed through context objects
- **Signal-Slot Pattern**: Qt signals and slots for inter-component communication
- **Thread-Based Data Flow**: Background threads handle real-time data acquisition

### Important Directories

- `sps_apps/hysteresis_prediction/` - Main application code
- `resources/ui/` - Qt Designer UI files (.ui)
- `sps_apps/hysteresis_prediction/generated/` - Auto-generated Python UI files
- `sps_apps/hysteresis_prediction/local/event_building/` - Real-time event processing
- `sps_apps/hysteresis_prediction/widgets/` - Custom GUI components

### Data Flow Architecture

The application uses a multi-threaded architecture:
- **Main Thread**: GUI and user interaction
- **Data Thread**: Real-time data acquisition from CERN control systems
- **Worker Classes**: `LocalFlowWorker` and `UcapFlowWorker` handle different operation modes

### Device Context System

The application supports three magnetic element types:
- **MBI**: Main dipole magnets
- **QF**: Focusing quadrupole magnets  
- **QD**: Defocusing quadrupole magnets

Each device type has specific parameter names and configurations defined in the context system.

## Key Dependencies

- **PyQt5**: GUI framework
- **accwidgets**: CERN-specific Qt widgets
- **pyda**: CERN data acquisition library
- **torch**: Machine learning predictions
- **numpy/scipy**: Scientific computing
- **matplotlib**: Plotting capabilities

## Testing

- Main test files in `tests/` directory
- Additional test scripts in `test_scripts/` for specific functionality
- Use `pytest` for running tests
- Test coverage includes hysteresis prediction and track precycle functionality

## Generated Files

- UI files in `generated/` directory are auto-generated from `.ui` files
- Do not edit generated files directly - modify the source `.ui` files instead
- Run `python build_ui.py` after UI changes

## Configuration

- Device selection is required via `-d` parameter
- LSA server selection via `--lsa-server` (sps/next)
- Online vs offline mode affects prediction behavior
- Log directory must be specified for proper operation
