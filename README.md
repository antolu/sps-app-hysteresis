# SPS App Hysteresis

A PyQt5-based GUI application for predicting and monitoring magnetic field hysteresis in the CERN SPS (Super Proton Synchrotron) accelerator. This application provides real-time monitoring and prediction of magnetic field compensation for different magnetic elements in the SPS beam line.

## Overview

The SPS Hysteresis Prediction application is designed to:

- Monitor and predict magnetic field hysteresis effects in SPS magnetic elements
- Provide real-time correction calculations for magnetic field compensation
- Support different magnetic element types: Main Dipoles (MBI), Focusing Quadrupoles (QF), and Defocusing Quadrupoles (QD)
- Interface with CERN's control systems for data acquisition and trim application
- Offer both online and offline prediction modes

## Features

- **Real-time Data Acquisition**: Interfaces with CERN's JAPC and LSA systems
- **Machine Learning Predictions**: Uses PyTorch models for hysteresis prediction
- **Multi-Device Support**: Configurable for different magnetic elements
- **GUI Interface**: Built with PyQt5 and custom CERN widgets
- **Logging and Metrics**: Comprehensive logging with optional Tensorboard integration
- **RBAC Integration**: Authentication through CERN's Role-Based Access Control

## Installation

### Prerequisites

- Python 3.11 or 3.12
- Access to CERN network and control systems
- Valid CERN authentication credentials

### Install from Source

```bash
# Clone the repository
git clone https://gitlab.cern.ch/sps-apps/sps-app-hysteresis.git
cd sps-app-hysteresis

# Install the package
pip install -e .

# Install with development dependencies
pip install -e .[dev,test]
```

## Usage

### Basic Usage

The application requires a device parameter to specify which magnetic element to monitor:

```bash
# Run for main dipole magnets
app-hysteresis -d MBI --logdir /tmp/logs

# Run for focusing quadrupoles
app-hysteresis -d QF --logdir /tmp/logs

# Run for defocusing quadrupoles
app-hysteresis -d QD --logdir /tmp/logs
```

### Available Command Line Flags

#### Required Parameters

- `-d, --device {MBI,QF,QD}` - Device to apply field compensation to
  - `MBI`: SPS main dipoles
  - `QF`: SPS focusing quadrupoles
  - `QD`: SPS defocusing quadrupoles

#### Optional Parameters

- `-v` - Increase verbosity (use multiple times for more verbose output)
- `-b, --buffer-size INTEGER` - Buffer size for data acquisition (default: 60000)
- `--online` - Enable online prediction and trim monitoring (no local predictions)
- `--lsa-server {sps,next}` - LSA server to use (default: next)
- `--logdir PATH` - Directory to save logs (required)
- `--metrics-writer {txt,tensorboard}` - Metrics writer type (default: txt)

#### Usage Examples

```bash
# Basic usage with main dipoles
app-hysteresis -d MBI --logdir /var/log/hysteresis

# Verbose output with custom buffer size
app-hysteresis -d QF -vv --buffer-size 100000 --logdir /tmp/logs

# Online mode with SPS LSA server
app-hysteresis -d QD --online --lsa-server sps --logdir /tmp/logs

# With Tensorboard metrics
app-hysteresis -d MBI --metrics-writer tensorboard --logdir /tmp/logs
```

## Development

### UI Development

The application uses Qt Designer for UI design. After modifying `.ui` files:

```bash
# Generate Python UI files
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

### Project Structure

```
sps_apps/hysteresis_prediction/
├── application.py          # Main application entry point
├── main_window.py         # Primary GUI window
├── contexts/              # Device-specific configurations
├── flow/                  # Data flow and acquisition
├── widgets/               # Custom GUI widgets
├── local/                 # Local prediction and event building
├── trim/                  # Trim calculation and settings
├── history/               # Prediction history management
├── io/                    # Input/output and metrics
└── generated/             # Auto-generated UI files
```

## Architecture

The application follows a multi-threaded architecture:

- **Main Thread**: Handles GUI interactions and display
- **Data Thread**: Manages real-time data acquisition from CERN systems
- **Context System**: Provides device-specific parameter configurations
- **Signal-Slot Pattern**: Qt-based communication between components

## Requirements

- Python 3.11+
- PyQt5 ~= 5.12
- PyTorch ~= 2.5.1
- NumPy >= 1.26, < 2.0
- CERN-specific libraries (accwidgets, pyda, pjlsa)
- Machine learning dependencies (lightning, scikit-learn)

## License

Other/Proprietary License - CERN

## Author

Anton Lu (anton.lu@cern.ch)

## Links

- [Project Repository](https://gitlab.cern.ch/sps-apps/sps-app-hysteresis)
- [Documentation](https://acc-py.web.cern.ch/gitlab/dsb/applications/sps-app-hysteresis/docs/stable/)
