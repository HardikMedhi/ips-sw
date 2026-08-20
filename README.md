# Interplanetary Scintillation (IPS) Analysis Toolkit

This repository contains the core analysis code for my PhD work on interplanetary scintillation (IPS) and solar wind studies. It provides utilities for reading and processing filterbank data, computing spectra, handling time/coordinate geometry, applying RFI mitigation, and generating scientific plots.

This package is intended to support downstream analysis workflows and related projects such as SHIPS and ITACHI, which build on the shared functionality provided here.

## Overview

The library is organized around a few main themes:

- Filterbank reading and handling in `ips_sw/classes/`
- Time, geometry, and coordinate utilities in `ips_sw/utils/`
- Power spectrum analysis in `ips_sw/power_spectra/`
- RFI detection and cleaning in `ips_sw/rfi/`
- Plotting and visualization in `ips_sw/plotting/`
- Telescope metadata and configuration in `ips_sw/yaml_info/`

## Project structure

```text
ips_sw/
├── classes/
│   └── filterbank.py
├── plotting/
│   ├── pipeline_plot_ds.py
│   ├── pipeline_plot_ps.py
│   └── ...
├── power_spectra/
│   └── power_spec.py
├── rfi/
│   ├── rfi_iqr.py
│   ├── rfi_ransac.py
│   └── timeseries_rfi.py
├── utils/
│   ├── geometry_utils.py
│   ├── time_utils.py
│   ├── file_utils.py
│   └── ...
├── yaml_info/
│   └── tel_info.yaml
├── matplotlib_styles/
│   └── style_paper.mplstyle
├── tests/
│   └── ...
└── __init__.py
```

## Installation

Clone the repository and install it in editable mode from the project root:

```bash
git clone <repo-url>
cd ips-main
pip install -e .
```

Once installed, the package can be imported as `ips_sw`.

## Example usage

```python
from ips_sw.classes.filterbank import Filterbank

fb = Filterbank("path/to/data.fil")
# continue with processing, calibration, or plotting workflows
```

Additional helpers can be imported from the utilities package, for example:

```python
from ips_sw.utils.geometry_utils import get_solar_elongation
```

## Notes

- This project is currently under active development.
- The package name and some interfaces may evolve as the analysis pipeline matures.
- The repository is intended to be a shared core library for related IPS analysis projects.

## Author

Hardik Medhi
