"""Thin kconfiglib menuconfig wrapper.

Usage: python tools/menuconfig.py <Kconfig-root-file>
Respects KCONFIG_CONFIG env var for .config path.
"""

import sys
import os
import importlib

if len(sys.argv) < 2:
    print("Usage: python tools/menuconfig.py <Kconfig>", file=sys.stderr)
    sys.exit(1)

os.environ.setdefault("KCONFIG_CONFIG", ".config")

# Remove this script's directory from sys.path to avoid circular import
# (this file is named menuconfig.py, same as the system module).
script_dir = os.path.dirname(os.path.abspath(__file__))
original_path = sys.path[:]
sys.path = [p for p in sys.path if os.path.abspath(p) != script_dir]

_menuconfig = importlib.import_module("menuconfig")

sys.path = original_path

_menuconfig._main()
