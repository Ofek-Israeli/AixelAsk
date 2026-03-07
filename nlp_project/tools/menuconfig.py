"""Thin kconfiglib menuconfig wrapper.

Usage: python tools/menuconfig.py <Kconfig-root-file>
Respects KCONFIG_CONFIG env var for .config path.
"""

import sys

import kconfiglib
from kconfiglib import menuconfig

if len(sys.argv) < 2:
    print("Usage: python tools/menuconfig.py <Kconfig>", file=sys.stderr)
    sys.exit(1)

menuconfig(kconfiglib.Kconfig(sys.argv[1]))
