"""Resolve separately installed controller resources without vendoring them."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SONIC_ROOT = Path(os.environ.get('SONIC_ROOT', ROOT / 'external/sonic')).expanduser().resolve()
DEPLOY = SONIC_ROOT / 'gear_sonic_deploy'
