"""
Configuración global y fixtures para pytest.
"""

from unittest.mock import MagicMock
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def mock_voxcpm_for_unit_tests(monkeypatch):
    """
    Mockea VoxCPM.from_pretrained en pruebas unitarias para evitar
    descargas de gigabytes y ejecuciones pesadas en CPU/RAM durante la suite de tests.
    """
    mock_instance = MagicMock()

    def fake_generate(*args, **kwargs):
        sr = 48000
        dur = 1.5
        t = np.linspace(0, dur, int(sr * dur), endpoint=False, dtype=np.float32)
        return 0.2 * np.sin(2 * np.pi * 440 * t)

    mock_instance.generate.side_effect = fake_generate

    try:
        import voxcpm
        monkeypatch.setattr(voxcpm.VoxCPM, "from_pretrained", classmethod(lambda cls, *a, **kw: mock_instance))
    except Exception:
        pass
