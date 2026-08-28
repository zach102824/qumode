"""Packaged numerical targets from the authors' published notebooks."""

from __future__ import annotations

import json
from importlib.resources import files

import numpy as np


def load_reference() -> dict:
    text = files("qumode_vqe.data").joinpath("reference.json").read_text(encoding="utf-8")
    data = json.loads(text)
    data["xvec"] = np.asarray(data["xvec_after_200_bfgs"], dtype=float)
    data["target_qnm"] = tuple(data["target_qnm"])
    return data
