"""Bit partition of the 2-transmon × 3-cavity × 8-level register.

Canonical 11-bit map (bit 0 = MSB of the logical string when all modes are live)::

    bit 0        T0     transmon, dim 2
    bit 1        T1     transmon, dim 2
    bits 2–4     C0     cavity, 8 Fock levels, 3 bits (Fock binary, MSB first)
    bits 5–7     C1
    bits 8–10    C2

n=11 is an exact fill of this 2048-dimensional space.
n=8…10 drop trailing C2 bits (then C2 itself once it has 0 bits).
n=7 uses the production-like subspace T0 ⊕ C0 ⊕ C1 (skip T1 and C2), dim 128.

Fock decoding uses the lowest ``n_bits`` of the occupation, listed MSB-first
among those bits — identical to production ``bits_from_qnm`` when ``n_bits=3``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import FOCK_CUTOFF, HARDWARE_DIM, MAX_LOGICAL_BITS, N_CAVITIES, N_TRANSMONS


@dataclass(frozen=True)
class Mode:
    name: str
    kind: str  # "transmon" | "cavity"
    dim: int
    n_bits: int
    axis: int


@dataclass(frozen=True)
class Embedding:
    n_qubits: int
    modes: tuple[Mode, ...]

    @property
    def dims(self) -> tuple[int, ...]:
        return tuple(m.dim for m in self.modes)

    @property
    def dim(self) -> int:
        out = 1
        for d in self.dims:
            out *= d
        return out

    @property
    def n_transmons(self) -> int:
        return sum(1 for m in self.modes if m.kind == "transmon")

    @property
    def n_cavities(self) -> int:
        return sum(1 for m in self.modes if m.kind == "cavity")

    @property
    def n_prep_params(self) -> int:
        return self.n_transmons + 2 * self.n_cavities

    @property
    def n_pairs(self) -> int:
        return self.n_transmons * self.n_cavities

    def transmon_axes(self) -> tuple[int, ...]:
        return tuple(m.axis for m in self.modes if m.kind == "transmon")

    def cavity_axes(self) -> tuple[int, ...]:
        return tuple(m.axis for m in self.modes if m.kind == "cavity")

    def ecd_pairs(self) -> tuple[tuple[int, int], ...]:
        """(transmon_axis, cavity_axis) pairs, cavity-major then transmon.

        n=7 (1T×2C) yields (T0,C0), (T0,C1) — the production UER order.
        """
        t_axes = self.transmon_axes()
        c_axes = self.cavity_axes()
        return tuple((t, c) for c in c_axes for t in t_axes)

    def decode_occupations(self, occupations: Sequence[int]) -> np.ndarray:
        """Map a hybrid basis label to the n-bit logical string (MSB first)."""
        if len(occupations) != len(self.modes):
            raise ValueError(f"expected {len(self.modes)} occupations, got {len(occupations)}")
        bits: list[int] = []
        for occ, mode in zip(occupations, self.modes, strict=True):
            n = int(occ)
            if not 0 <= n < mode.dim:
                raise ValueError(f"{mode.name} occupation {n} outside dim {mode.dim}")
            if mode.kind == "transmon":
                if mode.n_bits:
                    bits.append(n & 1)
                continue
            for k in range(mode.n_bits - 1, -1, -1):
                bits.append((n >> k) & 1)
        if len(bits) != self.n_qubits:
            raise RuntimeError(f"decoded {len(bits)} bits, expected {self.n_qubits}")
        return np.asarray(bits, dtype=np.int64)

    def encode_bits(self, bits: Sequence[int]) -> tuple[int, ...]:
        """Canonical computational embedding of an n-bit string (unused Fock bits 0)."""
        x = np.asarray(bits, dtype=int).reshape(self.n_qubits)
        if not np.all(np.isin(x, (0, 1))):
            raise ValueError("bits must be 0/1")
        occ: list[int] = []
        offset = 0
        for mode in self.modes:
            if mode.kind == "transmon":
                if mode.n_bits:
                    occ.append(int(x[offset]))
                    offset += 1
                else:
                    occ.append(0)
                continue
            chunk = x[offset : offset + mode.n_bits]
            offset += mode.n_bits
            n = 0
            for b in chunk:
                n = (n << 1) | int(b)
            if n >= mode.dim:
                raise ValueError(f"Fock {n} does not fit {mode.name} dim {mode.dim}")
            occ.append(n)
        if offset != self.n_qubits:
            raise RuntimeError("bit/mode alignment mismatch")
        return tuple(occ)

    def bitstring(self, bits: Sequence[int]) -> str:
        return "".join(str(int(b)) for b in np.asarray(bits).reshape(self.n_qubits))

    def decode_index(self, index: int) -> np.ndarray:
        occ = np.unravel_index(int(index), self.dims)
        return self.decode_occupations(occ)

    def as_dict(self) -> dict:
        return {
            "n_qubits": self.n_qubits,
            "dim": self.dim,
            "hardware_dim": HARDWARE_DIM,
            "max_logical_bits": MAX_LOGICAL_BITS,
            "dims": list(self.dims),
            "n_transmons": self.n_transmons,
            "n_cavities": self.n_cavities,
            "n_prep_params": self.n_prep_params,
            "n_pairs": self.n_pairs,
            "ecd_pairs": [
                {"transmon_axis": int(t), "cavity_axis": int(c)} for t, c in self.ecd_pairs()
            ],
            "modes": [
                {
                    "name": m.name,
                    "kind": m.kind,
                    "dim": m.dim,
                    "n_bits": m.n_bits,
                    "axis": m.axis,
                }
                for m in self.modes
            ],
            "bit_partition": bit_partition_doc(self),
            "subspace_of_2048": self.dim < HARDWARE_DIM,
        }


def _cavity_bits_for_n(n: int) -> tuple[int, int, int]:
    """How many logical bits sit on C0, C1, C2 for n=8…11 (T0+T1 always on)."""
    leftover = int(n) - N_TRANSMONS
    if leftover < 0:
        raise ValueError(f"n={n} cannot use both transmons")
    caps = [3, 3, 3]
    assigned = [0, 0, 0]
    for i in range(N_CAVITIES):
        take = min(caps[i], leftover)
        assigned[i] = take
        leftover -= take
    if leftover != 0:
        raise ValueError(f"n={n} does not fit the 2T×3C×3bit map")
    return assigned[0], assigned[1], assigned[2]


def embedding_for_n(n: int) -> Embedding:
    """Active-mode embedding for one system size."""
    n = int(n)
    if n < 7 or n > MAX_LOGICAL_BITS:
        raise ValueError(f"this study covers n=7…{MAX_LOGICAL_BITS}, got {n}")

    modes: list[Mode] = []
    if n == 7:
        # Production-compatible subspace: 1 transmon + 2 cavities (1+3+3).
        spec = (
            ("T0", "transmon", 2, 1),
            ("C0", "cavity", FOCK_CUTOFF, 3),
            ("C1", "cavity", FOCK_CUTOFF, 3),
        )
    else:
        c0, c1, c2 = _cavity_bits_for_n(n)
        spec_list: list[tuple[str, str, int, int]] = [
            ("T0", "transmon", 2, 1),
            ("T1", "transmon", 2, 1),
            ("C0", "cavity", FOCK_CUTOFF, c0),
            ("C1", "cavity", FOCK_CUTOFF, c1),
        ]
        if c2 > 0:
            spec_list.append(("C2", "cavity", FOCK_CUTOFF, c2))
        spec = tuple(spec_list)

    for axis, (name, kind, dim, n_bits) in enumerate(spec):
        if n_bits <= 0:
            continue
        modes.append(Mode(name=name, kind=kind, dim=int(dim), n_bits=int(n_bits), axis=axis))

    remapped = []
    for i, m in enumerate(modes):
        remapped.append(Mode(name=m.name, kind=m.kind, dim=m.dim, n_bits=m.n_bits, axis=i))
    emb = Embedding(n_qubits=n, modes=tuple(remapped))
    if sum(m.n_bits for m in emb.modes) != n:
        raise RuntimeError(f"bit sum {sum(m.n_bits for m in emb.modes)} != n={n}")
    return emb


def bit_partition_doc(emb: Embedding) -> list[dict]:
    """Human-readable map from logical bit index → mode."""
    rows: list[dict] = []
    bit = 0
    for mode in emb.modes:
        for k in range(mode.n_bits):
            if mode.kind == "transmon":
                label = mode.name
            else:
                # MSB-first among the assigned Fock bits.
                fock_bit = mode.n_bits - 1 - k
                label = f"{mode.name}[fock bit {fock_bit}]"
            rows.append({"bit": bit, "mode": mode.name, "label": label, "kind": mode.kind})
            bit += 1
    return rows


def hardware_idle_modes(n: int) -> list[str]:
    """Modes present in the 2048-dim hardware but omitted (vacuum) for this n."""
    live = {m.name for m in embedding_for_n(n).modes}
    all_names = ["T0", "T1", "C0", "C1", "C2"]
    return [name for name in all_names if name not in live]
