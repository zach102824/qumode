"""Bit partition of a growing transmon × cavity register (Fock cutoff 8).

Locked n=7…11 map (do not change)::

    n=7          T0 ⊕ C0 ⊕ C1          (1T×2C production subspace)
    n=8…11       T0,T1 + C0,C1[,C2]    (2T×3C, C2 bits 0…3)

    bit 0        T0     transmon, dim 2
    bit 1        T1     transmon, dim 2   (n≥8)
    bits 2–4     C0     cavity, 8 Fock levels, 3 bits (Fock binary, MSB first)
    bits 5–7     C1
    bits 8–10    C2

When n exceeds the current capacity, append **one transmon and one cavity**
(cutoff still 8). Capacities: 2T+3C=11, 3T+4C=15, 4T+5C=19, …

    n=12         add T2; leftover 9 bits fill C0–C2; C3 idle
                 simulate 3T×3C, dim 2³·8³=4096, pairs=9
    n=13…15      C3 gets 1,2,3 bits → 3T×4C, dim 2³·8⁴=32768, pairs=12
    n=16…19      add T3+C4; C4 idle at n=16 (4T×4C), live at n=17…19

Idle modes (0 assigned bits) stay vacuum and are omitted from the simulated
tensor — same rule as C2 at n=8.

Fock decoding uses the lowest ``n_bits`` of the occupation, listed MSB-first
among those bits — identical to production ``bits_from_qnm`` when ``n_bits=3``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import (
    FOCK_BITS,
    FOCK_CUTOFF,
    HARDWARE_DIM,
    MAX_LOGICAL_BITS,
    N_CAVITIES,
    N_TRANSMONS,
    hardware_plan,
    plan_capacity,
    plan_hardware_dim,
)


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
        n_t_plan, n_c_plan = hardware_plan(self.n_qubits)
        plan_dim = (
            HARDWARE_DIM if self.n_qubits == 7 else plan_hardware_dim(n_t_plan, n_c_plan)
        )
        return {
            "n_qubits": self.n_qubits,
            "dim": self.dim,
            "hardware_dim": plan_dim,
            "max_logical_bits": MAX_LOGICAL_BITS,
            "hardware_plan": {
                "n_transmons": int(n_t_plan if self.n_qubits != 7 else N_TRANSMONS),
                "n_cavities": int(n_c_plan if self.n_qubits != 7 else N_CAVITIES),
                "capacity": int(
                    plan_capacity(N_TRANSMONS, N_CAVITIES)
                    if self.n_qubits == 7
                    else plan_capacity(n_t_plan, n_c_plan)
                ),
                "simulated_n_transmons": self.n_transmons,
                "simulated_n_cavities": self.n_cavities,
            },
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
            "subspace_of_hardware": self.dim < plan_dim,
        }


def cavity_bits_for_n(n: int) -> tuple[int, ...]:
    """Logical bits on C0, C1, … for the hardware plan at this n (idle cavities = 0)."""
    n = int(n)
    n_t, n_c = hardware_plan(n)
    leftover = n - n_t
    if leftover < 0:
        raise ValueError(f"n={n} cannot use {n_t} transmons")
    assigned = [0] * n_c
    for i in range(n_c):
        take = min(FOCK_BITS, leftover)
        assigned[i] = take
        leftover -= take
    if leftover != 0:
        raise ValueError(f"n={n} does not fit the {n_t}T×{n_c}C×{FOCK_BITS}bit map")
    return tuple(assigned)


def _cavity_bits_for_n(n: int) -> tuple[int, int, int]:
    """Back-compat: C0,C1,C2 bits for n=8…11 (T0+T1 always on)."""
    bits = cavity_bits_for_n(n)
    if len(bits) != N_CAVITIES:
        raise ValueError(f"n={n} is not on the 2T×3C register")
    return bits[0], bits[1], bits[2]


def embedding_for_n(n: int) -> Embedding:
    """Active-mode embedding for one system size (idle modes omitted)."""
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
        n_t, _n_c = hardware_plan(n)
        cav_bits = cavity_bits_for_n(n)
        spec_list: list[tuple[str, str, int, int]] = [
            (f"T{i}", "transmon", 2, 1) for i in range(n_t)
        ]
        for i, n_bits in enumerate(cav_bits):
            if n_bits > 0:
                spec_list.append((f"C{i}", "cavity", FOCK_CUTOFF, n_bits))
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


def hardware_register_names(n: int) -> list[str]:
    """Modes present in the hardware plan (including idle vacuum modes).

    n=7 is reported against the parent 2T×3C register (T1 and C2 idle),
    matching the locked PR #15 map.
    """
    n = int(n)
    if n == 7:
        n_t, n_c = N_TRANSMONS, N_CAVITIES
    else:
        n_t, n_c = hardware_plan(n)
    return [f"T{i}" for i in range(n_t)] + [f"C{i}" for i in range(n_c)]


def hardware_idle_modes(n: int) -> list[str]:
    """Modes present in the hardware plan but omitted (vacuum) for this n."""
    live = {m.name for m in embedding_for_n(n).modes}
    return [name for name in hardware_register_names(n) if name not in live]
