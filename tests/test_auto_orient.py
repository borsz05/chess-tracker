"""A gyűjtő automatikus orientáció-igazítása.

A board_detector rácsának orientációja nem garantált: ugyanaz a fizikai tábla
más rácsállást is adhat, a tools/fen_labels viszont FIX leképezést feltételez.
A gyűjtő ezért megkeresi, melyik szimmetria illeszkedik, és ahhoz igazítja a
címkézést. Ezek a tesztek azt rögzítik, hogy az igazítás matematikailag
helyes — enélkül némán elrontott címkéket mentenénk.

Az éles pipeline (vision.pipeline.tracker.raw_to_standard) NEM érintett.
"""
from __future__ import annotations

import numpy as np
import pytest

from tools.collect_fen_dataset import ModelChecker
from tools.fen_labels import fen_to_raw_labels, fen_to_raw_occupancy

FEN = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
ALL_SYMS = ("rot0", "rot0+flip", "rot90", "rot90+flip",
            "rot180", "rot180+flip", "rot270", "rot270+flip")


@pytest.mark.parametrize("name", ALL_SYMS)
def test_apply_symmetry_matches_symmetries(name):
    """Az apply_symmetry ugyanazt csinálja, mint amit a check keres."""
    grid = np.arange(64).reshape(8, 8)
    expected = dict(ModelChecker.symmetries(grid))[name]
    assert np.array_equal(ModelChecker.apply_symmetry(grid, name), expected)


def test_apply_symmetry_on_object_grid():
    """Objektum-rácson (SquareLabel) is ugyanaz, mint numpy tömbön."""
    lst = [[f"{r}{c}" for c in range(8)] for r in range(8)]
    got = ModelChecker.apply_symmetry(lst, "rot90")
    exp = np.rot90(np.array(lst, dtype=object), 1)
    assert all(got[r][c] == exp[r, c] for r in range(8) for c in range(8))


@pytest.mark.parametrize("rotation", ALL_SYMS)
def test_rotated_grid_is_recovered(rotation, monkeypatch):
    """A detektor bármely rácsállásából vissza kell kapnunk a helyes címkéket.

    A modellt kiváltjuk: pontosan azt "látja", ami a fizikai valóság — vagyis
    a FEN foglaltságát az adott rácsállásban. A checknek meg kell találnia a
    szimmetriát, és az azzal igazított címkéknek egyezniük kell a valósággal.
    """
    truth_occ = fen_to_raw_occupancy(FEN)
    truth_labels = fen_to_raw_labels(FEN)
    reality = ModelChecker.apply_symmetry(truth_occ, rotation)

    checker = ModelChecker.__new__(ModelChecker)          # __init__ nélkül: nem kell modellfájl
    monkeypatch.setattr(checker, "predict_occupancy", lambda rois: reality, raising=False)

    res = checker.check(rois=None, fen_occ=truth_occ)

    fixed = ModelChecker.apply_symmetry(truth_labels, res["best_symmetry"])
    fixed_occ = np.array([[lab.occ for lab in row] for row in fixed], dtype=np.int32)

    assert res["best_n"] == 0, f"{rotation}: nem talalta meg a szimmetriat"
    assert np.array_equal(fixed_occ, reality), f"{rotation}: az igazitott cimke nem egyezik a valosaggal"


def test_identity_is_preferred_when_correct():
    """Ha az alap leképezés jó, ne igazítson feleslegesen."""
    truth_occ = fen_to_raw_occupancy(FEN)
    checker = ModelChecker.__new__(ModelChecker)
    checker.predict_occupancy = lambda rois: truth_occ  # type: ignore[method-assign]
    res = checker.check(rois=None, fen_occ=truth_occ)
    assert res["best_symmetry"] == "rot0"
    assert res["n_mismatch"] == 0
