#!/usr/bin/env python3
"""
Franka FR3 szimulációs teszteszköz.

Futtatás (MoveIt2 már fut a Dockerben):
    source /opt/ros/jazzy/setup.bash
    source ~/ws_moveit/install/setup.bash
    source .venv/bin/activate
    python test_robot.py

A DEMO_CALIBRATION alább csak tájékoztató jellegű koordinátákat tartalmaz.
Igazi kísérletekhez előbb futtasd: python -m robot.calibrate
"""
from __future__ import annotations

import chess

# ── Demo kalibrációs értékek ─────────────────────────────────────────────────
# Ezek HOZZÁVETŐLEGES értékek egy tipikus Franka elrendezésnél.
# A1 és H1 sarok mm-ben, a robot báziskeretéhez képest.
# Valódi roboton a python -m robot.calibrate paranccsal kell felmérni!
DEMO_A1 = (450.0, -175.0)   # A1 sarok: x előre, y balra
DEMO_H8 = (100.0,  175.0)   # H8 sarok: átlós, 7×50mm fájl + 7×50mm sor
DEMO_GRAVEYARD_START = (560.0, -200.0)
DEMO_GRAVEYARD_STEP  = ( 40.0,    0.0)


def _init_robot():
    print("Robot inicializálása (pymoveit2 + MoveIt2)...")
    from robot.impl import FrankaRobot
    robot = FrankaRobot()
    print("Robot kész.\n")
    return robot


def _load_calibration():
    from robot.calibration import Calibration
    from robot.graveyard import Graveyard
    try:
        cal      = Calibration.load()
        graveyard = Graveyard.from_calibration_file()
        print("Kalibrációs fájl betöltve.")
    except FileNotFoundError:
        print("Kalibrációs fájl nem található — demo koordinátákat használok.")
        cal      = Calibration(a1=DEMO_A1, h8=DEMO_H8)
        graveyard = Graveyard(start=DEMO_GRAVEYARD_START, step=DEMO_GRAVEYARD_STEP)
    return cal, graveyard


# ── Tesztek ──────────────────────────────────────────────────────────────────

def test_connection(robot):
    pos = robot.get_position()
    print(f"Pozíció: x={pos[0]:.1f}  y={pos[1]:.1f}  z={pos[2]:.1f} mm")


def test_gripper(robot):
    print("Gripper nyitás...")
    robot.open_gripper()
    print("Gripper zárás...")
    robot.close_gripper()
    print("Gripper nyitás...")
    robot.open_gripper()
    print("Kész.")


def test_move_xyz(robot):
    print("Célpozíció (mm):")
    try:
        x = float(input("  x: "))
        y = float(input("  y: "))
        z = float(input("  z: "))
    except ValueError:
        print("Érvénytelen szám.")
        return

    print(f"Mozgás: x={x:.0f}  y={y:.0f}  z={z:.0f} mm")
    robot.move_to(x, y, z)
    pos = robot.get_position()
    print(f"Elért pozíció: x={pos[0]:.1f}  y={pos[1]:.1f}  z={pos[2]:.1f} mm")


def test_square(robot):
    cal, _ = _load_calibration()

    sq = input("Mező (pl. e4): ").strip().lower()
    if len(sq) != 2 or sq[0] not in "abcdefgh" or sq[1] not in "12345678":
        print("Érvénytelen mező.")
        return

    xy = cal.square_to_xy(sq)
    print(f"{sq.upper()}: x={xy[0]:.1f}  y={xy[1]:.1f} mm")
    print("Mozgás a mező fölé (z=250 mm)...")
    robot.move_to(xy[0], xy[1], 250)
    pos = robot.get_position()
    print(f"Elért pozíció: x={pos[0]:.1f}  y={pos[1]:.1f}  z={pos[2]:.1f} mm")


def test_pick_place(robot):
    cal, _ = _load_calibration()

    from_sq = input("Forrás mező (pl. e2): ").strip().lower()
    to_sq   = input("Cél mező (pl. e4): ").strip().lower()

    from_xy = cal.square_to_xy(from_sq)
    to_xy   = cal.square_to_xy(to_sq)
    print(f"{from_sq.upper()} → {to_sq.upper()}")
    confirm = input("Indítás? [i/N]: ").strip().lower()
    if confirm != "i":
        print("Megszakítva.")
        return

    robot.execute_move({
        "type": "simple",
        "piece_from_xy": list(from_xy),
        "piece_to_xy": list(to_xy),
        "captured_xy": None,
        "graveyard_xy": None,
        "castling_rook": None,
        "promotion": None,
    })
    print("Kész.")


def test_chess_move(robot, uci: str | None = None):
    cal, graveyard = _load_calibration()

    if uci is None:
        uci = input("UCI lépés (pl. e2e4): ").strip().lower()

    board = chess.Board()
    try:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            print(f"'{uci}' nem legális a kezdőállásból.")
            return
    except ValueError:
        print(f"Érvénytelen UCI: '{uci}'")
        return

    from robot.move_descriptor import build_move_descriptor
    descriptor = build_move_descriptor(uci, board, cal, graveyard)

    print(f"Lépés: {descriptor['type']}  {descriptor['piece_from_xy']} → {descriptor['piece_to_xy']}")
    confirm = input("Végrehajtás? [i/N]: ").strip().lower()
    if confirm != "i":
        print("Megszakítva.")
        return

    robot.execute_move(descriptor)
    print("Kész.")


def test_heights(robot):
    cal, _ = _load_calibration()
    sq = input("Mező (pl. e4): ").strip().lower()
    xy = cal.square_to_xy(sq)

    z_travel = float(input("Utazó magasság mm (pl. 250): ") or "250")
    robot.move_to(xy[0], xy[1], z_travel)
    input("Enter a folytatáshoz...")

    z_pick = float(input("Fogási magasság mm (pl. 40): ") or "40")
    robot.move_to(xy[0], xy[1], z_pick)
    input("Enter a folytatáshoz...")

    robot.move_to(xy[0], xy[1], z_travel)
    print(f"Z_TRAVEL = {z_travel/1000:.3f}  Z_PICK = {z_pick/1000:.3f}")


# ── Menü ─────────────────────────────────────────────────────────────────────

MENU = """
╔══════════════════════════════════════════╗
║     FRANKA FR3 — SZIMULÁCIÓS TESZTEK     ║
╠══════════════════════════════════════════╣
║  1. Kapcsolat + pozíció lekérdezés       ║
║  2. Gripper nyit/zár teszt               ║
║  3. Mozgás XYZ koordinátákra (mm)        ║
║  4. Mozgás sakkmező fölé (pl. e4)        ║
║  5. Pick-and-place két mező között       ║
║  6. Teljes sakklépés (pl. e2e4)          ║
║  7. Z magasságok ellenőrzése             ║
║  q. Kilépés                              ║
╚══════════════════════════════════════════╝"""

def run():
    robot = _init_robot()

    while True:
        choice = input("\nTeszt (1-7, q=kilépés): ").strip().lower()

        try:
            if   choice == "1": test_connection(robot)
            elif choice == "2": test_gripper(robot)
            elif choice == "3": test_move_xyz(robot)
            elif choice == "4": test_square(robot)
            elif choice == "5": test_pick_place(robot)
            elif choice == "6": test_chess_move(robot)
            elif choice == "7": test_heights(robot)
            elif choice == "q": print("Kilépés."); break
            else: print("Ismeretlen opció.")
        except KeyboardInterrupt:
            print("\nMegszakítva.")
        except Exception as e:
            print(f"Hiba: {e}")

        input("\nEnter a menühöz...")


if __name__ == "__main__":
    run()
