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

import sys
import time
import chess

# ── Demo kalibrációs értékek ─────────────────────────────────────────────────
# Ezek HOZZÁVETŐLEGES értékek egy tipikus Franka elrendezésnél.
# A1 és H1 sarok mm-ben, a robot báziskeretéhez képest.
# Valódi roboton a python -m robot.calibrate paranccsal kell felmérni!
DEMO_A1 = (450.0, -175.0)   # x előre, y balra
DEMO_H1 = (450.0,  175.0)   # x előre, y jobbra (7 * ~50mm = 350mm távolság)
DEMO_GRAVEYARD_START = (560.0, -200.0)
DEMO_GRAVEYARD_STEP  = ( 40.0,    0.0)


def _init_robot():
    print("Robot inicializálása (pymoveit2 + MoveIt2)...")
    from robot.impl import RobotImpl
    robot = RobotImpl()
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
        cal      = Calibration(a1=DEMO_A1, h1=DEMO_H1)
        graveyard = Graveyard(start=DEMO_GRAVEYARD_START, step=DEMO_GRAVEYARD_STEP)
    return cal, graveyard


# ── Tesztek ──────────────────────────────────────────────────────────────────

def test_connection(robot):
    print("─" * 50)
    print("TEST: Kapcsolat és pozíció lekérdezés")
    print("─" * 50)
    pos = robot.get_position()
    print(f"  End-effector pozíció:")
    print(f"    x = {pos[0]:+8.2f} mm")
    print(f"    y = {pos[1]:+8.2f} mm")
    print(f"    z = {pos[2]:+8.2f} mm")


def test_gripper(robot):
    print("─" * 50)
    print("TEST: Gripper nyit / zár / nyit")
    print("─" * 50)
    print("  → Nyitás...")
    robot.open_gripper()
    print("  → Zárás (fogás)...")
    robot.close_gripper()
    print("  → Nyitás (visszaállítás)...")
    robot.open_gripper()
    print("  Gripper teszt kész.")


def test_move_xyz(robot):
    print("─" * 50)
    print("TEST: Mozgás kézzel megadott koordinátákra")
    print("─" * 50)
    print("  Add meg a célpozíciót mm-ben (robot báziskerete).")
    print("  Tipikus tartományok: x=300–600, y=-300–300, z=50–400")
    try:
        x = float(input("  x (mm): "))
        y = float(input("  y (mm): "))
        z = float(input("  z (mm): "))
    except ValueError:
        print("  Érvénytelen szám, teszt kihagyva.")
        return

    print(f"  → Mozgás: x={x:.0f} y={y:.0f} z={z:.0f} mm")
    robot.move_to(x, y, z)
    pos = robot.get_position()
    print(f"  Elért pozíció: x={pos[0]:.1f} y={pos[1]:.1f} z={pos[2]:.1f} mm")


def test_square(robot):
    print("─" * 50)
    print("TEST: Mozgás sakktábla mezőre")
    print("─" * 50)
    cal, _ = _load_calibration()

    sq = input("  Mező neve (pl. e4, a1, h8): ").strip().lower()
    if len(sq) != 2 or sq[0] not in "abcdefgh" or sq[1] not in "12345678":
        print("  Érvénytelen mező.")
        return

    xy = cal.square_to_xy(sq)
    print(f"  {sq.upper()} fizikai koordináta: x={xy[0]:.1f} y={xy[1]:.1f} mm")
    print(f"  → Mozgás 250 mm magasságra a mező fölé...")
    robot.move_to(xy[0], xy[1], 250)
    pos = robot.get_position()
    print(f"  Elért pozíció: x={pos[0]:.1f} y={pos[1]:.1f} z={pos[2]:.1f} mm")


def test_pick_place(robot):
    print("─" * 50)
    print("TEST: Teljes pick-and-place két mező között")
    print("─" * 50)
    cal, _ = _load_calibration()

    from_sq = input("  Forrás mező (pl. e2): ").strip().lower()
    to_sq   = input("  Cél mező   (pl. e4): ").strip().lower()

    from_xy = cal.square_to_xy(from_sq)
    to_xy   = cal.square_to_xy(to_sq)
    print(f"  {from_sq.upper()} → {to_sq.upper()}")
    print(f"  Forrás: x={from_xy[0]:.1f} y={from_xy[1]:.1f} mm")
    print(f"  Cél:    x={to_xy[0]:.1f} y={to_xy[1]:.1f} mm")
    confirm = input("  Indítás? [i/N]: ").strip().lower()
    if confirm != "i":
        print("  Megszakítva.")
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
    print("  Pick-and-place kész.")


def test_chess_move(robot, uci: str | None = None):
    print("─" * 50)
    print("TEST: Teljes sakklépés (descriptor → execute_move)")
    print("─" * 50)
    cal, graveyard = _load_calibration()

    if uci is None:
        uci = input("  UCI lépés (pl. e2e4, e1g1, d5e6): ").strip().lower()

    board = chess.Board()
    try:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            print(f"  '{uci}' nem legális a kezdőállásból.")
            return
    except ValueError:
        print(f"  Érvénytelen UCI: '{uci}'")
        return

    from robot.move_descriptor import build_move_descriptor
    descriptor = build_move_descriptor(uci, board, cal, graveyard)

    print(f"  Típus:  {descriptor['type']}")
    print(f"  Forrás: {descriptor['piece_from_xy']}")
    print(f"  Cél:    {descriptor['piece_to_xy']}")
    if descriptor["captured_xy"]:
        print(f"  Ütött:  {descriptor['captured_xy']} → temető: {descriptor['graveyard_xy']}")

    confirm = input("  Végrehajtás? [i/N]: ").strip().lower()
    if confirm != "i":
        print("  Megszakítva.")
        return

    robot.execute_move(descriptor)
    print("  Lépés végrehajtva.")


def test_heights(robot):
    print("─" * 50)
    print("TEST: Z magasságok ellenőrzése")
    print("─" * 50)
    print("  A magasságok a chess_executor.py-ban állíthatók (méterben):")
    print("    Z_TRAVEL = utazó magasság")
    print("    Z_PICK   = fogási magasság")
    print("    Z_PLACE  = letételi magasság")
    print()
    cal, _ = _load_calibration()
    sq = input("  Melyik mező fölé tesztelünk? (pl. e4): ").strip().lower()
    xy = cal.square_to_xy(sq)

    z_travel = float(input("  Z_TRAVEL mm (javasolt 250): ") or "250")
    print(f"  → Mozgás {z_travel:.0f} mm magasságra...")
    robot.move_to(xy[0], xy[1], z_travel)
    input("    Ellenőrizd. Enter a folytatáshoz...")

    z_pick = float(input("  Z_PICK mm (javasolt 40): ") or "40")
    print(f"  → Mozgás {z_pick:.0f} mm magasságra (fogás)...")
    robot.move_to(xy[0], xy[1], z_pick)
    input("    Ellenőrizd hogy a bábu közepénél van-e. Enter...")

    print(f"  → Visszatérés {z_travel:.0f} mm-re...")
    robot.move_to(xy[0], xy[1], z_travel)
    print()
    print(f"  Ha jók az értékek, írd be a chess_executor.py-ba:")
    print(f"    Z_TRAVEL = {z_travel/1000:.3f}")
    print(f"    Z_PICK   = {z_pick/1000:.3f}")


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
        print(MENU)
        choice = input("Választás: ").strip().lower()

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
