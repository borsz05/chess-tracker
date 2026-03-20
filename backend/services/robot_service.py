from __future__ import annotations

from typing import Any

FILES = "abcdefgh"


def square_name_to_coords(square_name: str) -> dict[str, int]:
    """
    Példa:
    e2 -> {"row": 6, "col": 4}
    """
    file_char = square_name[0]
    rank_char = square_name[1]

    col = FILES.index(file_char)
    row = 8 - int(rank_char)

    return {"row": row, "col": col}


def uci_to_robot_payload(uci: str) -> dict[str, Any]:
    """
    Alap payload robotkarhoz.
    Később bővíthető:
    - capture kezelés
    - promotion kezelés
    - pick/place magasság
    - home pozíció
    """
    uci = uci.strip()

    if len(uci) < 4:
        raise ValueError("Invalid UCI for robot payload")

    from_sq = uci[:2]
    to_sq = uci[2:4]
    promotion = uci[4] if len(uci) > 4 else None

    return {
        "uci": uci,
        "from_square": from_sq,
        "to_square": to_sq,
        "from": square_name_to_coords(from_sq),
        "to": square_name_to_coords(to_sq),
        "promotion": promotion,
    }


def best_move_payload_from_top_lines(top_lines: list[dict] | None) -> dict | None:
    """
    A legjobb engine sor első lépését alakítja robotbarát payload-dá.
    A fen most még nincs használva, de később hasznos lehet capture/promóció
    részletesebb kezeléséhez.
    """
    if not top_lines:
        return None

    first_line = top_lines[0]
    pv_uci = first_line.get("pv_uci") or []
    if not pv_uci:
        return None

    best_uci = pv_uci[0]

    payload = uci_to_robot_payload(best_uci)
    payload["score"] = first_line.get("score")
    payload["mate_in"] = first_line.get("mate_in")
    payload["line_san"] = first_line.get("line_san")

    return payload
