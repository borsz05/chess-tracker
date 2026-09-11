"""A háromszoros ismétlés felismerése — regressziós teszt a Board.copy()-ra.

A chess_logic.game.Board korábban felülírta a python-chess `copy(*, stack=True)`
metódusát egy `Board(self.fen())`-t adó változattal, ami ELNYELTE a
lépéstörténetet. A MoveRecord így egy történet nélküli másolaton kérdezte az
ismétlést, ezért a parti sosem ért véget háromszoros ismétléssel: a
`result_after_move` `*` maradt, és a felület sem tudta kiírni a lezárás okát.
"""
import chess

from chess_logic.game import Board, Game

# Nf3 Nf6 Ng1 Ng8 kétszer: a nyolcadik féllépés az alapállást harmadszor adná.
REPETITION_LINE = ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"]


def test_board_copy_keeps_move_stack():
    board = Board()
    for uci in REPETITION_LINE[:4]:
        board.push(chess.Move.from_uci(uci))

    copied = board.copy()
    assert copied.move_stack == board.move_stack
    assert copied.is_repetition(2) == board.is_repetition(2)
    assert isinstance(copied, Board) and copied.side_to_move == board.side_to_move


def test_threefold_repetition_ends_the_game():
    game = Game(chess.STARTING_FEN)
    applied = [uci for uci in REPETITION_LINE if game.apply_uci(uci) is not None]

    # A claim_draw=True az ISMÉTELHETŐ állásban zár le, egy féllépéssel azelőtt,
    # hogy az alapállás harmadszor a táblán lenne — a nyolcadik féllépés tehát
    # már nem kerül a partiba.
    assert applied == REPETITION_LINE[:7]
    assert game.result == "1/2-1/2"

    last = game.move_history[-1]
    assert last.result_after_move == "1/2-1/2"
    # A felület EBBŐL írja ki, hogy "Háromszori ismétlés" (frontend/js/utils/game.js).
    assert last.is_threefold_repetition is True
    assert game.to_state_dict()["status"]["is_threefold_repetition"] is True


def test_normal_opening_is_not_a_repetition_draw():
    game = Game(chess.STARTING_FEN)
    for uci in ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]:
        assert game.apply_uci(uci) is not None

    assert game.result == "*"
    assert not any(record.is_threefold_repetition for record in game.move_history)
