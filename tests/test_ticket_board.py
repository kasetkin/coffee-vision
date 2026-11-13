"""coffeecv.ticket_board: the ticket board is generated from the tickets (rows, order, status, plan links), the repo's
own board is the generated one, and the hook's step puts the staged tickets' board into the index and the disk's
board onto the disk. Plain unittest.

    python -m unittest discover -s tests -p 'test_ticket_board.py'
"""
from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from coffeecv import ticket_board as tb
from coffeecv.ticket_check import nesting_errors


def ticket(tid: str, status: str = "Open", title: str = "A title") -> str:
    return (f'<title>{title}</title>\n<main>\n<div class="eyebrow">Ticket {tid} · coffee-vision · area</div>\n'
            f'<div class="field"><span class="k">Status</span><span class="v"><span class="pill open">{status}'
            f'</span> <span class="tag">written</span></span></div>\n</main>\n')


A, B, C = "ml-0007_ticket_alpha.html", "ml-0012_ticket_gamma.html", "ops-0002_ticket_beta.html"
A_PLAN = "ml-0007_plan_alpha.html"


def rows(board: str) -> list[str]:
    return [line.strip() for line in board.splitlines() if line.strip().startswith('<tr><td class="id">')]


class Render(unittest.TestCase):
    def setUp(self):
        self.docs = {A: ticket("ML-7", title="Alpha &amp; more"), A_PLAN: "<p>plan</p>\n",
                     B: ticket("ML-12", "In progress"), C: ticket("OPS-2", "Closed"),
                     "inference_path.html": "<p>not a ticket</p>\n", tb.BOARD: "<p>stale</p>\n"}

    def test_rows_newest_first_per_table_with_status_and_plan(self):
        board, found = tb.render(self.docs)
        self.assertEqual(found, [])
        self.assertEqual(rows(board), [
            f'<tr><td class="id">ML-12</td><td><a href="{B}">A title</a></td>'
            '<td><span class="pill open">In progress</span></td><td>&mdash;</td></tr>',
            f'<tr><td class="id">ML-7</td><td><a href="{A}">Alpha &amp; more</a></td>'
            f'<td><span class="pill open">Open</span></td><td><a href="{A_PLAN}">plan</a></td></tr>',
            f'<tr><td class="id">OPS-2</td><td><a href="{C}">A title</a></td>'
            '<td><span class="pill ok">Closed</span></td><td>&mdash;</td></tr>'])
        self.assertLess(board.index("ML &mdash;"), board.index("ML-12"))
        self.assertLess(board.index("ML-7"), board.index("OPS &mdash;"))
        self.assertEqual(nesting_errors(board), [])

    def test_the_board_does_not_depend_on_the_old_board(self):
        board, _ = tb.render(self.docs)
        self.docs[tb.BOARD] = board
        self.assertEqual(tb.render(self.docs)[0], board)

    def test_a_ticket_that_cannot_have_a_row_is_found(self):
        self.docs[B] = ticket("ML-12").replace("<title>A title</title>", "")
        self.docs[C] = ticket("OPS-2").replace("Ticket OPS-2", "OPS-2").replace('class="pill open"', 'class="x"')
        self.assertEqual([str(f) for f in tb.render(self.docs)[1]], [
            f"docs/{B}:1: no row on the ticket board without a <title>",
            f"docs/{C}:1: no row on the ticket board without one 'Ticket ML-n/OPS-n' eyebrow, a Status pill"])


class RepoBoard(unittest.TestCase):
    def test_the_repo_board_lists_every_ticket_as_the_tickets_say(self):
        docs = tb.working_tree()
        board, found = tb.render(docs)
        self.assertEqual(found, [])
        self.assertEqual(docs[tb.BOARD], board, "run `python -m coffeecv.ticket_board --write`")
        tickets = sorted(n for n in docs if "_ticket_" in n)
        self.assertEqual(len(rows(board)), len(tickets))
        for name in tickets:
            self.assertIn(f'<a href="{name}">', board)


class HookStep(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.docs = self.root / "docs"
        self.docs.mkdir()
        self.git("init", "-q")

    def tearDown(self):
        self.tmp.cleanup()

    def git(self, *args: str) -> str:  # init, add and show only: no identity needed
        return subprocess.run(["git", "-C", str(self.root), *args], capture_output=True, text=True,
                              check=True).stdout

    def write(self, name: str, text: str):
        (self.docs / name).write_text(text)

    def test_the_index_gets_the_staged_board_and_the_disk_the_disks(self):
        self.write(A, ticket("ML-7"))
        self.git("add", "docs")
        self.write(A, ticket("ML-7", "Closed"))  # changed after staging: not in this commit
        self.write(C, ticket("OPS-2"))           # new, not staged

        self.assertEqual(tb.update_staged(self.root), (True, []))
        in_index = self.git("show", f":docs/{tb.BOARD}")
        on_disk = (self.docs / tb.BOARD).read_text()
        self.assertEqual(in_index, tb.render({A: ticket("ML-7")})[0])
        self.assertEqual(on_disk, tb.render({A: ticket("ML-7", "Closed"), C: ticket("OPS-2")})[0])
        self.assertEqual(self.git("diff", "--cached", "--name-only").split(), [f"docs/{A}", f"docs/{tb.BOARD}"])
        self.assertEqual(tb.update_staged(self.root), (False, []))  # nothing left to do

    def test_a_hand_edit_to_the_staged_board_is_overwritten(self):
        self.write(A, ticket("ML-7"))
        self.write(tb.BOARD, "<p>edited by hand</p>\n")
        self.git("add", "docs")
        self.assertEqual(tb.update_staged(self.root), (True, []))
        self.assertEqual(self.git("show", f":docs/{tb.BOARD}"), tb.render({A: ticket("ML-7")})[0])

    def test_a_finding_writes_nothing(self):
        self.write(A, ticket("ML-7").replace("<title>A title</title>", ""))
        self.git("add", "docs")
        changed, found = tb.update_staged(self.root)
        self.assertEqual((changed, len(found)), (False, 1))
        self.assertFalse((self.docs / tb.BOARD).exists())
        self.assertEqual(self.git("ls-files", "docs").split(), [f"docs/{A}"])


if __name__ == "__main__":
    unittest.main()
