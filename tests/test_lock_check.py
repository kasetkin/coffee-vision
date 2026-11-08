"""coffeecv.lock_check (ML-19 D41): a package that two of the three uv locks hold must be held at one version, and the
repo's own locks agree (D35). Plain unittest on small lock files written here, and on the real locks.

    python -m unittest discover -s tests -p 'test_lock_check.py'
"""
from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from coffeecv import lock_check as lk

REPO = Path(__file__).resolve().parents[1]


def lock(*packages: tuple[str, str] | tuple[str, str, str]) -> str:
    """A uv.lock with these (name, version[, source]) packages."""
    out = ['version = 1', 'requires-python = ">=3.12"', ""]
    for name, version, *source in packages:
        src = source[0] if source else '{ registry = "https://pypi.org/simple" }'
        out += ["[[package]]", f'name = "{name}"', f'version = "{version}"', f"source = {src}", ""]
    return "\n".join(out)


class LockCheck(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def write(self, name: str, text: str) -> Path:
        path = self.dir / name
        path.write_text(text)
        return path

    def check(self, **locks: str) -> list[lk.Mismatch]:
        return lk.mismatches({label: lk.read_versions(self.write(f"{label}.lock", text))
                              for label, text in locks.items()})

    def test_the_same_versions_pass(self):
        self.assertEqual(self.check(a=lock(("numpy", "2.5.3"), ("pillow", "12.3.0")),
                                    b=lock(("pillow", "12.3.0"), ("numpy", "2.5.3"))), [])

    def test_a_package_at_two_versions_is_named_with_each_lock_and_version(self):
        found = self.check(training=lock(("pillow", "12.3.0"), ("numpy", "2.5.3")),
                           web=lock(("pillow", "12.2.0"), ("numpy", "2.5.3")))
        self.assertEqual(found, [lk.Mismatch("pillow", {"training": ("12.3.0",), "web": ("12.2.0",)})])

    def test_a_package_in_one_lock_only_is_not_compared(self):
        self.assertEqual(self.check(a=lock(("torch", "2.9.0")), b=lock(("flask", "3.1.3"))), [])

    def test_every_mismatch_is_reported_across_three_locks(self):
        found = self.check(a=lock(("click", "8.5.0"), ("werkzeug", "3.1.9")),
                           b=lock(("click", "8.5.0"), ("werkzeug", "3.1.8")),
                           c=lock(("click", "8.4.2")))
        self.assertEqual([m.name for m in found], ["click", "werkzeug"])
        self.assertEqual(found[0].versions, {"a": ("8.5.0",), "b": ("8.5.0",), "c": ("8.4.2",)})

    def test_names_are_compared_normalized(self):
        found = self.check(a=lock(("typing_extensions", "4.16.0")), b=lock(("Typing.Extensions", "4.15.0")))
        self.assertEqual([m.name for m in found], ["typing-extensions"])

    def test_a_package_held_at_two_versions_in_one_lock_is_compared_as_the_set(self):
        both = lock(("numpy", "2.5.3"), ("numpy", "2.4.4"))
        self.assertEqual(self.check(a=both, b=both), [])
        self.assertEqual(len(self.check(a=both, b=lock(("numpy", "2.5.3")))), 1)

    def test_the_projects_themselves_are_not_compared(self):
        self.assertEqual(self.check(a=lock(("coffee-cv", "0.1.0", '{ virtual = "." }')),
                                    b=lock(("coffee-cv", "0.2.0", '{ editable = "." }'))), [])

    def test_the_report_names_the_package_and_each_locks_version(self):
        text = lk.report([lk.Mismatch("pillow", {"uv.lock": ("12.3.0",), "webapp/uv.lock": ("12.2.0",)})])
        self.assertIn("pillow", text)
        self.assertIn("uv.lock 12.3.0", text)
        self.assertIn("webapp/uv.lock 12.2.0", text)

    def test_main_fails_on_a_mismatch_and_passes_without(self):
        good = self.write("good.lock", lock(("numpy", "2.5.3")))
        same = self.write("same.lock", lock(("numpy", "2.5.3")))
        bad = self.write("bad.lock", lock(("numpy", "2.4.4")))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(lk.main([str(good), str(same)]), 0)
        self.assertIn("1 packages shared by 2 locks", out.getvalue())
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(lk.main([str(good), str(bad)]), 1)
        self.assertIn("numpy", out.getvalue())


class RepoLocks(unittest.TestCase):
    def test_the_three_locks_hold_one_version_of_every_shared_package(self):
        found = lk.mismatches({p: lk.read_versions(REPO / p) for p in lk.LOCKS})
        self.assertEqual(found, [], lk.report(found))

    def test_the_lists_locks_exist_and_convert_is_not_among_them(self):
        for p in lk.LOCKS:
            self.assertTrue((REPO / p).is_file(), p)
        self.assertNotIn("convert/uv.lock", lk.LOCKS)


if __name__ == "__main__":
    unittest.main()
