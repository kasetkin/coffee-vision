"""coffeecv.leak_check: public IPs, emails, credentials, private photo metadata and the owner's own patterns
are found in what git would publish; the things the owner allows (SSH aliases, the DVC remote URL,
internal addresses, package versions) are not. Plain unittest on strings and on a throwaway git repo.

    python -m unittest discover -s tests -p 'test_leak_check.py'

The image test needs exiftool (libimage-exiftool-perl, in the devcontainer).
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from coffeecv import leak_check as lc

HAVE_EXIFTOOL = shutil.which("exiftool") is not None
# Assembled at run time: this file is published too, and leak_check must find nothing in it.
PUBLIC_IP = "93.184." + "216.34"
EMAIL = "me.name+tag" + "@" + "mail.example-isp.fi"
DB_URL = "postgres://umami:s3cret" + "@" + "localhost/umami"


def kinds(text: str, private=()) -> list[tuple[int, str, str]]:
    return [(f.line, f.kind, f.shown) for f in lc.text_findings("f.txt", text.encode(), list(private))]


class TextFindings(unittest.TestCase):
    def test_public_ip_found_with_its_line(self):
        self.assertEqual(kinds(f"a\nthe VM is at {PUBLIC_IP}."), [(2, "public IP address", PUBLIC_IP)])

    def test_internal_addresses_and_versions_pass(self):
        text = ("bridge 172.17.0.3, lan 192.168.1.10, 10.0.0.1, 127.0.0.1:8000\n"
                'opencv-python-headless==5.0.0.93\nclick_plugins-1.1.1.2-py2\nversion = "1.1.1.2"\n'
                "v1.2.3.4 and 1.2.3.4.5 and 0.9688")
        self.assertEqual(kinds(text), [])

    def test_allowed_ip_passes(self):
        self.assertEqual(kinds("opencv 5.0.0.93. Both install"), [])

    def test_email_found_and_allowed_ones_pass(self):
        text = (f"{EMAIL}\n"
                "Co-Authored-By: Claude <noreply@anthropic.com>\n123+me@users.noreply.github.com\n"
                "git@github.com:owner/repo.git\nuser@example.com\n@real_data\n@unittest.skipUnless(x)")
        self.assertEqual(kinds(text), [(1, "email address", EMAIL)])

    def test_escape_before_at_is_not_a_local_part(self):
        self.assertEqual(kinds(r"doc = 'x\n@unittest.skipUnless(y)'"), [])

    def test_credentials_found_and_masked(self):
        token = "ghp_" + "a1B2" * 9
        found = kinds(f"x\ntoken = '{token}'\nurl = {DB_URL}")
        self.assertEqual(found, [(2, "GitHub token", "ghp_a1..."), (3, "password in URL", "://uma...")])

    def test_ssh_and_dvc_urls_pass(self):
        self.assertEqual(kinds("url = ssh://alioth@powervpsssh/home/alioth/dvc-store\nhttps://coffee.kasetkin.com/"), [])

    def test_private_patterns(self):
        private = [lc.re.compile(rb"secret-street", lc.re.IGNORECASE)]
        self.assertEqual(kinds("ok\nlives on Secret-Street 5", private),
                         [(2, "private pattern", "pattern 'secret-street'")])


class PatternsFile(unittest.TestCase):
    def test_absent_file_is_none_and_comments_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "leak_patterns.txt"
            self.assertIsNone(lc.load_private_patterns(path))
            path.write_text("# a comment\n\n  home-town  \n")
            self.assertEqual([p.pattern for p in lc.load_private_patterns(path)], [b"home-town"])


class Scan(unittest.TestCase):
    """scan() over a real git repo: tracked and untracked files count, ignored ones do not."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, rel: str, data: bytes) -> None:
        (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
        (self.root / rel).write_bytes(data)

    def test_untracked_counts_ignored_and_binary_do_not(self):
        self.write(".gitignore", b"private/\n")
        self.write("private/notes.md", f"the VM is {PUBLIC_IP}".encode())
        self.write("docs/new.md", f"mail me: {EMAIL}".encode())
        self.write("blob.bin", b"\0" + PUBLIC_IP.encode())
        self.write("home-town/readme.md", b"fine")
        private = [lc.re.compile(rb"home-town")]
        found = [(f.path, f.kind) for f in lc.scan(self.root, private)]
        self.assertEqual(sorted(found), [("docs/new.md", "email address"),
                                         ("home-town/readme.md", "private pattern in file path")])

    def test_staged_sees_only_the_index(self):
        self.write("staged.md", f"mail me: {EMAIL}".encode())
        self.write("unstaged.md", f"the VM is {PUBLIC_IP}".encode())
        subprocess.run(["git", "-C", str(self.root), "add", "staged.md"], check=True)
        files = lc.published_files(self.root, staged=True)
        self.assertEqual(files, ["staged.md"])
        self.assertEqual([(f.path, f.kind) for f in lc.scan(self.root, files=files)],
                         [("staged.md", "email address")])

    @unittest.skipUnless(HAVE_EXIFTOOL, "needs exiftool")
    def test_photo_gps_found_plain_plot_passes(self):
        arr = np.random.default_rng(0).integers(0, 255, (32, 48, 3), dtype=np.uint8)
        Image.fromarray(arr).save(self.root / "photo.jpg", quality=90)
        Image.fromarray(arr).save(self.root / "plot.png")
        subprocess.run(["exiftool", "-q", "-overwrite_original", "-GPSLatitude=48.1", "-GPSLatitudeRef=N",
                        str(self.root / "photo.jpg")], check=True)
        found = lc.scan(self.root)
        self.assertEqual([(f.path, f.kind) for f in found], [("photo.jpg", "private photo metadata")])
        self.assertIn("GPSLatitude", found[0].shown)


if __name__ == "__main__":
    unittest.main()
