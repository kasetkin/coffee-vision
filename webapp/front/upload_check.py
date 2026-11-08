"""The front's check of an upload's format (tickets OPS-15 P1, OPS-18; ADR 0034): is this file in an accepted format?

    refusal("IMG_0001.HEIC", first_bytes)   # None, or a reason a visitor can read

The accepted formats are coffeecv/data/accepted_formats.csv, and which one a file is in is coffeecv.canonical's
recogniser, the one canonicalization dispatches on (OPS-18 D17): a name with a RAW extension must start with that RAW
family's signature, and any other file is told by its first bytes alone. A file in none is refused here, before any
decoder sees it (D3), even when one could read it (a plain TIFF, D2). This module words the answer: the refusal names
the accepted formats from the file (D20). A file that passes may still fail to decode; canonicalization answers that
one with its own 400.
"""
from __future__ import annotations

from pathlib import PurePath

from coffeecv import canonical

HEAD_BYTES = canonical.HEAD_BYTES   # what the recogniser reads


def named_formats(conjunction: str, list_groups: bool = False) -> str:
    """The accepted formats by name, in the file's order, a group's rows gathered under its name: "JPEG, PNG, ... or
    camera RAW", or with `list_groups` "... or camera RAW (DNG, CR2, ...)". The page and the agent files name them
    the same way (tests/test_accepted_formats_named.py)."""
    names: list[str] = []
    groups: dict[str, list[str]] = {}
    for f in canonical.ACCEPTED_FORMATS.values():
        if not f.group:
            names.append(f.label)
            continue
        if f.group not in groups:
            groups[f.group] = []
            names.append(f.group)
        groups[f.group].append(f.label)
    if list_groups:
        names = [f"{n} ({', '.join(groups[n])})" if n in groups else n for n in names]
    return f"{', '.join(names[:-1])} {conjunction} {names[-1]}"


NOT_A_PHOTO = f"not a {named_formats('or')} file"


def refusal(filename: str, head: bytes) -> str | None:
    """None if `head`, the file's first HEAD_BYTES bytes, is in an accepted format; else why not."""
    if canonical.recognise(head, filename) is not None:
        return None
    ext = PurePath(filename).suffix.lower()
    if ext in canonical.RAW_EXTENSIONS:
        return f"the file is named {ext} but is not a {ext[1:].upper()} camera RAW file"
    return NOT_A_PHOTO
