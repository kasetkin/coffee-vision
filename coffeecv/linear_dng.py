"""A DNG whose main image is LinearRaw compressed with JPEG XL (DNG 1.7), read and developed (ticket ML-31 D1).

LibRaw 0.22 (rawpy 0.27) cannot open such a file: Samsung's Galaxy Z Fold8 Ultra writes its RAW photos so, a merged,
demosaiced picture of three 16-bit channels, each strip or tile a JPEG XL codestream (compression 52546), which
imagecodecs decodes. Every other RAW stays LibRaw's: `parse` claims only this kind and answers None for the rest.

`develop` follows LibRaw's steps for a linear DNG as canonicalization asks LibRaw for them (canonical.decode:
use_camera_wb, Rec.2020 output, linear, 16-bit, auto-bright on), from dcraw's code, which LibRaw keeps:
  1. black level subtracted and the white level scaled to 1 (BlackLevel per sample and its repeat pattern);
  2. the camera's white balance, 1 / AsShotNeutral, scaled so its smallest multiplier is 1, then clipped at 1
     (LibRaw's highlight mode 0);
  3. the colour matrix: XYZ -> camera of the D65 illuminant if the file has one (else ColorMatrix2, else 1), times
     CameraCalibration and AnalogBalance as DNG 1.7 composes them; with Rec.2020's primaries, its rows normalised
     so camera white is output white, inverted (dcraw's cam_xyz_coeff), and clipped to 0-1;
  4. auto-bright: the 16-bit output's histogram in 8192 bins, per channel the value above which 1% of the pixels
     lie, the largest of them made white (dcraw's auto_bright_thr 0.01), linear, clipped.
Not applied, as LibRaw does not apply them either: the opcode lists (the Fold8 Ultra's GainMap opcodes are flat),
the ProfileGainTableMap and BaselineExposure (ML-29 D18). The result can be no closer to LibRaw's than its steps:
LibRaw has nothing to compare it with (ML-31 R1).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import imagecodecs as ic
import numpy as np

JPEG_XL, LINEAR_RAW, D65 = 52546, 34892, 21
# XYZ (D65) of linear Rec.2020 and of linear sRGB (BT.709) primaries: the output spaces asked for.
XYZ_FROM = {"rec2020": np.array([[0.636958, 0.144617, 0.168881], [0.262700, 0.677998, 0.059302],
                                 [0.000000, 0.028073, 1.060985]]),
            "srgb": np.array([[0.412391, 0.357584, 0.180481], [0.212639, 0.715169, 0.072192],
                              [0.019331, 0.119195, 0.950532]])}
TYPES = {1: "B", 2: "s", 3: "H", 4: "I", 5: "II", 7: "B", 8: "h", 9: "i", 10: "ii", 11: "f", 12: "d", 16: "Q"}
SIZES = {"B": 1, "s": 1, "H": 2, "I": 4, "II": 8, "h": 2, "i": 4, "ii": 8, "f": 4, "d": 8, "Q": 8}


@dataclass(frozen=True)
class LinearDng:
    """What `develop` needs, read from the file's main image IFD (and IFD0's Orientation)."""
    data: bytes
    width: int
    height: int
    orientation: int
    pieces: tuple[tuple[int, int, int, int], ...]   # (offset, length, row, column) of each strip or tile
    piece_size: tuple[int, int]                    # (rows, columns) of a strip or tile
    black: np.ndarray                              # (rows, columns, 3) repeat pattern
    white: np.ndarray                              # (3,)
    colour_matrix: np.ndarray                      # XYZ -> camera, 3 x 3, calibration and balance applied
    neutral: np.ndarray                            # AsShotNeutral, (3,)
    unsupported: str = ""                          # a tag develop refuses (a LinearizationTable)


def parse(data: bytes) -> LinearDng | None:
    """The file as a JPEG XL LinearRaw DNG, or None if it is not one (another RAW, a broken TIFF, not a TIFF)."""
    try:
        return _parse(data)
    except (struct.error, IndexError, KeyError, ValueError, ZeroDivisionError):
        return None


def develop(dng: LinearDng, output: str = "rec2020") -> np.ndarray:
    """The main image developed (the module's steps), as stored, not turned: uint16 (h, w, 3), linear, in
    Rec.2020 primaries ("rec2020") or sRGB's ("srgb"). ValueError for a layout this reader does not take."""
    if dng.unsupported:
        raise ValueError(f"a JPEG XL LinearRaw DNG with {dng.unsupported}")
    light = _stored(dng)                                          # float32 (h, w, 3), 0 - 1 of the white level
    mult = 1 / dng.neutral
    light *= (mult / mult.min()).astype(np.float32)
    np.minimum(light, 1, out=light)
    camera_from_out = dng.colour_matrix @ XYZ_FROM[output]
    camera_from_out /= camera_from_out.sum(axis=1, keepdims=True)
    out = light @ np.linalg.inv(camera_from_out).T.astype(np.float32)
    np.clip(out, 0, 1, out=out)
    o16 = np.rint(out * 65535).astype(np.uint16)
    return _auto_bright(o16)


def _auto_bright(o16: np.ndarray) -> np.ndarray:
    """dcraw's auto-bright with a linear curve: per channel, counting 8192-bin histogram bins down from the top until
    more than 1% of the pixels are counted; the largest such bin times 8 is white."""
    total = o16.shape[0] * o16.shape[1]
    white = 0
    for c in range(3):
        hist = np.bincount((o16[..., c] >> 3).ravel(), minlength=8192)
        counted = np.cumsum(hist[::-1])                         # pixels at or above each bin, from the top
        above = 8191 - int(np.argmax(counted > total * 0.01))
        white = max(white, above)
    top = max(white, 32) << 3
    scaled = np.minimum(o16.astype(np.float32) * (65536 / top), 65535)
    return np.floor(scaled).astype(np.uint16)


def samples(dng: LinearDng) -> np.ndarray:
    """The main image's stored samples, uint16 (h, w, 3), its strips or tiles decoded and put together: what the
    strip's pixel check compares (strip_metadata.pixel_digest), as LibRaw's raw_image is for another RAW."""
    out = np.empty((dng.height, dng.width, 3), np.uint16)
    rows, cols = dng.piece_size
    for offset, length, y, x in dng.pieces:
        piece = ic.jpegxl_decode(dng.data[offset:offset + length])
        if piece.ndim != 3 or piece.shape[2] != 3 or piece.dtype != np.uint16:
            raise ValueError(f"a JPEG XL piece of {piece.dtype} {piece.shape}, not three 16-bit channels")
        h, w = min(rows, dng.height - y), min(cols, dng.width - x)
        out[y:y + h, x:x + w] = piece[:h, :w]
    return out


def _stored(dng: LinearDng) -> np.ndarray:
    """The stored samples, black subtracted and divided by white - black, float32 (h, w, 3)."""
    out = samples(dng).astype(np.float32)
    r, c = dng.black.shape[:2]
    black = np.tile(dng.black, (-(-dng.height // r), -(-dng.width // c), 1))[:dng.height, :dng.width]
    out -= black
    out /= (dng.white - dng.black.mean(axis=(0, 1))).astype(np.float32)
    np.maximum(out, 0, out=out)
    return out


# ----- the TIFF structure -----

def _ifd(d: bytes, end: str, at: int) -> dict[int, tuple]:
    count = struct.unpack(end + "H", d[at:at + 2])[0]
    tags = {}
    for i in range(count):
        tag, kind, n, raw = struct.unpack(end + "HHI4s", d[at + 2 + 12 * i:at + 14 + 12 * i])
        code = TYPES.get(kind)
        if code is None:
            continue
        size = SIZES[code] * n
        body = raw[:size] if size <= 4 else d[struct.unpack(end + "I", raw)[0]:][:size]
        if code == "s":
            tags[tag] = (body.rstrip(b"\0").decode("latin1"),)
        else:
            values = struct.unpack(end + code * n, body)
            if code in ("II", "ii"):
                values = tuple(values[k] / values[k + 1] for k in range(0, len(values), 2))
            tags[tag] = values
    return tags


def data_end(data: bytes) -> int:
    """Where the image data of a TIFF (a DNG) ends: the end of the last strip or tile of IFD0 and its SubIFDs (the
    main image and the previews), or 0 for a file that is not a TIFF. What a trailer cut must not reach into
    (strip_metadata, ML-31)."""
    try:
        if data[:4] not in (b"II*\0", b"MM\0*"):
            return 0
        end = "<" if data[:2] == b"II" else ">"
        ifd0 = _ifd(data, end, struct.unpack(end + "I", data[4:8])[0])
        last = 0
        for ifd in [ifd0] + [_ifd(data, end, o) for o in ifd0.get(330, ())]:
            for offsets, counts in ((273, 279), (324, 325), (513, 514)):
                for o, n in zip(ifd.get(offsets, ()), ifd.get(counts, ())):
                    last = max(last, o + n)
        return last
    except (struct.error, IndexError, ValueError):
        return len(data)                         # unreadable: nothing may be cut


def _parse(d: bytes) -> LinearDng | None:
    if d[:4] not in (b"II*\0", b"MM\0*"):
        return None
    end = "<" if d[:2] == b"II" else ">"
    ifd0 = _ifd(d, end, struct.unpack(end + "I", d[4:8])[0])
    if 50706 not in ifd0:                                        # DNGVersion: not a DNG
        return None
    main = ifd0 if ifd0.get(254, (1,))[0] == 0 else None
    for offset in ifd0.get(330, ()):                             # SubIFDs
        sub = _ifd(d, end, offset)
        if main is None and sub.get(254, (1,))[0] == 0:
            main = sub
    if main is None or main.get(259, (0,))[0] != JPEG_XL or main.get(262, (0,))[0] != LINEAR_RAW:
        return None
    if main.get(277, (1,))[0] != 3:
        raise ValueError("a LinearRaw main image without three samples")
    width, height = main[256][0], main[257][0]
    if 324 in main:                                              # tiles
        rows, cols = main[323][0], main[322][0]
        across = -(-width // cols)
        pieces = tuple((o, n, (i // across) * rows, (i % across) * cols)
                       for i, (o, n) in enumerate(zip(main[324], main[325])))
    else:
        rows, cols = main.get(278, (height,))[0], width
        pieces = tuple((o, n, i * rows, 0) for i, (o, n) in enumerate(zip(main[273], main[279])))
    repeat = main.get(50713, (1, 1))
    black = np.array(main.get(50714, (0,)), np.float32)
    black = np.broadcast_to(black, (repeat[0] * repeat[1] * 3,)) if black.size == 1 else black
    black = black.reshape(repeat[0], repeat[1], 3)
    white = np.array(main.get(50717, (65535,)), np.float32)
    white = np.broadcast_to(white, (3,)).copy()
    return LinearDng(d, width, height, ifd0.get(274, (1,))[0], pieces, (rows, cols), black, white,
                     _colour_matrix(ifd0, main), np.array(ifd0.get(50728) or main.get(50728) or (1, 1, 1), np.float64),
                     "a LinearizationTable" if 50712 in main else "")


def _colour_matrix(ifd0: dict, main: dict) -> np.ndarray:
    """XYZ -> camera (DNG 1.7, Chapter 6): AnalogBalance x CameraCalibration x ColorMatrix, of the D65 illuminant if
    the file has one, else of its second illuminant, else of its first."""
    def tag(t: int):
        return ifd0.get(t) or main.get(t)
    illuminants = [tag(50778), tag(50779)]
    chosen = next((i for i, v in enumerate(illuminants) if v and v[0] == D65), 1 if tag(50722) else 0)
    matrix = np.array(tag((50721, 50722)[chosen]), np.float64).reshape(3, 3)
    calibration = tag((50723, 50724)[chosen])
    if calibration:
        matrix = np.array(calibration, np.float64).reshape(3, 3) @ matrix
    balance = tag(50727)
    if balance:
        matrix = np.diag(balance) @ matrix
    return matrix
