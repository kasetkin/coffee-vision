"""ML-19 D52: libjxl's chunked (streaming) encoder, reached through ctypes from the libjxl inside imagecodecs' wheel.

imagecodecs' jpegxl_encode hands libjxl the whole frame through JxlEncoderAddImageFrame, which copies it
(encode.cc: "we need to create an internal copy of the frame data") and keeps it for the whole encode. With
JxlEncoderAddChunkedFrame and an output processor both set, libjxl streams instead: it asks for rectangles of at most
2048 x 2048 pixels when it needs them and releases each, so the caller can produce each rectangle on demand (here: the
colour conversion of that rectangle only) and the full-size 16-bit frame never exists.

The encoder is configured as imagecodecs configures it (_jpegxl.pyx, 2026.10.10): one RGB frame, uint16, no
container, D7's colour encoding set as an enumeration, the distance from JxlEncoderDistanceFromQuality(level), the
effort, a frame header with duration 1. So the same pinned libjxl does the work (D44); only the input path differs.
"""
from __future__ import annotations

import ctypes as C
import glob
import threading
from collections.abc import Callable
from pathlib import Path

import imagecodecs
import numpy as np

_LIBS = Path(imagecodecs.__file__).resolve().parent.parent / "imagecodecs.libs"


def _load(stem: str) -> C.CDLL:
    paths = glob.glob(str(_LIBS / f"{stem}-*.so*"))
    if len(paths) != 1:
        raise OSError(f"expected one {stem} in {_LIBS}, found {paths}")
    return C.CDLL(paths[0])


_jxl = _load("libjxl")
_threads = _load("libjxl_threads")

# libjxl 0.12.0's enumerations (lib/include/jxl/*.h)
JXL_TYPE_UINT16 = 3
JXL_NATIVE_ENDIAN = 0
JXL_COLOR_SPACE_RGB, JXL_WHITE_POINT_D65, JXL_PRIMARIES_2100 = 0, 1, 9
JXL_TRANSFER_FUNCTION_709, JXL_RENDERING_INTENT_RELATIVE = 1, 1
JXL_ENC_SUCCESS = 0
JXL_ENC_FRAME_SETTING_EFFORT = 0
JXL_ENC_FRAME_SETTING_BUFFERING = 34


class PixelFormat(C.Structure):
    _fields_ = [("num_channels", C.c_uint32), ("data_type", C.c_int), ("endianness", C.c_int),
                ("align", C.c_size_t)]


class PreviewHeader(C.Structure):
    _fields_ = [("xsize", C.c_uint32), ("ysize", C.c_uint32)]


class AnimationHeader(C.Structure):
    _fields_ = [("tps_numerator", C.c_uint32), ("tps_denominator", C.c_uint32), ("num_loops", C.c_uint32),
                ("have_timecodes", C.c_int)]


class BasicInfo(C.Structure):
    _fields_ = [("have_container", C.c_int), ("xsize", C.c_uint32), ("ysize", C.c_uint32),
                ("bits_per_sample", C.c_uint32), ("exponent_bits_per_sample", C.c_uint32),
                ("intensity_target", C.c_float), ("min_nits", C.c_float), ("relative_to_max_display", C.c_int),
                ("linear_below", C.c_float), ("uses_original_profile", C.c_int), ("have_preview", C.c_int),
                ("have_animation", C.c_int), ("orientation", C.c_int), ("num_color_channels", C.c_uint32),
                ("num_extra_channels", C.c_uint32), ("alpha_bits", C.c_uint32), ("alpha_exponent_bits", C.c_uint32),
                ("alpha_premultiplied", C.c_int), ("preview", PreviewHeader), ("animation", AnimationHeader),
                ("intrinsic_xsize", C.c_uint32), ("intrinsic_ysize", C.c_uint32), ("padding", C.c_uint8 * 100)]


class ColorEncoding(C.Structure):
    _fields_ = [("color_space", C.c_int), ("white_point", C.c_int), ("white_point_xy", C.c_double * 2),
                ("primaries", C.c_int), ("primaries_red_xy", C.c_double * 2),
                ("primaries_green_xy", C.c_double * 2), ("primaries_blue_xy", C.c_double * 2),
                ("transfer_function", C.c_int), ("gamma", C.c_double), ("rendering_intent", C.c_int)]


class BlendInfo(C.Structure):
    _fields_ = [("blendmode", C.c_int), ("source", C.c_uint32), ("alpha", C.c_uint32), ("clamp", C.c_int)]


class LayerInfo(C.Structure):
    _fields_ = [("have_crop", C.c_int), ("crop_x0", C.c_int32), ("crop_y0", C.c_int32), ("xsize", C.c_uint32),
                ("ysize", C.c_uint32), ("blend_info", BlendInfo), ("save_as_reference", C.c_uint32)]


class FrameHeader(C.Structure):
    _fields_ = [("duration", C.c_uint32), ("timecode", C.c_uint32), ("name_length", C.c_uint32),
                ("is_last", C.c_int), ("layer_info", LayerInfo)]


GET_FORMAT = C.CFUNCTYPE(None, C.c_void_p, C.POINTER(PixelFormat))
GET_DATA = C.CFUNCTYPE(C.c_void_p, C.c_void_p, C.c_size_t, C.c_size_t, C.c_size_t, C.c_size_t,
                       C.POINTER(C.c_size_t))
GET_EC_FORMAT = C.CFUNCTYPE(None, C.c_void_p, C.c_size_t, C.POINTER(PixelFormat))
GET_EC_DATA = C.CFUNCTYPE(C.c_void_p, C.c_void_p, C.c_size_t, C.c_size_t, C.c_size_t, C.c_size_t, C.c_size_t,
                          C.POINTER(C.c_size_t))
RELEASE_INPUT = C.CFUNCTYPE(None, C.c_void_p, C.c_void_p)
GET_BUFFER = C.CFUNCTYPE(C.c_void_p, C.c_void_p, C.POINTER(C.c_size_t))
RELEASE_OUTPUT = C.CFUNCTYPE(None, C.c_void_p, C.c_size_t)
SEEK = C.CFUNCTYPE(None, C.c_void_p, C.c_uint64)
FINALIZED = C.CFUNCTYPE(None, C.c_void_p, C.c_uint64)


class ChunkedInput(C.Structure):
    _fields_ = [("opaque", C.c_void_p), ("get_color_channels_pixel_format", GET_FORMAT),
                ("get_color_channel_data_at", GET_DATA), ("get_extra_channel_pixel_format", GET_EC_FORMAT),
                ("get_extra_channel_data_at", GET_EC_DATA), ("release_buffer", RELEASE_INPUT)]


class OutputProcessor(C.Structure):
    _fields_ = [("opaque", C.c_void_p), ("get_buffer", GET_BUFFER), ("release_buffer", RELEASE_OUTPUT),
                ("seek", SEEK), ("set_finalized_position", FINALIZED)]


def _sig(lib: C.CDLL, name: str, restype, *argtypes) -> None:
    f = getattr(lib, name)
    f.restype, f.argtypes = restype, list(argtypes)


_P = C.c_void_p
_sig(_jxl, "JxlEncoderCreate", _P, _P)
_sig(_jxl, "JxlEncoderDestroy", None, _P)
_sig(_jxl, "JxlEncoderSetParallelRunner", C.c_int, _P, _P, _P)
_sig(_jxl, "JxlEncoderInitBasicInfo", None, C.POINTER(BasicInfo))
_sig(_jxl, "JxlEncoderSetBasicInfo", C.c_int, _P, C.POINTER(BasicInfo))
_sig(_jxl, "JxlEncoderSetColorEncoding", C.c_int, _P, C.POINTER(ColorEncoding))
_sig(_jxl, "JxlEncoderUseContainer", C.c_int, _P, C.c_int)
_sig(_jxl, "JxlEncoderFrameSettingsCreate", _P, _P, _P)
_sig(_jxl, "JxlEncoderSetFrameDistance", C.c_int, _P, C.c_float)
_sig(_jxl, "JxlEncoderDistanceFromQuality", C.c_float, C.c_float)
_sig(_jxl, "JxlEncoderFrameSettingsSetOption", C.c_int, _P, C.c_int, C.c_int64)
_sig(_jxl, "JxlEncoderInitFrameHeader", None, C.POINTER(FrameHeader))
_sig(_jxl, "JxlEncoderSetFrameHeader", C.c_int, _P, C.POINTER(FrameHeader))
_sig(_jxl, "JxlEncoderSetOutputProcessor", C.c_int, _P, OutputProcessor)
_sig(_jxl, "JxlEncoderAddChunkedFrame", C.c_int, _P, C.c_int, ChunkedInput)
_sig(_jxl, "JxlEncoderFlushInput", C.c_int, _P)
_sig(_jxl, "JxlEncoderGetError", C.c_int, _P)
_sig(_threads, "JxlThreadParallelRunnerCreate", _P, _P, C.c_size_t)
_sig(_threads, "JxlThreadParallelRunnerDestroy", None, _P)
_RUNNER = C.cast(_threads.JxlThreadParallelRunner, _P)

# A rectangle of the image: (ypos, xpos, ysize, xsize) in, uint16 array (ysize, xsize, 3) out, C-contiguous.
Region = Callable[[int, int, int, int], np.ndarray]


class JxlError(RuntimeError):
    pass


def _check(status: int, what: str, enc=None) -> None:
    if status != JXL_ENC_SUCCESS:
        detail = f", encoder error {_jxl.JxlEncoderGetError(enc)}" if enc else ""
        raise JxlError(f"{what} failed (status {status}{detail})")


def encode_chunked(height: int, width: int, region: Region, *, level: float = 95, effort: int = 7,
                   numthreads: int = 1, buffering: int | None = None, stats: dict | None = None) -> bytes:
    """A uint16 RGB image `height` x `width`, produced a rectangle at a time by `region`, as JPEG XL bytes in D7's
    colour encoding (Rec.2020 primaries, D65, BT.709 transfer, relative intent)."""
    held: dict[int, np.ndarray] = {}    # rectangles handed to libjxl and not yet released
    lock = threading.Lock()
    out = bytearray()
    pos = [0]
    outbuf: list = [None]
    errors: list[BaseException] = []
    counts = {"rects": 0, "pixels": 0, "max_held": 0}

    def get_format(_opaque, fmt):
        fmt.contents.num_channels, fmt.contents.data_type = 3, JXL_TYPE_UINT16
        fmt.contents.endianness, fmt.contents.align = JXL_NATIVE_ENDIAN, 0

    def get_data(_opaque, xpos, ypos, xsize, ysize, row_offset):
        try:
            a = np.ascontiguousarray(region(ypos, xpos, ysize, xsize), dtype=np.uint16)
            if a.shape != (ysize, xsize, 3):
                raise ValueError(f"region gave {a.shape}, not {(ysize, xsize, 3)}")
            row_offset[0] = a.strides[0]
            ptr = a.ctypes.data
            with lock:
                held[ptr] = a
                counts["rects"] += 1
                counts["pixels"] += xsize * ysize
                counts["max_held"] = max(counts["max_held"], len(held))
            return ptr
        except BaseException as e:   # an exception cannot cross into C: record it and hand back nothing
            errors.append(e)
            return None

    def get_ec_format(_opaque, _index, fmt):
        get_format(None, fmt)

    def get_ec_data(_opaque, _index, _x, _y, _xs, _ys, _row_offset):
        errors.append(RuntimeError("libjxl asked for an extra channel; the frame has none"))
        return None

    def release_input(_opaque, buf):
        with lock:
            held.pop(buf, None)

    def get_buffer(_opaque, size):
        n = max(int(size[0]), 1 << 16)
        outbuf[0] = (C.c_uint8 * n)()
        size[0] = n
        return C.addressof(outbuf[0])

    def release_output(_opaque, written):
        end = pos[0] + written
        if end > len(out):
            out.extend(bytes(end - len(out)))
        out[pos[0]:end] = C.string_at(C.addressof(outbuf[0]), written)
        pos[0] = end
        outbuf[0] = None

    def seek(_opaque, position):
        pos[0] = position

    def finalized(_opaque, _position):
        pass

    # Keep every callback object alive until the encoder is gone.
    callbacks = (GET_FORMAT(get_format), GET_DATA(get_data), GET_EC_FORMAT(get_ec_format), GET_EC_DATA(get_ec_data),
                 RELEASE_INPUT(release_input), GET_BUFFER(get_buffer), RELEASE_OUTPUT(release_output), SEEK(seek),
                 FINALIZED(finalized))
    enc = _jxl.JxlEncoderCreate(None)
    runner = _threads.JxlThreadParallelRunnerCreate(None, numthreads if numthreads > 1 else 0)
    try:
        _check(_jxl.JxlEncoderSetParallelRunner(enc, _RUNNER, runner), "JxlEncoderSetParallelRunner", enc)
        info = BasicInfo()
        _jxl.JxlEncoderInitBasicInfo(C.byref(info))
        info.xsize, info.ysize, info.num_color_channels, info.num_extra_channels = width, height, 3, 0
        info.bits_per_sample, info.exponent_bits_per_sample, info.uses_original_profile = 16, 0, 0
        _check(_jxl.JxlEncoderSetBasicInfo(enc, C.byref(info)), "JxlEncoderSetBasicInfo", enc)
        ce = ColorEncoding(color_space=JXL_COLOR_SPACE_RGB, white_point=JXL_WHITE_POINT_D65,
                           primaries=JXL_PRIMARIES_2100, transfer_function=JXL_TRANSFER_FUNCTION_709,
                           rendering_intent=JXL_RENDERING_INTENT_RELATIVE)
        _check(_jxl.JxlEncoderSetColorEncoding(enc, C.byref(ce)), "JxlEncoderSetColorEncoding", enc)
        _check(_jxl.JxlEncoderUseContainer(enc, 0), "JxlEncoderUseContainer", enc)
        settings = _jxl.JxlEncoderFrameSettingsCreate(enc, None)
        _check(_jxl.JxlEncoderSetFrameDistance(settings, _jxl.JxlEncoderDistanceFromQuality(level)),
               "JxlEncoderSetFrameDistance", enc)
        if effort != 7:
            _check(_jxl.JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_EFFORT, effort),
                   "effort", enc)
        if buffering is not None:
            _check(_jxl.JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_BUFFERING, buffering),
                   "buffering", enc)
        header = FrameHeader()
        _jxl.JxlEncoderInitFrameHeader(C.byref(header))
        header.duration = 1
        _check(_jxl.JxlEncoderSetFrameHeader(settings, C.byref(header)), "JxlEncoderSetFrameHeader", enc)
        _check(_jxl.JxlEncoderSetOutputProcessor(enc, OutputProcessor(None, *callbacks[5:])),
               "JxlEncoderSetOutputProcessor", enc)
        status = _jxl.JxlEncoderAddChunkedFrame(settings, 1, ChunkedInput(None, *callbacks[:5]))
        if errors:
            raise errors[0]
        _check(status, "JxlEncoderAddChunkedFrame", enc)
        _check(_jxl.JxlEncoderFlushInput(enc), "JxlEncoderFlushInput", enc)
        if errors:
            raise errors[0]
    finally:
        _jxl.JxlEncoderDestroy(enc)
        _threads.JxlThreadParallelRunnerDestroy(runner)
    if stats is not None:
        stats.update(counts)
    return bytes(out)
