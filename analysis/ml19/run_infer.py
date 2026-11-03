"""ML-19: coffeecv.infer with the dataset tools' pixel limit, so a 200 MP photo decodes (run with PYTHONPATH=repo)."""
import sys
import runpy
from PIL import Image
Image.MAX_IMAGE_PIXELS = 2**28
sys.argv = ["coffeecv.infer", *sys.argv[1:]]
runpy.run_module("coffeecv.infer", run_name="__main__")
