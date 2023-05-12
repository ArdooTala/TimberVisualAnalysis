from json import JSONEncoder
import numpy as np
from flask import Flask, request, make_response
import knot_detector
import cv2

app = Flask(__name__)


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.reshape(-1).tolist()
        return JSONEncoder.default(self, obj)


@app.route("/")
def hello_world():
    params = {
        "crop": {
            "min_x": int(request.args.get('crop_min_x', 0)),
            "max_x": int(request.args.get('crop_max_x', 1300)),
            "min_y": int(request.args.get('crop_min_y', 0)),
            "max_y": int(request.args.get('crop_max_y', 1000)),
        },
        "blur": {
            "kernel_size": int(request.args.get('blur_kernel_size', 15))
        },
        "threshold_background": {
            "threshold": int(request.args.get('threshold_background', 127))
        },
        "threshold_knots": {
            "threshold": int(request.args.get('threshold_knots', 60))
        },
        "opening": {
            "kernel_size": int(request.args.get('opening_kernel_size', 7)),
            "iterations": int(request.args.get('opening_iterations', 5))
        }
    }
    file_name = request.args.get('file_name', None)
    if not file_name:
        return None
    return knot_detector.extract_contours(file_name, params)[1]


@app.route("/verbose")
def verbose_process():
    params = {
        "general": {
            "verbose": False,
            "step": 1
        },
        "crop": {
            "min_x": int(request.args.get('crop_min_x', 0)),
            "max_x": int(request.args.get('crop_max_x', 1300)),
            "min_y": int(request.args.get('crop_min_y', 0)),
            "max_y": int(request.args.get('crop_max_y', 1000)),
        },
        "blur": {
            "kernel_size": int(request.args.get('blur_kernel_size', 15))
        },
        "threshold_background": {
            "threshold": int(request.args.get('threshold_background', 127))
        },
        "threshold_knots": {
            "threshold": int(request.args.get('threshold_knots', 60))
        },
        "opening": {
            "kernel_size": int(request.args.get('opening_kernel_size', 7)),
            "iterations": int(request.args.get('opening_iterations', 5))
        }
    }

    file_name = request.args.get('file_name', None)
    if not file_name:
        return None
    img = knot_detector.extract_contours(file_name, params)[0]

    retval, buffer = cv2.imencode('.png', img)
    response = make_response(buffer.tobytes())
    response.headers['Content-Type'] = 'image/png'

    return response


@app.route("/grains")
def grain_process():
    params = {
        "crop": {
            "min_x": int(request.args.get('crop_min_x', 100)),
            "max_x": int(request.args.get('crop_max_x', 1200)),
            "min_y": int(request.args.get('crop_min_y', 400)),
            "max_y": int(request.args.get('crop_max_y', 700)),
        },
        "blur": {
            "kernel_size": int(request.args.get('blur_kernel_size', 15))
        },
        "threshold_background": {
            "threshold": int(request.args.get('threshold_background', 127))
        },
        "gradients": {
            "kernel_size": int(request.args.get('gradients_kernel_size', 31))
        },
        "gradients_blur": {
            "kernel_size": int(request.args.get('direction_blur_kernel_size', 31)),
        },
        "downsample": {
            "factor": int(request.args.get('downsample_factor', 2))
        }
    }

    file_name = request.args.get('file_name', None)
    if not file_name:
        return None
    return knot_detector.grain_direction(file_name, params)[1]


@app.route("/grains/verbose")
def grain_verbose_process():
    params = {
        "crop": {
            "min_x": int(request.args.get('crop_min_x', 100)),
            "max_x": int(request.args.get('crop_max_x', 1200)),
            "min_y": int(request.args.get('crop_min_y', 400)),
            "max_y": int(request.args.get('crop_max_y', 700)),
        },
        "blur": {
            "kernel_size": int(request.args.get('blur_kernel_size', 15))
        },
        "threshold_background": {
            "threshold": int(request.args.get('threshold_background', 127))
        },
        "gradients": {
            "kernel_size": int(request.args.get('gradients_kernel_size', 31))
        },
        "gradients_blur": {
            "kernel_size": int(request.args.get('direction_blur_kernel_size', 31)),
        },
        "downsample": {
            "factor": int(request.args.get('downsample_factor', 2))
        }
    }

    file_name = request.args.get('file_name', None)
    if not file_name:
        return None
    img = knot_detector.grain_direction(file_name, params)[0]

    retval, buffer = cv2.imencode('.png', img)
    response = make_response(buffer.tobytes())
    response.headers['Content-Type'] = 'image/png'

    return response
