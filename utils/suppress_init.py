import os
import logging
"""
Suppress warnings and logs from various libraries.
"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
logging.getLogger('faiss.loader').setLevel(logging.CRITICAL)
logging.getLogger('faiss').setLevel(logging.CRITICAL)
logging.getLogger('tensorflow').setLevel(logging.CRITICAL)
logging.getLogger('onnxruntime').setLevel(logging.CRITICAL)
logging.getLogger('onnxruntime.capi._pybind_state').setLevel(logging.CRITICAL)
logging.getLogger('onnxruntime.capi._pybind_state').propagate = False
logging.getLogger('onnxruntime.capi._pybind_state').disabled = True
logging.getLogger('onnxruntime.capi._pybind_state').handlers = []
logging.getLogger('onnx').setLevel(logging.CRITICAL)
logging.getLogger('onnx').propagate = False
logging.getLogger('onnx').disabled = True
logging.getLogger('onnx').handlers = []
logging.getLogger('onnxruntime').propagate = False
logging.getLogger('onnxruntime').disabled = True
logging.getLogger('onnxruntime').handlers = []
