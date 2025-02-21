import numpy as np
import onnx
import glob
import json
import os
from onnx import numpy_helper
import onnxruntime as ort
import fire
from collections import OrderedDict
from google.protobuf.json_format import MessageToJson


class ONNXProgram:
    def compute_graph_extraction(self,onnx_file,outfile,savePath=''):
   
        # Load ONNX model
        model = onnx.load(onnx_file)
        # not including the raw data of the initializers
        model.graph.ClearField("initializer")    
        model_json = MessageToJson(model, including_default_value_fields=False)

        # Save as JSON
        #with open(f"{model_name}.json", "w") as f:
        #    json.dump(model_dict, f, indent=4)
        if savePath == '':
            savePath = '.'
        with open(f"{savePath}/{outfile}", "w") as f:
            f.write(model_json)

if __name__ == '__main__':
    #extract_properties('adv_inception_v3_Opset16.onnx',model_type="cnn",model_name="inception")
    fire.Fire(ONNXProgram)
