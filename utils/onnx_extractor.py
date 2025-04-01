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
from pathlib import Path
from scipy import interpolate

layer_mapping = {
    "Conv2d": "Convolution",
    "ConvTranspose2d": "Deconvolution",
    "Linear": "Linear",
    "BatchNorm2d": "BatchNormalization",
    "BatchNorm1d": "BatchNormalization",
    "LayerNorm": "Normalization",
    "InstanceNorm2d": "InstanceNormalization",
    "Dropout": "Dropout",
    "ReLU": "Activation",
    "LeakyReLU": "Activation",
    "PReLU": "Activation",
    "Sigmoid": "Activation",
    "Tanh": "Activation",
    "MaxPool2d": "Pooling",
    "AvgPool2d": "Pooling",
    "AdaptiveMaxPool2d": "AdaptivePooling",
    "AdaptiveAvgPool2d": "AdaptivePooling",
    "Flatten": "Flatten",
    "Add": "Add",
    "Concat": "Concatenate",
    # ONNX types
    "Conv": "Convolution",
    "ConvTranspose": "Deconvolution",
    "Gemm": "Linear",
    "BatchNormalization": "BatchNormalization",
    "InstanceNormalization": "InstanceNormalization",
    "Relu": "Activation",
    "LeakyRelu": "Activation",
    "PRelu": "Activation",
    "Sigmoid": "Activation",
    "Tanh": "Activation",
    "MaxPool": "Pooling",
    "AveragePool": "Pooling",
    "GlobalMaxPool": "GlobalPooling",
    "GlobalAveragePool": "GlobalPooling",
    "Dropout": "Dropout",
    "Flatten": "Flatten",
    "Add": "Add",
    "Concat": "Concatenate"
}

class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert ndarray to list
        return super().default(obj)

def parse_onnx_attributes(node,onlytensorsize=True):
    # Extract attributes as parameters in a dict
    parameters = {}
    for attr in node.attribute:
        if attr.type == onnx.AttributeProto.FLOAT:
            parameters[attr.name] = attr.f
        elif attr.type == onnx.AttributeProto.INT:
            parameters[attr.name] = attr.i
        elif attr.type == onnx.AttributeProto.STRING:
            # attr.s is a byte string so decode it
            parameters[attr.name] = attr.s.decode('utf-8') if isinstance(attr.s, bytes) else attr.s
        elif attr.type == onnx.AttributeProto.TENSOR:
            # Convert tensor attribute to a list (for JSON serialization)
            output = numpy_helper.to_array(attr.t)
            if onlytensorsize:
                parameters[attr.name] = output.shape
            else:
                parameters[attr.name] = output.tolist()
        elif attr.type == onnx.AttributeProto.FLOATS:
            parameters[attr.name] = list(attr.floats)
        elif attr.type == onnx.AttributeProto.INTS:
            parameters[attr.name] = list(attr.ints)
        elif attr.type == onnx.AttributeProto.STRINGS:
            parameters[attr.name] = [
                s.decode('utf-8') if isinstance(s, bytes) else s for s in attr.strings
            ]
        else:
            parameters[attr.name] = None  # Fallback for unknown types
    return parameters

def extract_compute_graph_taxonomy_style(filename,savePath='./outdata',model_name='',parameters=False,writeFile=True):
    interpolate_space = np.linspace(0, 1, num=256)
    if model_name == '':
        model_name = os.path.splitext(os.path.basename(filename))[0]
    if writeFile and not os.path.exists(savePath):
        os.mkdir(savePath)
    
    # Gather the names of nodes already present (using the 'name' field)
    model = onnx.load(filename)
    outjson = {'name':model_name,'library':'onnx'}
    inferred_model = onnx.shape_inference.infer_shapes(model)
    
    #print(inferred_model.graph)
    shapes = {}
    for value_info in list(inferred_model.graph.input) + list(inferred_model.graph.output) + list(inferred_model.graph.value_info):
        dims = []
        for dim in value_info.type.tensor_type.shape.dim:
            # Use dim_value if available; otherwise, use dim_param (for symbolic dimensions)
            candidate = dim.dim_value if dim.HasField("dim_value") else dim.dim_param
            if type(candidate) is int:
                dims.append(candidate)
        shapes[value_info.name] = dims

    graph = model.graph
    existing_names = [ node.name for node in graph.node]
    outdata = {'model':model_name,'nodes':[],'library':'onnx'}
    parameters = []
    layers = []
    for node in graph.node:
        layer = {}
        layer['name'] = node.name or '<unnamed>'
        if node.op_type == "ConstantOfShape":
            continue
        else:
            layer['type'] = node.op_type
        layer['parameters'] = {}
        layer['attribute'] = {}
        node_info = {}
        node_info['op'] = node.op_type
        node_info['name'] = node.name or '<unnamed>'
        node_info['input'] = list(node.input) #[ {'name':inp, 'shape': shapes.get(inp,[]) } for inp in list(node.input)]
        node_info['target'] = list(node.output)
        layer['attributes'] = node_info['attributes'] = parse_onnx_attributes(node)
        layer['parameters'] = []
        node_info['parameters'] = {}
        for inp in list(node.input):
            parameter = {}
            parameter['name'] = inp
            parameter['shape'] = node_info['parameters'][inp] = shapes.get(inp,[])
            initializer =  next((init for init in graph.initializer if init.name == inp), None)
            if initializer:
                weight_data = numpy_helper.to_array(initializer).flatten()
                try: 
                    interpolator = interpolate.interp1d(np.linspace(0,1,weight_data.shape[0]), weight_data ,kind='linear')
                    weight_data = interpolator(interpolate_space)
                    #parameter['interpolated_vector'] =  weight_data.tolist()
                except:
                    #parameter['interpolated_vector'] = np.zeros((256))
                    print('WARNING!: something went wrong in interpolation',inp)
            if 'interpolated_vector' in parameter:
                layer['parameters'].append(parameter)
        layers.append(layer)
        outdata['nodes'].append(node_info)

        if parameters:
            layers.append({'name':node.name})
            print(f"Node Name: {node.name}, Type: {node.op_type}")
            for input_name in node.input:
                
                # Check if the input is an initializer
                initializer = next((init for init in graph.initializer if init.name == input_name), None)
                if initializer:
                    shape = [dim for dim in initializer.dims]
                    #print(initializer)
                    print(f"  Initializer - Name: {initializer.name}, Shape: {shape}")
                    weight_data = numpy_helper.to_array(initializer)
                    #np.frombuffer(initializer.raw_data, dtype=np.float32).reshape(initializer.dims)
                    filename = f'{savePath}{model_name}_{initializer.name}'
                    np.save( filename, weight_data)
                    layer_type = layer_mapping.get(node.op_type,node.op_type)
                    parameters.append({'filename':os.path.basename(filename+'.npy'),'layer_type':layer_type,'layer_name':node.name,'parameter_name':initializer.name,'shape':shape})
                else:
                    print(f"  Input - Name: {input_name} (Not an initializer)")
    if parameters:
        outdata['parameter_in_order_files'] = parameters
        outdata['layers'] = layers
    outjson['network'] = layers
    #outjson['graph'] = outdata

    if writeFile:
        handle = open(f'{savePath}/{model_name}.json','w')
        handle.write(json.dumps(outdata))
        handle.close()
    return outjson


class ONNXProgram:
    def extract_properties(self,filepath,savePath='./outdata/',model_name='',parameters=False):
        
        #files = glob.glob(filepath+'*.onnx',recursive=True)
        out = extract_compute_graph_taxonomy_style(filepath,savePath=savePath,parameters=parameters)
        with open("data/onnx_testing/resnet50_test.json" , "w") as f:
            json.dump(out , f , cls=NumpyJSONEncoder , indent=3 , )
    
    def inference_extraction(self,filename,model_type,model_name,savePath='./outdata'):
        # https://github.com/microsoft/onnxruntime/issues/1455
        
        #providers = ["CUDAExecutionProvider"]
        ort_session = ort.InferenceSession(filename)#, providers=providers)
        org_outputs = [x.name for x in ort_session.get_outputs()]
        model = onnx.load(filename)
        ignored_types = ['transpose','reshape','constant','cast','squeeze','unsqueeze','gather']
        for node in model.graph.node:
            #print(node.op_type)
            if node.op_type.lower() in ignored_types:
                continue
            for output in node.output:
                print(output)
                if output not in org_outputs:
                    model.graph.output.extend([onnx.ValueInfoProto(name=output)])
            
            #input()
        # excute onnx
        ort_session = ort.InferenceSession(model.SerializeToString())#, providers=providers)
        outputs = [x.name for x in ort_session.get_outputs()]
        #ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}

        input_data_dict = {}
        for input_info in ort_session.get_inputs():
            input_name = input_info.name
            shape = [dim if dim is not None else 1 for dim in input_info.shape]
            if 'float' in input_info.type:
                input_data_dict[input_name] = np.random.rand(*shape).astype(np.float32)
            elif 'int' in input_info.type:
                input_data_dict[input_name] = np.random.randint(0, 10, size=shape, dtype=np.int64)

        ort_outs = ort_session.run(outputs, input_data_dict)
        ort_outs = OrderedDict(zip(outputs, ort_outs))

    def compute_graph_extraction(self,onnx_file,outfile=None,savePath=''):
   
        # Load ONNX model
        model = onnx.load(onnx_file)
        # not including the raw data of the initializers
        model.graph.ClearField("initializer")    
        #model_json = MessageToJson(model, including_default_value_fields=False)
        model_json = MessageToJson(model)

        # Save as JSON
        #with open(f"{model_name}.json", "w") as f:
        #    json.dump(model_dict, f, indent=4)
        if savePath == '':
            savePath = '.'
        # with open(f"{savePath}/{outfile}", "w") as f:
        #     f.write(model_json)
        outfile = onnx_file.replace(".onnx" , "_parsed.json")
        with open(outfile , "w") as f:
            f.write(model_json)

if __name__ == '__main__':
    #extract_properties('adv_inception_v3_Opset16.onnx',model_type="cnn",model_name="inception")
    #fire.Fire(ONNXProgram)
    ONNXProgram().extract_properties("data/onnx_testing/light_resnet50.onnx")