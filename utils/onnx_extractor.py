import numpy as np
import onnx
import glob
import json
import os
from onnx import numpy_helper
import onnxruntime as ort
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
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

def extract_compute_graph_taxonomy_style(filename,savePath='./outdata',model_name='',parameters=True,writeFile=True):
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

def fetch_input_layer_params(input_layer , model):
    """
    Calculate the number of parameters for different types of layers.

    :param input_layer: name of input layer
    :param node: layer node within the JSON
    :param model: ONNX model
    :return Number of parameters within the specific layer
    """
    num_params = 0
    # want to only calc for the specific input layer
    for shape in model.graph.input:
        access_shape = shape.type.tensor_type.shape.dim
        input_name: str = shape.name

        if input_name.endswith("__SHAPE"):
            num_params = access_shape[0].dim_value
        else: # conv layer?
            num_params = access_shape[1].dim_value

    return num_params

def calculate_params(parameters , layer , model):
    """
    Calculate the number of parameters for different types of layers.

    :param parameters: List of parameters for the given layer
    :param layer: Node of the layer being calculated
    :param model: ONNX model

    :return Number of parameters for the specified layer
    """



    if layer.op_type == "Conv":
        pass

    # For Conv layers, the parameters are the weights and possibly the biases
    #if layer.op_type == "Conv":

        #input_list: list = layer.name

        ########################## SELF DONE MEAT
        # print("INPUT LAYER NAME: " , layer)
        # print(type(layer))


        # for input in model.graph.input:
        #     print("INPUT NAME " , input.name)
        #     print("LAYER NAME " , layer.name)
        #     if layer.name in input.name:
        #         print("marker2")
        #         for dim_value in input.type.tensor_type.shape.dim:
        #             print("Dim value test: " , dim_value.dim_value)
        #########################################


        # Conv layer typically has weights (filters) and optional biases
        # weights_name_with_shape = layer.input[1]  # The weight shape tensor
        # biases_name = layer.input[2] if len(layer.input) > 2 else None  # Optional bias input

        # # Debugging: Print the weights and biases names
        # print(f"Conv layer: weights_name_with_shape = {weights_name_with_shape}, biases_name = {biases_name}")

        # for input in model.graph.input:
        #     if weights_name_with_shape in input.name:
        #         print(f"Found weight in input: {input.name}")
        #         print(f"Dims: {input.type.tensor_type.shape.dim}")

        #         for dim_value in input.type.tensor_type.shape.dim:
        #             print("dim_value test: " , dim_value.dim_value)
        #         print("Num params: " , num_params)

        # # Find the weight shape initializer and calculate its size
        # for initializer in model.graph.initializer:
        #     if weights_name_with_shape in initializer.name:
        #         print(f"Found weight shape initializer: {initializer.name}")  # Debugging
        #         print(f"Dims: " , initializer.dims[0])
        #         weight_shape = [int(dim.dim_value) for dim in initializer.dims]
        #         print(f"Weight shape: {weight_shape}")  # Debugging

        #         # Calculate the number of parameters using the weight shape
        #         # Conv layer has (out_channels * in_channels * kernel_height * kernel_width) parameters
        #         if len(weight_shape) == 4:  # Conv layer with 4D weights (out_channels, in_channels, kernel_height, kernel_width)
        #             num_params += weight_shape[0] * weight_shape[1] * weight_shape[2] * weight_shape[3]
        #         else:
        #             print(f"Unexpected weight shape: {weight_shape}")

        # Find and add the bias parameters if they exist
        # if biases_name:
        #     for initializer in model.graph.initializer:
        #         if initializer.name == biases_name:
        #             print(f"Found bias initializer: {initializer.name}")  # Debugging
        #             num_params += len(initializer.float_data)  # Biases are 1D, so the length is the number of output channels
    
    # For Dense layers (Fully Connected), the parameters are the weights and biases
    # elif layer.op_type == "Gemm":  # Gemm is typically used for Fully Connected layers in ONNX
    #     weights_name = layer.input[1]
    #     biases_name = layer.input[2] if len(layer.input) > 2 else None  # Optional bias input

    #     # Find the weight initializer and calculate its size (out_features * in_features)
    #     for initializer in model.graph.initializer:
    #         if initializer.name == weights_name:
    #             weight_size = 1
    #             for dim in initializer.dims:
    #                 weight_size *= dim
    #             num_params += weight_size

    #     # Find and add the bias parameters if they exist
    #     if biases_name:
    #         for initializer in model.graph.initializer:
    #             if initializer.name == biases_name:
    #                 num_params += len(initializer.float_data)  # Biases are 1D, so the length is the number of output features

    # # Add logic for other layer types here (BatchNorm, RNN, etc.)
    # # Example for BatchNorm:
    # elif layer.op_type == "BatchNormalization":
    #     # BatchNorm has gamma, beta, mean, and variance parameters
    #     for initializer in model.graph.initializer:
    #         if initializer.name in layer.input:  # Gamma and Beta are usually inputs
    #             num_params += len(initializer.float_data)

    #return num_params

def parse_conv_layer(node , model , input_cache):
    """
    Handle the calculation for convolutional layers
    """
    in_channels , out_channels = None

    for input_name in node.input: # take care of input layers
        if input_name not in input_cache:
            input_info = resolve_input_info(model , input_name)

            if input_info is not None:
                dims = input_info.type.tensor_shape.dim

                if len(dims) == 1 or input.endswith("__SHAPE"):
                    out_channels = dims[0].dim_value
                    input_cache[input_name] = out_channels
                else:
                    in_channels = dims[1].dim_value
        else:
            if input_name.endswith("__SHAPE"):
                out_channels = input_cache[input_name]
            else:
                in_channels = input_cache[input_name]

        # fallback plan in case channels aren't provided (prevent 0 param conv layers)
        in_channels = in_channels or 1
        out_channels = out_channels or 1

        kernel_shape = [1 , 1] # also a fallback
        for attr in node.attribute:
            if attr.name == "kernel_shape":
                kernel_shape = attr.ints # tuple

        return (kernel_shape[0] * kernel_shape[1] * in_channels + 1) * out_channels # conv param formula

def gemm_layer_parsing(node , model , input_cache):
    pass # WRITE STUFF HERE

def parse_bn_layer(node , model , input_cache):
    pass

def resolve_input_info(model , input_name):
    """
    Fetch input section of ONNX parsed JSON to obtain layer info
    """

    input_info = next((i for i in model.graph.input if i.name == input_name) , None)

    if input_info:
        return input_info
    else:
        alt_name = input_name + "__SHAPE"
        input_info = next((i for i in model.graph.input if i.name == alt_name) , None)

        return input_info

def extract_params_by_layer(model): 
    """
    Fetch & calculate the number of parameters per layer for a given model
    """

    layer_params: dict = {}
    input_cache: dict = {} # track calculated input layers

    for node in model.graph.node:
        if not node.name:
            continue
        
        handler = LAYER_PARSERS.get(node.op_type)
        if handler:
            try:
                param_count = handler(node , model , input_cache)
                layer_params[node.name] = param_count
            except Exception as e:
                print(f"Error processing {node.op_type} node {node.name}: {e}")
        else:
            print(f"Layer type {node.op_type} has no learnable parameters (hopefully)")
            layer_params = 0

    return layer_params

LAYER_PARSERS = {
    "Conv": parse_conv_layer,
    # "Gemm": parse_gemm_layer,
    # "BatchNormalization": parse_bn_layer,
}

def count_params(tensor):
    arr = onnx.numpy_helper.to_array(tensor)
    return arr.size  # Total number of parameters in this tensor

def count_onnx_params_by_layer(onnx_model):
    """
    Another attempt at a less manual approach, UNSUCCESSFUL
    """
    initializer_map = {init.name: init for init in onnx_model.graph.initializer}
    layer_param_counts = []

    for node in onnx_model.graph.node:
        param_count = 0
        param_names = []

        # Count parameters from initializers (weights/biases)
        for input_name in node.input:
            initializer = initializer_map.get(input_name)
            if initializer:
                dims = initializer.dims
                count = np.prod(dims) if dims else 0
                param_count += count
                param_names.append(input_name)

        if param_count > 0:
            layer_info = {
                "layer_name": node.name or node.output[0],
                "op_type": node.op_type,
                "params": int(param_count),
                "param_tensors": param_names
            }
            layer_param_counts.append(layer_info)

    return layer_param_counts


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

    def compute_graph_extraction(self,onnx_file,outfile=None,savePath='') -> dict:
        model = onnx.load(onnx_file)

        params = count_onnx_params_by_layer(model)

        # Print summary
        for p in params:
            print(f"{p['layer_name']} ({p['op_type']}): {p['params']} parameters")

        # less manual way of calculating parameters per layer
        inferred_model = SymbolicShapeInference.infer_shapes(model)

        # Map initializers by name for quick lookup
        initializer_map = {init.name: init for init in inferred_model.graph.initializer}

        layer_param_counts = {}

        for node in inferred_model.graph.node:
            param_count = 0
            for input_name in node.input:
                if input_name in initializer_map:
                    param_count += count_params(initializer_map[input_name])
            
            if param_count > 0:
                layer_param_counts[node.name or node.output[0]] = {
                    "op_type": node.op_type,
                    "params": param_count
        }


        for layer, info in layer_param_counts.items():
            print(f"{layer}: {info['op_type']} → {info['params']} learnable parameters")


        # Calculate the parameters by layer
        layer_params = extract_params_by_layer(model)

        # Convert the modified model to JSON format
        model_json = MessageToJson(model)

        #model.graph.ClearField("initializer") # OEM json output

        model_json = MessageToJson(model)
        model_json_dict = json.loads(model_json)

        # modify onnx dict to include num params by corresponding layer
        for node in model_json_dict['graph']['node']:
            layer_name = node.get('name', '')
            if layer_name in layer_params:
                node['num_param'] = layer_params[layer_name]

        return model_json_dict

        #return json.dumps(model_json_dict , indent=4)

        # Save as JSON
        # if savePath == '':
        #     savePath = '.'
        # outfile = onnx_file.replace(".onnx" , "_parsed.json")
        # with open(outfile , "w") as f:
        #     json.dump(model_json_dict , f , indent=2)

if __name__ == '__main__':
    #extract_properties('adv_inception_v3_Opset16.onnx',model_type="cnn",model_name="inception")
    #fire.Fire(ONNXProgram)
    ONNXProgram().extract_properties("data/onnx_testing/light_resnet50.onnx" , parameters=True)
    onnx_graph = ONNXProgram().compute_graph_extraction("data/onnx_testing/light_resnet50.onnx")
    
    with open("data/onnx_testing/light_resnet50_parsed.onnx" , "w") as f:
        json.dump(onnx_graph , f , indent=2)

    print("ONNX parsed!")