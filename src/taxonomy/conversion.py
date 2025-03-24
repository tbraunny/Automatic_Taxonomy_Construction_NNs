from rdflib import Graph, Namespace, URIRef, RDF, RDFS, OWL, Literal

# Load the FAIRnets ontology (assumed to be in Turtle format)
fairnet_graph = Graph()
fairnet_graph.parse("fairnet.ttl", format="turtle")

# Create a new graph for the ANNett-o version
annett_graph = Graph()
annett_graph.parse("annett-o-0.1.owl", format="xml")

# Define the namespaces (adjust these to match your ontologies)
ANN = Namespace("http://w3id.org/annett-o/")
# For FAIRnets, use the appropriate namespace; here we assume a placeholder:
#FAIR = Namespace("http://example.org/fairnet#")
FAIR = Namespace("https://w3id.org/nno/ontology#")
FOAF = Namespace("http://xmlns.com/foaf/0.1/")


# Bind common prefixes
#annett_graph.bind("annett-o", ANN)
#annett_graph.bind("owl", OWL)
#annett_graph.bind("rdfs", RDFS)
#annett_graph.bind("rdf", RDF)


loss_function_mapping = {
    FAIR.categoricalcrossentropy: ANN.CategoricalCrossEntropy,
    FAIR.binarycrossentropy: ANN.BinaryCrossEntropy,
    FAIR.mse: ANN.MSE,
    FAIR.meansquarederror: ANN.MSE
}

# --- Define a mapping dictionary ---
# Map FAIRnets URIs to ANNett‑O URIs. You will need to complete or adjust these mappings.
#mapping = {
#    FAIR.Organization: ANN.Organization,
#    FAIR.Person: ANN.Person,
    # Add more class mappings:
    # FAIR.SomeLayer: ANN.SomeLayer, etc.
    # Similarly, map properties if needed:
#    FAIR.hasName: ANN.hasName,
#    FAIR.belongsTo: ANN.belongsTo,
    # You can also map other URIs (e.g. from annotations, etc.)
#}

# -------------------------------------
# Example: fairnets_to_annetto_mapping.py
# -------------------------------------

mapping = {
    # -------------------------------------------------
    # FOAF Classes
    # -------------------------------------------------
    # "foaf:Person" is roughly "ann:People" (people or persons).
    # "foaf:Organization" doesn't have a direct counterpart in ANNett-O,
    # so we either leave it unmapped or guess the nearest concept.
    "http://xmlns.com/foaf/0.1/Person": "http://w3id.org/annett-o/People",
    "http://xmlns.com/foaf/0.1/Organization": None,  # or "http://w3id.org/annett-o/DataCharacterization"

    # -------------------------------------------------
    # NNO Classes => ANNett-O Classes
    # -------------------------------------------------
    


    # A. Core Mappings
    "https://w3id.org/nno/ontology#Layer": "http://w3id.org/annett-o/Layer",
    "https://w3id.org/nno/ontology#NeuralNetwork": "http://w3id.org/annett-o/ANNConfiguration",
    "https://w3id.org/nno/ontology#Model": "http://w3id.org/annett-o/Network",
    "https://w3id.org/nno/ontology#hasModel": "http://w3id.org/annett-o/hasNetwork",
    "https://w3id.org/nno/ontology#hasLayer": "http://w3id.org/annett-o/hasLayer",
    "https://w3id.org/nno/ontology#Optimizer": "http://w3id.org/annett-o/TrainingOptimizer",
    "https://w3id.org/nno/ontology#LossFunction": "http://w3id.org/annett-o/LossFunction",
    
    # B. Activation Layers
    "https://w3id.org/nno/ontology#Activation": "http://w3id.org/annett-o/ActivationLayer",
    "https://w3id.org/nno/ontology#Dropout": "http://w3id.org/annett-o/DropoutLayer",
    "https://w3id.org/nno/ontology#BatchNormalization": "http://w3id.org/annett-o/BatchNormLayer",
    "https://w3id.org/nno/ontology#Lambda": "http://w3id.org/annett-o/ModificationLayer",  # or just "Layer"
    "https://w3id.org/nno/ontology#Masking": "http://w3id.org/annett-o/ModificationLayer",
    "https://w3id.org/nno/ontology#SpatialDropout1D": "http://w3id.org/annett-o/DropoutLayer",
    "https://w3id.org/nno/ontology#SpatialDropout2D": "http://w3id.org/annett-o/DropoutLayer",
    "https://w3id.org/nno/ontology#SpatialDropout3D": "http://w3id.org/annett-o/DropoutLayer",
    
    # C. Convolutional / Deconvolutional Layers
    "https://w3id.org/nno/ontology#ConvolutionalLayer": "http://w3id.org/annett-o/ConvolutionLayer",
    "https://w3id.org/nno/ontology#Conv1D": "http://w3id.org/annett-o/ConvolutionLayer",
    "https://w3id.org/nno/ontology#Conv2D": "http://w3id.org/annett-o/ConvolutionLayer",
    "https://w3id.org/nno/ontology#Conv3D": "http://w3id.org/annett-o/ConvolutionLayer",
    "https://w3id.org/nno/ontology#Conv2DTranspose": "http://w3id.org/annett-o/DeconvolutionLayer",
    "https://w3id.org/nno/ontology#Conv3DTranspose": "http://w3id.org/annett-o/DeconvolutionLayer",
    "https://w3id.org/nno/ontology#DepthwiseConv2D": "http://w3id.org/annett-o/SeparableConvolutionLayer",
    "https://w3id.org/nno/ontology#SeparableConv1D": "http://w3id.org/annett-o/SeparableConvolutionLayer",
    "https://w3id.org/nno/ontology#SeparableConv2D": "http://w3id.org/annett-o/SeparableConvolutionLayer",
    "https://w3id.org/nno/ontology#LocallyConnected1D": "http://w3id.org/annett-o/ConvolutionLayer",
    "https://w3id.org/nno/ontology#LocallyConnected2D": "http://w3id.org/annett-o/ConvolutionLayer",

    # D. Pooling Layers
    "https://w3id.org/nno/ontology#PoolingLayer": "http://w3id.org/annett-o/PoolingLayer",
    "https://w3id.org/nno/ontology#AveragePooling1D": "http://w3id.org/annett-o/PoolingLayer",
    "https://w3id.org/nno/ontology#AveragePooling2D": "http://w3id.org/annett-o/PoolingLayer",
    "https://w3id.org/nno/ontology#AveragePooling3D": "http://w3id.org/annett-o/PoolingLayer",
    "https://w3id.org/nno/ontology#MaxPooling1D": "http://w3id.org/annett-o/PoolingLayer",
    "https://w3id.org/nno/ontology#MaxPooling2D": "http://w3id.org/annett-o/PoolingLayer",
    "https://w3id.org/nno/ontology#MaxPooling3D": "http://w3id.org/annett-o/PoolingLayer",
    "https://w3id.org/nno/ontology#GlobalAveragePooling1D": "http://w3id.org/annett-o/PoolingLayer",
    "https://w3id.org/nno/ontology#GlobalAveragePooling2D": "http://w3id.org/annett-o/PoolingLayer",
    "https://w3id.org/nno/ontology#GlobalAveragePooling3D": "http://w3id.org/annett-o/PoolingLayer",
    "https://w3id.org/nno/ontology#GlobalMaxPooling1D": "http://w3id.org/annett-o/PoolingLayer",
    "https://w3id.org/nno/ontology#GlobalMaxPooling2D": "http://w3id.org/annett-o/PoolingLayer",
    "https://w3id.org/nno/ontology#GlobalMaxPooling3D": "http://w3id.org/annett-o/PoolingLayer",

    # E. Upsampling / Cropping
    "https://w3id.org/nno/ontology#UpSampling1D": "http://w3id.org/annett-o/UpscaleLayer",
    "https://w3id.org/nno/ontology#UpSampling2D": "http://w3id.org/annett-o/UpscaleLayer",
    "https://w3id.org/nno/ontology#UpSampling3D": "http://w3id.org/annett-o/UpscaleLayer",
    # "Cropping" layers have no direct match in ANNett-O; treat as "Modification":
    "https://w3id.org/nno/ontology#Cropping1D": "http://w3id.org/annett-o/ModificationLayer",
    "https://w3id.org/nno/ontology#Cropping2D": "http://w3id.org/annett-o/ModificationLayer",
    "https://w3id.org/nno/ontology#Cropping3D": "http://w3id.org/annett-o/ModificationLayer",
    "https://w3id.org/nno/ontology#ZeroPadding1D": "http://w3id.org/annett-o/ModificationLayer",
    "https://w3id.org/nno/ontology#ZeroPadding2D": "http://w3id.org/annett-o/ModificationLayer",
    "https://w3id.org/nno/ontology#ZeroPadding3D": "http://w3id.org/annett-o/ModificationLayer",

    # F. Dense / Fully Connected
    "https://w3id.org/nno/ontology#Dense": "http://w3id.org/annett-o/FullyConnectedLayer",

    # G. Embedding / Flatten / Permute
    "https://w3id.org/nno/ontology#Embedding": None,  # No direct embedding concept in ANNett-O
    "https://w3id.org/nno/ontology#Flatten": "http://w3id.org/annett-o/FlattenLayer",
    "https://w3id.org/nno/ontology#Permute": "http://w3id.org/annett-o/ModificationLayer",
    "https://w3id.org/nno/ontology#Reshape": "http://w3id.org/annett-o/ModificationLayer",
    "https://w3id.org/nno/ontology#RepeatVector": "http://w3id.org/annett-o/ModificationLayer",

    # H. Recurrent Layers
    "https://w3id.org/nno/ontology#RecurrentLayer": "http://w3id.org/annett-o/RNNLayer",
    "https://w3id.org/nno/ontology#GRU": "http://w3id.org/annett-o/GRULayer",
    "https://w3id.org/nno/ontology#GRUCell": "http://w3id.org/annett-o/GRULayer",
    "https://w3id.org/nno/ontology#CuDNNGRU": "http://w3id.org/annett-o/GRULayer",
    "https://w3id.org/nno/ontology#LSTM": "http://w3id.org/annett-o/LSTMLayer",
    "https://w3id.org/nno/ontology#LSTMCell": "http://w3id.org/annett-o/LSTMLayer",
    "https://w3id.org/nno/ontology#CuDNNLSTM": "http://w3id.org/annett-o/LSTMLayer",
    "https://w3id.org/nno/ontology#SimpleRNN": "http://w3id.org/annett-o/RNNLayer",
    "https://w3id.org/nno/ontology#SimpleRNNCell": "http://w3id.org/annett-o/RNNLayer",
    "https://w3id.org/nno/ontology#ConvLSTM2D": "http://w3id.org/annett-o/RNNLayer",
    "https://w3id.org/nno/ontology#ConvLSTM2DCell": "http://w3id.org/annett-o/RNNLayer",

    # I. Input / Output / Base
    "https://w3id.org/nno/ontology#Input": "http://w3id.org/annett-o/InputLayer",
    "https://w3id.org/nno/ontology#InputLayer": "http://w3id.org/annett-o/InputLayer",
    "https://w3id.org/nno/ontology#Output": None,  # no direct "Output" class in nno (?), or "OutputLayer"?
    "https://w3id.org/nno/ontology#BaseModel": "http://w3id.org/annett-o/Network",

    # J. Losses
    "https://w3id.org/nno/ontology#ClassificationLoss": "http://w3id.org/annett-o/LossFunction",
    "https://w3id.org/nno/ontology#RegressiveLoss": "http://w3id.org/annett-o/LossFunction",

    # K. Additional constructs
    "https://w3id.org/nno/ontology#CoreLayer": "http://w3id.org/annett-o/HiddenLayer",
    "https://w3id.org/nno/ontology#ActivityRegularization": "http://w3id.org/annett-o/ModificationLayer",
    "https://w3id.org/nno/ontology#Locally-connectedLayer": "http://w3id.org/annett-o/ConvolutionLayer",
    "https://w3id.org/nno/ontology#EmbeddingLayer": None,      # No direct match
    "https://w3id.org/nno/ontology#NormalizationLayer": None,  # Could approximate as "BatchNormLayer" or "ModificationLayer"
    "https://w3id.org/nno/ontology#PoolingLayer": "http://w3id.org/annett-o/PoolingLayer",
}



def map_uri(uri):
    """Return the ANNett-O equivalent if defined; else return the same URI."""
    #uri = str(uri)
    return URIRef(mapping[str(uri)]) if (str(uri) in mapping and mapping[str(uri)]) else uri

def parse_rdf_list(list_node, graph):
    """Given a BNode or URIRef that starts an RDF list,
    return a Python list of the items in that sequence."""
    items = []
    while list_node and list_node != RDF.nil:
        first_item = graph.value(list_node, RDF.first)
        items.append(first_item)
        list_node = graph.value(list_node, RDF.rest)
    return items

# --- Transformation Process ---
# This example simply rewrites the subjects, predicates, and object URIs (if they appear in the mapping)

FAIRDATA = 'https://w3id.org/nno/data#'

# get all neural networks


exclusionList = []
#network
for network in fairnet_graph.subjects(RDF.type, FAIR.NeuralNetwork):
    print('network',network)
    adding = True
    #for s,p,o in fairnet_graph.triples((network,None,None)):
    #    if 'custom' in str(o):
    #        adding=False
    for s,p,o in fairnet_graph.triples((network,None,None)):
        for s,p,o2 in fairnet_graph.triples((o,None,None)):
            if 'custom' in str(o2).lower():
                adding=False
            for s3,p3,o3 in fairnet_graph.triples((o2,None,None)):
                if 'custom' in str(o3).lower():
                    adding=False
                    fairnet_graph.remove((s3,p3,o3))
    #for s,p,model in fairnet_graph.triples((network,FAIR.hasModel,None)):
    #    for s,p,layer in fairnet_graph.triples((model,None,None)):
    #        if 'custom' in str(layer).lower(): 
    #            adding = False
    #            print('not adding',model)
    #    print('model',model)
    if not adding:
        for s,p,o in fairnet_graph.triples((network,None,None)):
            #exclusionList.append((s,p,o))
            fairnet_graph.remove((s,p,o))
            for s2,p2,o2 in fairnet_graph.triples((o,None,None)):
                fairnet_graph.remove((s2,p2,o2))
print(exclusionList)
print('exclusionList')
#input()

for s, p, o in fairnet_graph:
    if (s,p,o) in exclusionList:
        print('found one')
        continue


    new_s = map_uri(s)
    new_p = map_uri(p)
    new_o = map_uri(o) if isinstance(o, URIRef) else o

    if str(s) == "https://w3id.org/nno/ontology":
        continue
    if FAIR.hasLossFunction == p:
        continue
    if FAIR.hasLayerSequence == p:
        continue

    new_s = URIRef(str(new_s).replace(str(FAIRDATA), str(ANN))) if str(new_s).startswith(str(FAIRDATA)) else new_s
    new_p = URIRef(str(new_p).replace(str(FAIRDATA), str(ANN))) if str(new_p).startswith(str(FAIRDATA)) else new_p
    new_o = URIRef(str(new_o).replace(str(FAIRDATA), str(ANN))) if isinstance(new_o, URIRef) and str(new_o).startswith(str(FAIRDATA)) else new_o

    #print(FAIR)
    #input()

    new_s = URIRef(str(new_s).replace(str(FAIR), str(ANN))) if str(new_s).startswith(str(FAIR)) else new_s
    new_p = URIRef(str(new_p).replace(str(FAIR), str(ANN))) if str(new_p).startswith(str(FAIR)) else new_p
    new_o = URIRef(str(new_o).replace(str(FAIR), str(ANN))) if isinstance(new_o, URIRef) and str(new_o).startswith(str(FAIR)) else new_o
    annett_graph.add((new_s, new_p, new_o))




#for model in annett_graph.subjects(RDF.type, ANN.Network):

    #input()

for model in fairnet_graph.subjects(RDF.type, FAIR.Model):
    print(model)
    #input()

    omodel = model
    loss = fairnet_graph.value(omodel, FAIR.hasLossFunction)
    model = URIRef(str(model).replace(str(FAIRDATA), str(ANN)))

    if loss != None:
        loss = URIRef(str(loss).replace(str(FAIR), str(ANN)))
        
        new_objective = URIRef(str(model)+'_objective')  # This creates the new URIRef for "foo"
        new_loss = URIRef(str(model)+'_loss')

        # Assert that "foo" is a subclass of ObjectiveFunction
        annett_graph.add((new_loss,RDF.type,loss_function_mapping.get(loss,loss)))
        annett_graph.add((new_objective, RDF.type, ANN.ObjectiveFunction))

        # Add label and comment (optional but recommended)
        annett_graph.add((new_objective, ANN.label, new_objective))  # Label it as "foo"
        annett_graph.add((new_objective, ANN.comment, new_objective))

        annett_graph.add((new_objective,ANN.hasLoss, new_loss))
        #ANN
        #insert = (model, ANN.hasObjective,
        print(loss)

        annett_graph.add((model,ANN.hasObjective,new_objective))
        #input()

    ordering = []

    for s,p,layer in fairnet_graph.triples((omodel, FAIR.hasLayer, None)):
        print(layer)
        sequence_node = fairnet_graph.value(layer, FAIR.hasLayerSequence)
        parameter = fairnet_graph.value(layer,FAIR.hasLayerParameters)
        print(parameter)
        print(layer)
        layer = URIRef(str(layer).replace(str(FAIRDATA), str(ANN)))
        parameter = URIRef(str(parameter).replace(str(FAIRDATA), str(ANN)))
        
        print(layer,str(FAIR),str(ANN))
        #input()
        if sequence_node != None:
            ordering.append({'name':layer,'value':sequence_node, 'parameter': parameter})
        

        print(model)
        print(sequence_node)

    sorted_list = sorted(ordering, key= lambda x: x['value'])
    for index,item in enumerate(sorted_list):
        current = item['name']
        if index + 1 < len(sorted_list):
            annett_graph.add((sorted_list[index]['name'], ANN.nextLayer, sorted_list[index + 1]['name']))
        if index - 1 > 0:
            annett_graph.add((sorted_list[index]['name'], ANN.previousLayer, sorted_list[index - 1]['name']))


# Optionally, add an ontology header for ANNett-o
ontology_uri = ANN[""]
annett_graph.add((ontology_uri, RDF.type, OWL.Ontology))
annett_graph.add((ontology_uri, RDFS.comment, Literal("This ontology was converted from FAIRnets to ANNett‑O using a custom mapping.")))

# --- Serialize the new graph ---
annett_graph.serialize(destination="annett-o.owl", format="xml")
print("Conversion complete! Output saved as 'annett-o.owl'.")

