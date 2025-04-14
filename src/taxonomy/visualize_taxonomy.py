import networkx as nx
import matplotlib.pyplot as plt
from io import StringIO

graphml_data = """<graphml xmlns="http://graphml.graphdrawing.org/xmlns" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
  <key id="d3" for="node" attr.name="annConfigs" attr.type="string" />
  <key id="d2" for="node" attr.name="splitProperties" attr.type="string" />
  <key id="d1" for="node" attr.name="criteria" attr.type="string" />
  <key id="d0" for="node" attr.name="name" attr.type="string" />
  <graph edgedefault="directed">
    <node id="140183392034704">
      <data key="d0">Top of Taxonomy</data>
      <data key="d1">null</data>
      <data key="d2">{}</data>
      <data key="d3">["AAE", "GAN", "simple_classification"]</data>
    </node>
    <node id="140183389301056">
      <data key="d0">0</data>
      <data key="d1">{"Searchs": [{"HasType": "hasLoss", "Type": "", "Name": "", "Op": "", "Value": "", "HashOn": "type"}, {"HasType": "hasTaskType", "Type": "", "Name": "", "Op": "", "Value": "", "HashOn": "type"}]}</data>
      <data key="d2">{"http://w3id.org/annett-o/AAE": [{"type": "MSE", "name": "AAE_AE_MSE", "found": true, "hash": "MSE", "annconfig": "http://w3id.org/annett-o/AAE"}, {"type": "Reconstruction", "name": "AAE_AE_Type", "found": true, "hash": "Reconstruction", "annconfig": "http://w3id.org/annett-o/AAE"}, {"type": "Clustering", "name": "AAE_Encoder_Label_Type", "found": true, "hash": "Clustering", "annconfig": "http://w3id.org/annett-o/AAE"}, {"type": "Generation", "name": "AAE_Encoder_Style_Type", "found": true, "hash": "Generation", "annconfig": "http://w3id.org/annett-o/AAE"}, {"type": "BinaryCrossEntropy", "name": "AAE_Label_Discriminator_Loss", "found": true, "hash": "BinaryCrossEntropy", "annconfig": "http://w3id.org/annett-o/AAE"}, {"type": "Discrimination", "name": "AAE_Label_Discriminator_Type", "found": true, "hash": "Discrimination", "annconfig": "http://w3id.org/annett-o/AAE"}, {"type": "BinaryCrossEntropy", "name": "AAE_Label_Discriminator_Loss", "found": true, "hash": "BinaryCrossEntropy", "annconfig": "http://w3id.org/annett-o/AAE"}, {"type": "Adversarial", "name": "AAE_Label_GAN_Type", "found": true, "hash": "Adversarial", "annconfig": "http://w3id.org/annett-o/AAE"}, {"type": "BinaryCrossEntropy", "name": "AAE_Style_Discriminator_Loss", "found": true, "hash": "BinaryCrossEntropy", "annconfig": "http://w3id.org/annett-o/AAE"}, {"type": "Discrimination", "name": "AAE_Style_Discriminator_Type", "found": true, "hash": "Discrimination", "annconfig": "http://w3id.org/annett-o/AAE"}, {"type": "BinaryCrossEntropy", "name": "AAE_Style_Discriminator_Loss", "found": true, "hash": "BinaryCrossEntropy", "annconfig": "http://w3id.org/annett-o/AAE"}, {"type": "Adversarial", "name": "AAE_Style_GAN_Type", "found": true, "hash": "Adversarial", "annconfig": "http://w3id.org/annett-o/AAE"}]}</data>
      <data key="d3">["http://w3id.org/annett-o/AAE"]</data>
    </node>
    <node id="140183389511888">
      <data key="d0">0</data>
      <data key="d1">{"Searchs": [{"HasType": "hasLoss", "Type": "", "Name": "", "Op": "", "Value": "", "HashOn": "type"}, {"HasType": "hasTaskType", "Type": "", "Name": "", "Op": "", "Value": "", "HashOn": "type"}]}</data>
      <data key="d2">{"http://w3id.org/annett-o/GAN": [{"type": "BinaryCrossEntropy", "name": "GAN_Discriminator_BinaryCrossEntropy", "found": true, "hash": "BinaryCrossEntropy", "annconfig": "http://w3id.org/annett-o/GAN"}, {"type": "Discrimination", "name": "GAN_Discriminator_Type", "found": true, "hash": "Discrimination", "annconfig": "http://w3id.org/annett-o/GAN"}, {"type": "Generation", "name": "GAN_Generator_Type", "found": true, "hash": "Generation", "annconfig": "http://w3id.org/annett-o/GAN"}, {"type": "BinaryCrossEntropy", "name": "GAN_Discriminator_BinaryCrossEntropy", "found": true, "hash": "BinaryCrossEntropy", "annconfig": "http://w3id.org/annett-o/GAN"}, {"type": "Adversarial", "name": "GAN_net_type", "found": true, "hash": "Adversarial", "annconfig": "http://w3id.org/annett-o/GAN"}]}</data>
      <data key="d3">["http://w3id.org/annett-o/GAN"]</data>
    </node>
    <node id="140183389514368">
      <data key="d0">0</data>
      <data key="d1">{"Searchs": [{"HasType": "hasLoss", "Type": "", "Name": "", "Op": "", "Value": "", "HashOn": "type"}, {"HasType": "hasTaskType", "Type": "", "Name": "", "Op": "", "Value": "", "HashOn": "type"}]}</data>
      <data key="d2">{"http://w3id.org/annett-o/simple_classification": [{"type": "LossFunction", "name": "CategoricalCrossEntropy", "found": true, "hash": "LossFunction", "annconfig": "http://w3id.org/annett-o/simple_classification"}, {"type": "SupervisedClassification", "name": "simple_classification_type", "found": true, "hash": "SupervisedClassification", "annconfig": "http://w3id.org/annett-o/simple_classification"}]}</data>
      <data key="d3">["http://w3id.org/annett-o/simple_classification"]</data>
    </node>
    <edge source="140183392034704" target="140183389301056" />
    <edge source="140183392034704" target="140183389511888" />
    <edge source="140183392034704" target="140183389514368" />
  </graph>
</graphml>
"""

G = nx.read_graphml(StringIO(graphml_data))

# Create a dictionary of node labels from the 'name' attribute (which was key="d0" in the GraphML).
labels = {}
for node, data in G.nodes(data=True):
    # data is a dict like {"name": ..., "criteria": ..., "splitProperties": ..., "annConfigs": ...}
    name = data.get("name", "")
    criteria = data.get("criteria", "")
    ann_configs = data.get("annConfigs", "")
    
    # Build a custom label with line breaks or any format you like
    custom_label = f"{name}\n{criteria}\n{ann_configs}"
    labels[node] = custom_label


pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, labels=labels, with_labels=True, node_size=1000, font_size=8, arrows=True)
# nx.draw(G, pos, with_labels=True, node_size=1000, font_size=8, arrows=True)
plt.savefig('test_taxonomy.png')
print(nx.is_directed_acyclic_graph(G))
input()