from utils.owl_utils import *
from utils.annetto_utils import make_thing_classes_readable
from utils.constants import Constants as C
from typing import Optional
from criteria import *
from pathlib import Path
import logging
import json

import networkx as nx
# Set up logging @ STREAM level
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def SplitOnCriteria(ontology, annConfigs, has=[],equals=[]):
    '''
    Name: SplitOnCriteria
    Description: splits on a given set criteria. 
    Has is for any has properties in the ontology.
    Equals is for properties. 
    '''
    found = {}
    for ann_config in annConfigs:
        items = [{'name':item, 'subclass':item.is_a} for item in find_instance_properties(ann_config, has_property=has, found=[])]
        subclasses = set([ item['subclass'][0].name for item in items ])
        hashvalue = ''.join( item +',' for item in subclasses )
        if not hashvalue in found:
            found[hashvalue] = { ann_config : items }
        else:
            found[hashvalue][ann_config] = items
    return found


def serialize(obj):
    if isinstance(obj, Thing):
        return obj.iri
    elif isinstance(obj, ThingClass):
        return obj.name
    elif isinstance(obj, Criteria):
        return obj.model_dump()
    elif isinstance(obj, TaxonomyNode):
        return {
            "name": obj.name,
            "splitProperties": serialize(obj.splitProperties),
            "criteria": serialize(obj.criteria),
            "children": [serialize(c) for c in obj.children]
        }
    elif isinstance(obj, list):
        return [serialize(i) for i in obj]
    elif isinstance(obj, dict):
        outdict = {}
        for k,v in obj.items():
            #print(type(k))
            k= serialize(k)
            outdict[k] = serialize(v)
        return outdict
    return obj

class TaxonomyNode(BaseModel):
    name: str
    splitProperties: Optional[List|Dict] = []
    criteria: Criteria|None
    children: Optional[List] = []
    def __init__(self, name: str, criteria: Optional[Criteria|None]=None,splitProperties={}):
        super().__init__(name=name,criteria=criteria,children=[],splitProperties=splitProperties)
    def add_children(self, child):
        self.children.append(children)
    def to_json(self):
        return json.dumps(serialize(self)) #self.model_dump_json(indent=2, serialize_as_any=True)
    
    def to_graphml(self):
        G = nx.DiGraph()
        def add_nodes_edges(node, parent_id=None):
            node_id = id(node)
            node_data = {"name": node.name}
            node_data['criteria'] = json.dumps(serialize(node.criteria))
            node_data['splitProperties'] = json.dumps(serialize(node.splitProperties))
            G.add_node(node_id, **node_data)

            if parent_id:
                G.add_edge(parent_id, node_id)

            for child in node.children:
                add_nodes_edges(child, node_id)

        add_nodes_edges(self)
        return '\n'.join(nx.generate_graphml(G))

class TaxonomyCreator:

    levels: List[Criteria]
    
    def __init__(self, ontology: Ontology, criteria: [Criteria] = []):

        self.ontology = ontology # Load ontology as Ontology object
        self.taxonomy = None
        self.levels = criteria
    
    def create_taxonomy(self):
        annconfignodes = []
        # Get all ANNConfiguration Objects
        logger.info(f"ANNConfiguration Class: {self.ontology.ANNConfiguration}, type: {type(self.ontology.ANNConfiguration)}")

        ann_configurations = get_class_instances(self.ontology.ANNConfiguration)
        logger.info(f"ANNConfigurations: {ann_configurations}, type: {type(ann_configurations)}")
        splits = [ann_configurations]
        topnode = TaxonomyNode(name='Top of Taxonomy',criteria=None)
        nodes = [topnode]
        for level_index, level in enumerate(self.levels):
            newsplits = []
            newnodes = []
            for index,split in enumerate(splits):
                print('current_split',split)
                #input()
                for crit in level.criteria:
                    found = SplitOnCriteria(self.ontology, split, has=crit.has) # only supporting has properties right now
                
                    # TODO -- handle merging -- several different criteria -- don't handle logic or or logic ands
                    for key in found:
                        
                        split = list(found[key].keys())
                        childnode = TaxonomyNode(f'{level_index}',  criteria=level, splitProperties=found[key])
                        nodes[index].children.append(childnode)
                        newnodes.append(childnode)
                        newsplits.append(split)
                    
            splits = newsplits
            nodes = newnodes
            print(nodes)
        print(topnode.to_graphml())
        for ann_config in ann_configurations:
            annconfignodes.append(TaxonomyNode(ann_config.name))
            logger.info(f"{' ' * 3}ANNConfig: {ann_config}, type: {type(ann_config)}")
            # NOTE: ontology.hasNetwork is an ObjectProperty -> returns annett-o-0.1.hasNetwork of type: <class 'owlready2.prop.ObjectPropertyClass'>
            networks = get_instance_property_values(ann_config, self.ontology.hasNetwork.name)

            for network in networks:
                logger.info(f"{' '  * 5}Network: {network}, type: {type(network)}")
                
                task_characterizations = get_instance_property_values(network, self.ontology.hasTaskType.name)
                logger.info(f"{' ' * 5}Task Characterizations: {task_characterizations}, type: {type(task_characterizations)}")

                layers = get_instance_property_values(network, self.ontology.hasLayer.name)
                logger.info(f"{' ' * 5}Layers: {layers}, type: {type(layers)}")

                for layer in layers:
                    # NOTE: Here we can access the class (ie for layer the subclass we care about) in two ways, we use .is_a[0] more typically
                    logger.info(f"{' ' * 7}Layer: {layer}, type: {type(layer)}")
                    logger.info(f"{' ' * 7}Layer: {layer}, type: {layer.is_a}")

                    subclass = layer.is_a[0]
                    logger.info(f"{' ' * 9}Subclass: {subclass}, type: {type(subclass)}")

                logger.info('\n')

                for task_characterization in task_characterizations:
                    logger.info(f"{' ' * 7}Task Characterization: {task_characterization}, type: {type(task_characterization)}")
                    logger.info(f"{' ' * 7}Task Characterization: {task_characterization}, type: {task_characterization.is_a}")

                    subclass = task_characterization.is_a[0]
                    logger.info(f"{' ' * 9}Subclass: {subclass}, type: {type(subclass)}")


def main():

    logger.info("Loading ontology.")
    ontology_path = f"./data/owl/{C.ONTOLOGY.FILENAME}" 
    # ontology_path = f"./data/owl/annett-o-test.owl"

    # Example Criteria...
    op = SearchOperator(has= [HasTaskType] )
    criteria = Criteria()
    criteria.add(op)

    op2 = SearchOperator(has=[HasLayer] )
    criteria2 = Criteria()
    criteria2.add(op2)
    
    op3 = SearchOperator(has=[HasLoss] )
    criteria3 = Criteria()
    criteria3.add(op3)
    
    criterias = [criteria,criteria2,criteria3]

    ontology = load_ontology(ontology_path=ontology_path)
    logger.info("Ontology loaded.")

    logger.info("Creating taxonomy from Annetto annotations.")
    taxonomy_creator = TaxonomyCreator(ontology,criteria=criterias)
    taxonomy_creator.create_taxonomy()
    logger.info("Finished creating taxonomy.")


if __name__ == "__main__":
    main()
