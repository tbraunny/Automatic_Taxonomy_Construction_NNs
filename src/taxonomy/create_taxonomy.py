from utils.owl_utils import *
from utils.annetto_utils import make_thing_classes_readable
from utils.constants import Constants as C
from typing import Optional
from criteria import *
from pathlib import Path
import logging
import json

import networkx as nx
from visualizeutils import visualizeTaxonomy
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
        items = []
        for item in find_instance_properties(ann_config, has_property=has, equals=equals, found=[]):
            if type(item) == int:
                items.append( {'name': str(item), 'subclass': 'int' } )
            elif type(item) == dict:
                item['subclass'] = item['type']
                items.append( item )
            else:
                items.append( {'name': item.name, 'subclass': item.is_a[0].name } )
        #items = [{'name':item, 'subclass':item.is_a} for item in find_instance_properties(ann_config, has_property=has, equals=equals, found=[])]
        subclasses = set([ item['subclass'] for item in items]) #+ '~' + item['name'] for item in items ])
        hashvalue = ' '.join( item +',' for item in subclasses )
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
            "annConfigs": serialize(obj.annConfigs),
            "criteria": serialize(obj.criteria),
            "children": [serialize(c) for c in obj.children],
            "splitKey": serialize(obj.splitKey)
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
    annConfigs: Optional[List] = []
    splitProperties: Optional[List|Dict] = []
    criteria: Criteria|None
    children: Optional[List] = []
    splitKey: Optional[str] = "Empty"
    def __init__(self, name: str, criteria: Optional[Criteria|None]=None,splitProperties={}, annConfigs = [], splitKey = ""):
        super().__init__(name=name,criteria=criteria,children=[],splitProperties=splitProperties,annConfigs=annConfigs, splitKey=splitKey)
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
            node_data['annConfigs'] = json.dumps(serialize(node.annConfigs))
            node_data['splitKey'] = json.dumps(serialize(node.splitKey))
            G.add_node(node_id, **node_data)

            if parent_id:
                G.add_edge(parent_id, node_id)

            for child in node.children:
                add_nodes_edges(child, node_id)

        add_nodes_edges(self)
        return '\n'.join(nx.generate_graphml(G))




def query_instance_properties(instance, query):
    found = []
    for prop in instance.get_properties():
        print('property',prop.name)

        for value in prop[instance]:

            print('value',value,type(value))
            eq = query

            if eq.Op == 'sequal' and isinstance(value,Thing) and eq.Value == value.name:
                insert = {'type': value.is_a[0].name, 'value': value.name, 'name': prop.name, 'found': True}
                if not insert in found:
                    found.append(insert)
            if eq.Op == 'less' and type(value) == int and eq.Value > value and eq.Name == prop.name:
                insert = {'type': prop.name, 'value': value, 'name': instance.name, 'found': True}
                if not insert in found:
                    found.append(insert)
            if eq.Op == 'greater' and etype(value) == int and eq.Value < value and eq.Name == prop.name:
                insert = {'type': prop.name, 'value': value, 'name': instance.name, 'found': True}
                if not insert in found:
                    found.append(insert)
            if eq.Op == 'leq' and type(value) == int and eq.Value >= value and eq.Name == prop.name:
                insert = {'type': prop.name, 'value': value, 'name': instance.name, 'found': True}
                if not insert in found:
                    found.append(insert)
            if eq.Op == 'geq' and type(value) == int and  eq.Value <= value and eq.Name == prop.name:
                insert = {'type': prop.name, 'value': value, 'name': instance.name, 'found': True}
                if not insert in found:
                    found.append(insert)
            if eq.Op == 'equal' and type(value) == int and eq.Value == value and eq.Name == prop.name:
                insert = {'type': prop.name, 'value': value, 'name': instance.name, 'found': True}
                if not insert in found:
                    found.append(insert)
            if eq.Op == 'range' and type(value) == int and eq.Value[0] < value and eq.Value[1] > value and eq.Name == prop.name:
                insert = {'type': prop.name, 'value': value, 'name': instance.name, 'found': True}
                if not insert in found:
                    found.append(insert)
    return found

class TaxonomyCreator:

    levels: List[Criteria]
    
    def __init__(self, ontology: Ontology, criteria: [Criteria] = []):

        self.ontology = ontology # Load ontology as Ontology object
        self.taxonomy = None
        self.levels = criteria
    def create_level(self, ann_configurations, criteria):
        hashmap = {}

        hasTypes = [crit.HasType  for crit in criteria]
        for ann_config in ann_configurations:
            found = []
            networks = get_instance_property_values(ann_config, self.ontology.hasNetwork.name)
            

            logger.info(f"{' ' * 3}ANNConfig: {ann_config}, type: {type(ann_config)}")
            
            # NOTE: ontology.hasNetwork is an ObjectProperty -> returns annett-o-0.1.hasNetwork of type: <class 'owlready2.prop.ObjectPropertyClass'>
            
            # iterate over network
            for network in networks:

                #print(criteria.has)
                #for crit in criteria:
                #    print(has)
                #    found += get_instance_property_values(ann_config, has)

                logger.info(f"{' '  * 5}Network: {network}, type: {type(network)}")

                layers = get_instance_property_values(network, self.ontology.hasLayer.name)
                logger.info(f"{' ' * 5}Layers: {layers}, type: {type(layers)}")

                for crit in criteria:
                    items = []
                    if crit.HasType != '':

                        # BFS search for has properties 
                        items += find_instance_properties(network, has_property=[crit.HasType], equals=[], found=[])

                    if crit.Type != '':
                        searchType = crit.Type
                        if searchType == 'layer_num_units' or searchType == 'layer':
                            #query_instance_properties(network, crit)
                            
                            for layer in layers:
                                items += query_instance_properties(layer, crit)
                                # NOTE: Here we can access the class (ie for layer the subclass we care about) in two ways, we use .is_a[0] more typically
                                logger.info(f"{' ' * 7}Layer: {layer}, type: {type(layer)}")
                                logger.info(f"{' ' * 7}Layer: {layer}, type: {layer.is_a}")

                                subclass = layer.is_a[0]
                                logger.info(f"{' ' * 9}Subclass: {subclass}, type: {type(subclass)}")
                    for item in items:
                        item['hash'] = item[crit.HashOn]

                    found += items
            print('found',found)
            for index, data in enumerate(found): 
                found[index]['annconfig'] = ann_config
            
            hashvalue = set([item['hash'] for item in found])
            hashvalue = hashvalue = ' '.join( str(hash) for hash in hashvalue)
            print('hash: ', hashvalue)
            #input()
            if not hashvalue in hashmap:
                hashmap[hashvalue] = { ann_config : found }
            else:
                hashmap[hashvalue][ann_config] = found
            #print(hashmap)
            #annconfignodes.append(TaxonomyNode(ann_config.name))

            for network in networks:
                
                task_characterizations = get_instance_property_values(network, self.ontology.hasTaskType.name)
                logger.info('\n')

                for task_characterization in task_characterizations:
                    logger.info(f"{' ' * 7}Task Characterization: {task_characterization}, type: {type(task_characterization)}")
                    logger.info(f"{' ' * 7}Task Characterization: {task_characterization}, type: {task_characterization.is_a}")

                    subclass = task_characterization.is_a[0]
                    logger.info(f"{' ' * 9}Subclass: {subclass}, type: {type(subclass)}")
        logger.info('done')
        return hashmap

    def create_taxonomy(self,format='json',faceted=False):
        annconfignodes = []
        # Get all ANNConfiguration Objects
        logger.info(f"ANNConfiguration Class: {self.ontology.ANNConfiguration}, type: {type(self.ontology.ANNConfiguration)}")
        
        #self.ontology.load()
        print('test',self.ontology.ANNConfiguration)
        print(list(self.ontology.classes()))
        #input()
        ann_configurations = get_class_instances(self.ontology.ANNConfiguration)

        logger.info(f"ANNConfigurations: {ann_configurations}, type: {type(ann_configurations)}")
        splits = [ann_configurations]
        topnode = TaxonomyNode(name='Top of Taxonomy',criteria=None, annConfigs = [annconfig.name for annconfig in ann_configurations])
        
        # construct a category for eac
        facetedTaxonomy = { f'level_{index}' : {} for index, level in enumerate(self.levels)  }
        nodes = [topnode]
        for level_index, level in enumerate(self.levels):
            newsplits = []
            newnodes = []

            for index,split in enumerate(splits):
                found = self.create_level(split, level.Searchs) # only supporting has properties right now
                for key in found:
                    
                    split = list(found[key].keys())

                    childnode = TaxonomyNode(f'{level_index}',  criteria=level, splitProperties=found[key], splitKey=key if len(key) > 0 else 'empty', annConfigs = found[key].keys())
                    print(level_index,key,found[key].keys())
                    #input()
                    inserting = f'level_{level_index}'
                    if not key in facetedTaxonomy[inserting]:
                        facetedTaxonomy[inserting][key] = list(found[key].keys())
                    else:
                        facetedTaxonomy[inserting][key] += list(found[key].keys())

                    print(f'key: {key}', len(found[key]), len(key), key == ' ',len(key))
                    if not (len(key) == 0 and len(found[key]) == 1) or faceted: # don't expand leafs
                        nodes[index].children.append(childnode)
                        newnodes.append(childnode)
                        newsplits.append(split)
                    
            splits = newsplits
            nodes = newnodes
            print(nodes)
        print(facetedTaxonomy)
        input()
        if format == 'json':
            return topnode.to_json()
        else:
            return topnode.to_graphml()

def main():

    logger.info("Loading ontology.")
    #ontology_path = f"./data/owl/{C.ONTOLOGY.FILENAME}" 
    ontology_path = f"./data/owl/annett-o-test.owl"

    ontology_path = f"./data/owl/annett-o.owl" 
    # Example Criteria...
    op = SearchOperator(HasType=HasLoss )#, equals=[{'type':'name', 'value':'simple_classification_L2'}])
    op = SearchOperator(Type='layer_num_units',Value=[600,3001],Op='range',Name='layer_num_units', HashOn='found' )#, equals=[{'type':'name', 'value':'simple_classification_L2'}])
    #op = SearchOperator(has= [] , equals=[{'type':'name', 'value':'simple_classification_L2'}])
    #op = SearchOperator(has= [] , equals=[{'type':'value','value':1000,'op':'greater','name':'layer_num_units'}])
    criteria = Criteria()
    criteria.add(op)

    op2 = SearchOperator(HasType=HasTaskType )
    criteria2 = Criteria()
    #criteria2.add(op2)
    
    #op3 = SearchOperator(has=[HasLoss] )
    #criteria3 = Criteria()
    #criteria3.add(op3)
    
    criterias = [criteria]#,criteria2,criteria3]
    print('before load')
    ontology = load_ontology(ontology_path=ontology_path)

    #print(ontology.load())
    logger.info(ontology.instances)
    #input()
    logger.info("Ontology loaded.")

    logger.info("Creating taxonomy from Annetto annotations.")
    taxonomy_creator = TaxonomyCreator(ontology,criteria=criterias)

    output = taxonomy_creator.create_taxonomy(format='aaa')
    visualizeTaxonomy(output)
    print(output)
    logger.info("Finished creating taxonomy.")


if __name__ == "__main__":
    main()
