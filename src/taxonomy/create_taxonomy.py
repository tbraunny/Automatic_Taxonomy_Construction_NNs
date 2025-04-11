from utils.owl_utils import *
from utils.annetto_utils import make_thing_classes_readable
from utils.constants import Constants as C
import os
import sys
import re
from typing import Optional

from pathlib import Path
import logging
import json

from collections import deque, defaultdict

import networkx as nx


from rdflib import Graph, Literal, RDF, URIRef, BNode, Namespace
from rdflib.namespace import RDFS, XSD


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


from src.taxonomy.visualizeutils import visualizeTaxonomy
from src.taxonomy.clustering import kmeans_clustering
from src.taxonomy.criteria import *

# Set up logging @ STREAM level
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def find_paths_to_classes(onto):
    mapping = {}
    stack = [(onto.ANNConfiguration,[])]
    path = []
    visited = []
    ignored = [onto.sameLayerAs]
    classmapping = {}
    while len(stack) > 0:
        current,path = stack[0]
        #input('has available')
        del stack[0]
        #path.append(current)
        for prop in onto.object_properties():
            if current in prop.domain:
                copypath = list(path)
                copypath.append(prop)
                mapping[prop] = path
                if len(prop.range) > 0:
                    if type(current) == ThingClass:
                        if not current in classmapping:
                            classmapping[current] = []
                        classmapping[current].append(prop)
                    if not prop.range[0] in stack and not prop.range[0] in visited and not prop.range[0] in ignored:
                        stack.append((prop.range[0], copypath))
                        visited.append(prop.range[0])
        for prop in onto.data_properties():
            if current in prop.domain:
                copypath = list(path)
                copypath.append(prop)
                mapping[prop] = path
                if len(prop.range) > 0:
                    if type(current) == ThingClass:
                        if not current in classmapping:
                            classmapping[current] = []
                        classmapping[current].append(prop)
                    if not prop.range[0] in stack and not prop.range[0] in visited and not prop.range[0] in ignored:
                        stack.append((prop.range[0], copypath))
                        visited.append(prop.range[0])

    #print(classmapping)
    #print(mapping)
    #print(list(mapping.keys()))

    #print(list(classmapping.keys()))
    #print(classmapping[onto.Layer])
    #print(mapping[classmapping[onto.Layer]])
    #input('done')
    return classmapping, mapping

def parse_function(text):
    pattern = r"(\w+)\(([^)]*)\)"
    matches = re.findall(pattern, text)

    args = []
    for match in matches:
        function_name = match[0]
        arguments = match[1].split(',') if match[1] else []
    return function_name, arguments


def find_instance_properties_new(instance, query=[], found=None, visited=None):
    '''
    Finds all properties based on passed in has_properties
    
    Args:
        instance (ThingClass): the class for which is an instance
        has_property: the has properties we are looking for. This is typically edge properties
        equals: a set of dictionary that define some comparison operations
        found: a list of found elements associate to has_property
        visisted: all nodes that have been visisted
    '''
    if visited is None:
        visited = set()
    if instance in visited:
        return found
    visited.add(instance)
    for prop in instance.get_properties():

        for value in prop[instance]:
            eq = query
            values = eq.Value
            
            # single searches
            for index, searchValue in enumerate(values):
                if searchValue.Name == prop.name and searchValue.Op == 'name':
                    insert = {'type': type(value), 'value': value, 'name': prop.name} 
                    if not insert in found[index]:
                        found[index].append(insert)
                if isinstance(value, Thing) and searchValue.Op == "has":
                    inserts = instance.__getattr__(searchValue.Name)
                    if insert:
                        for insert in inserts:
                            insert = {'type': insert.is_a[0].name, 'name': insert.name, 'found': True, 'value': insert.name}
                            if not insert in found[index]:
                                found[index].append(insert)
                #if isinstance(value,Thing) and searchValue.Name == value.name and searchValue.Op == 'propertyvaluename':
                #    insert = {'type': value.is_a[0].name, 'value': value.name, 'name': prop.name} 
                #    if not insert in found:
                #        found[index].append(insert)
                if searchValue.Op == 'sequal' and isinstance(value,Thing) and searchValue.Name == value.name:
                    insert = {'type': value.is_a[0].name, 'value': value.name, 'name': prop.name, 'found': True}
                    if not insert in found[index]:
                        found[index].append(insert)
                if searchValue.Op == 'less' and type(value) == int and searchValue.Value[0] > value and searchValue.Name == prop.name:
                    insert = {'type': prop.name, 'value': value, 'name': instance.name, 'found': True}
                    if not insert in found[index]:
                        found[index].append(insert)
                if searchValue.Op == 'greater' and type(value) == int and searchValue.Value[0] < value and searchValue.Name == prop.name:
                    insert = {'type': prop.name, 'value': value, 'name': instance.name, 'found': True}
                    if not insert in found[index]:
                        found[index].append(insert)
                if searchValue.Op == 'leq' and type(value) == int and searchValue.Value[0] >= value and searchValue.Name == prop.name:
                    insert = {'type': prop.name, 'value': value, 'name': instance.name, 'found': True}
                    if not insert in found[index]:
                        found[index].append(insert)
                if searchValue.Op == 'geq' and type(value) == int and  searchValue.Value[0] <= value and searchValue.Name == prop.name:
                    insert = {'type': prop.name, 'value': value, 'name': instance.name, 'found': True}
                    if not insert in found[index]:
                        found[index].append(insert)
                if searchValue.Op == 'equal' and type(value) == int and searchValue.Value[0] == value and searchValue.Name == prop.name:
                    insert = {'type': prop.name, 'value': value, 'name': instance.name, 'found': True}
                    if not insert in found[index]:
                        found[index].append(insert)
                # range searches
                if searchValue.Op == 'range' and type(value) == int and searchValue.Value[0] < value and searchValue.Value[1] > value and searchValue.Name == prop.name:
                    insert = {'type': prop.name, 'value': value, 'name': instance.name, 'found': True}
                    if not insert in found[index]:
                        found[index].append(insert)


        try:
            if isinstance(value, Thing):
                find_instance_properties_new(value, query=query, found=found,visited=visited)
        except:
            pass

    return found



backTrackMap = {}

def find_instances(annConfig, ontology, query):
    found = [[] for value in query.Value]
    #print(query)

    #find_instance_properties_new(instance, query=[], found=None, visited=None)
    for index,value in enumerate(query.Value):
        val = ontology[value.Name]
        searchFound = False
        stack = [annConfig]
        if val in classmapping:
            #print('here1',classmapping)
            searchFound = True
            val = classmapping[val][0] # assuming this works -- needs testing
        if val in propertymapping:
            searchFound = True
            #print('here2',propertymapping)

            for j in propertymapping[val]:
                newstack = []
                for plate in stack:
                    newstack += plate.__getattr__(j.name)
                stack = newstack
                #print(stack)
                #nn_configurations = get_class_instances(self.ontology.ANNConfiguration)
        if searchFound:
            #print(searchFound,value)
            #input('found')
            for plate in stack:
                for food in plate.__getattr__(value.Name):
                    #found[index].append(food)
                    if value.Op == 'name':
                        insert = {'type': type(food), 'value': food, 'name': value} 
                        if not insert in found[index]:
                            found[index].append(insert)
                    if isinstance(food, Thing) and value.Op == "has":
                        insert = {'type': food.is_a[0].name, 'name': food.name, 'found': True, 'value': food.name}
                        if not insert in found[index]:
                            found[index].append(insert)
                    if value.Op == 'sequal' and isinstance(food,Thing) and value.Name == food.name:
                        insert = {'type': food.is_a[0].name, 'value': food.name, 'name': plate.name, 'found': True}
                        if not insert in found[index]:
                            found[index].append(insert)
                    if value.Op == 'less' and type(food) == int and value.Value[0] > food:
                        insert = {'type': query.Name, 'value': food, 'name': plate.name, 'found': True}
                        if not insert in found[index]:
                            found[index].append(insert)
                    if value.Op == 'greater' and type(food) == int and value.Value[0] < food:
                        insert = {'type': query.Name, 'value': food, 'name': plate.name, 'found': True}
                        if not insert in found[index]:
                            found[index].append(insert)
                    if value.Op == 'leq' and type(food) == int and value.Value[0] >= food:
                        insert = {'type': query.Name, 'value': food, 'name': plate.name, 'found': True}
                        if not insert in found[index]:
                            found[index].append(insert)
                    if value.Op == 'geq' and type(food) == int and value.Value[0] <= food:
                        insert = {'type': query.Name, 'value': food, 'name': plate.name, 'found': True}
                        if not insert in found[index]:
                            found[index].append(insert)
                    if value.Op == 'equal' and type(food) == int and value.Value[0] == food:
                        insert = {'type': query.Name, 'value': food, 'name': plate.name, 'found': True}
                        if not insert in found[index]:
                            found[index].append(insert)
                    # range searches
                    if value.Op == 'range' and type(food) == int and value.Value[0] < food and value.Value[1] > food:
                        insert = {'type': query.Name, 'value': food, 'name': plate.name, 'found': True}
                        if not insert in found[index]:
                            found[index].append(insert)
        if not searchFound:
            values = query.Value 
            query.Value = [value]
            found = find_instance_properties_new(annConfig, query=query, found=[[]], visited=None)
            found[index] = found[0]
    return found


def get_property_from_ann_for_clustering(annconfig, value, query, ontology, vectorize=True):
    items = []
    returnlist = [] 
    #find_instance_properties(ann_config, has_property=hs)
    #items += find_instance_properties(annconfig, has_property=[], equals=[{'type':'name','value':value}], found=[])
    
    # coping original values
    HasType = query.HasType
    values = query.Value
    
    # masking value
    query.Value = []
    

    if query.HasType != '':
        #items.append(find_instances(annconfig,ontology,query))
        #items.append(find_instance_properties_new(annconfig, query, found=[]))
    
        #returnlist = [[str(item['type']) for item in items[0]]]
        items = []

    # masking has type
    query.HasType = ''


    # going value by value to get properies
    if value != None and type(value) == list:
        #for value in values:
        #    query.Value = [value]
        query.Value = values
        items = find_instance_properties_new(annconfig, query, found=[ [] for index, value in enumerate(query.Value)])


    # restore original values 
    query.HasType = HasType
    query.Value = values
   
    # need to do this better....
    
    # vectorize if asked -- this default
    if vectorize:
        for itemlist in items:
            returnlist.append([ item['value'] if type(item['value']) == float or type(item['value']) == int or type(item['value']) == str else str(item['value'])  for item in itemlist]
                    )
    return returnlist

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
    name: str # the taxonomy node will take on the name of the criteria if not defined
    annConfigs: Optional[List] = Field([])
    splitProperties: Optional[List|Dict] = Field([])
    criteria: Criteria|None = Field(None)
    children: Optional[List] = Field([])
    splitKey: Optional[str] = Field("Empty")
    
    def __init__(self, name: str, criteria: Optional[Criteria|None]=None,splitProperties={}, annConfigs = [], splitKey = ""):
        super().__init__(name=name,criteria=criteria,children=[],splitProperties=splitProperties,annConfigs=annConfigs, splitKey=splitKey)
    
    def add_children(self, child):
        self.children.append(child)

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
    
    def to_rdf(self,format='xml'):
        """
        Creates an RDF representation (in Turtle) of this node, 
        including its criteria and recursive children.
        """
        # Create an RDF graph
        g = Graph()

        # Define a custom namespace
        MYNS = Namespace("http://example.org/taxonomy#")
        g.bind("myns", MYNS)

        def add_node(node: "TaxonomyNode", graph: Graph):
            """
            Recursively adds the node (and its children) to the graph,
            returning the URIRef for the current node.
            """
            # Convert the node name to a safe URI fragment
            node_uri = URIRef(MYNS + node.name.replace(" ", "_"))

            # Declare this node as a TaxonomyNode
            graph.add((node_uri, RDF.type, MYNS.TaxonomyNode))

            # Add label (the node's name)
            graph.add((node_uri, RDFS.label, Literal(node.name, datatype=XSD.string)))

            # Add splitKey if it exists
            if node.splitKey:
                graph.add((node_uri, MYNS.splitKey,
                           Literal(node.splitKey, datatype=XSD.string)))

            # Add annConfigs
            if node.annConfigs:
                # Serialize list/dict as JSON and store as literal
                graph.add((node_uri, MYNS.annConfigs,
                           Literal(json.dumps(serialize(node.annConfigs)),
                                   datatype=XSD.string)))

            # Add splitProperties
            if node.splitProperties:
                graph.add((node_uri, MYNS.splitProperties,
                           Literal(json.dumps(serialize(node.splitProperties)),
                                   datatype=XSD.string)))

            # Add Criteria if present
            if node.criteria:
                criteria_bnode = BNode()
                graph.add((node_uri, MYNS.hasCriteria, criteria_bnode))
                graph.add( (criteria_bnode, MYNS.Name, Literal(node.criteria, datatype=XSD.string)  ) )
                # Each SearchOperator in .criteria.Searchs can be a separate BNode
                for op in node.criteria.Searchs:
                    op_bnode = BNode()
                    graph.add((criteria_bnode, MYNS.hasSearchOperator, op_bnode))

                    # Add all fields of the SearchOperator
                    if op.HasType:
                        graph.add((op_bnode, MYNS.HasType,
                                   Literal(op.HasType, datatype=XSD.string)))
                    if op.Type:
                        graph.add((op_bnode, MYNS.Type,
                                   Literal(op.Type, datatype=XSD.string)))
                    if op.Name:
                        graph.add((op_bnode, MYNS.Name,
                                   Literal(op.Name, datatype=XSD.string)))
                    if op.Op:
                        graph.add((op_bnode, MYNS.Op,
                                   Literal(op.Op, datatype=XSD.string)))
                    if op.Value is not None:
                        # Convert to string for simplicity; could refine based on type
                        graph.add((op_bnode, MYNS.Value,
                                   Literal(str(op.Value), datatype=XSD.string)))
                    if op.HashOn:
                        graph.add((op_bnode, MYNS.HashOn,
                                   Literal(op.HashOn, datatype=XSD.string)))

            # Recursively handle children
            for child in node.children:
                child_uri = add_node(child, graph)
                graph.add((node_uri, MYNS.hasChild, child_uri))

            return node_uri

        # Start recursion with `self`
        add_node(self, g)

        # Return Turtle serialization (could switch to "xml", "json-ld", etc.)
        return g.serialize(format=format)


def query_instance_properties(instance, query):
    found = []
    for prop in instance.get_properties():
        #print('property',prop.name)
        for value in prop[instance]:
            #print('value',value,type(value))
            eq = query
            if eq.Op == 'sequal' and isinstance(value,Thing) and eq.Value == value.name:
                insert = {'type': value.is_a[0].name, 'value': value.name, 'name': prop.name, 'found': True}
                if not insert in found:
                    found.append(insert)
            if eq.Op == 'less' and type(value) == int and eq.Value > value and eq.Name == prop.name:
                insert = {'type': prop.name, 'value': value, 'name': instance.name, 'found': True}
                if not insert in found:
                    found.append(insert)
            if eq.Op == 'greater' and type(value) == int and eq.Value < value and eq.Name == prop.name:
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

def createFacetedTaxonomy(topNode: TaxonomyNode):
    searching = [topnode]
    fctaxo = {}
    count = 0
    while len(searching) > 0:
        currentlevel = [j for i in searching for j in i.children]
        #print(currentlevel)
        nameoflevel = ''
        if len(currentlevel) > 0:
            nameoflevel = currentlevel[0].name+ f'_level_{count}'
            fctaxo[nameoflevel] = {}
            fctaxo[nameoflevel]['criteria'] = currentlevel[0].criteria
        for node in currentlevel:
            if not node.splitKey in fctaxo[nameoflevel]:
                fctaxo[nameoflevel][node.splitKey] = node.annConfigs
            else:
                fctaxo[nameoflevel][node.splitKey] += node.annConfigs
        searching = currentlevel
        count += 1
    return fctaxo


class TaxonomyCreator:

    levels: List[Criteria]
    
    def __init__(self, ontology: Ontology, criteria: [Criteria] = []):
        global classmapping, propertymapping 
        self.ontology = ontology # Load ontology as Ontology object

        classmapping, propertymapping = find_paths_to_classes(self.ontology)
        
        self.taxonomy = None
        self.levels = criteria
    def create_level(self, ann_configurations, criteria, ontology):
        hashmap = {}
        criteria = list(criteria) # copy search operators

        #hasTypes = [crit.HasType  for crit in criteria]
        
        otherlist = []
        clusterlist = []

        # seperate out cluster criteria and other things that are not clustering
        for searchop in criteria:
            if searchop.Op != None and  'cluster' in searchop.Op:
                clusterlist.append(searchop)
            else:
                otherlist.append(searchop)

        # do cluster operations
        prefind = {}
        for clusterop in clusterlist:
            #print(clusterop)
            clustervecs = []
            length = -1
            for ann_config in ann_configurations:
                vector = get_property_from_ann_for_clustering(ann_config,clusterop.Value, clusterop, ontology, vectorize=True)

                #length = max(length,len(vector))
                clustervecs.append(vector)
                #print(clusterop,ann_config,vector)




            if len(clustervecs) != 0: # only do clustering if we have some sort of vector returned
                # fill in values that have nothing with a negative one
                #clustervecs = [vector + [-1 for _ in range(length - len(vector))] for vector in clustervecs]
                if clusterop.Type != None and 'kmeans' in clusterop.Type.Name:
                    #fname, arguments = parse_function(clusterop.Type)
                    #print(arguments)
                    centroids=10
                    #if len(arguments) > 0:
                    #    centroids = int(arguments[0])
                    #    whattocast = str(arguments[1])
                    centroids = int(clusterop.Type.Arguments[0])
                    whattocast = str(clusterop.Type.Arguments[1])

                    str_mapping = {i : {} for i in range(len(clusterop.Value))} # precreate mapping for strings
                    str_count = {i : 0 for i in range(len(clusterop.Value))} # precreate mapping for strings
                    remapping = False
                    #print(len(clustervecs[0]))
                    for index, vector in enumerate(clustervecs): # iterate through value list and convert strings to some encoding -- binary for now
                        for vecindex, vecs in enumerate(vector):
                            for vec in vecs:
                                if type(vec) == str:
                                    if not vecindex in str_mapping:
                                        str_mapping[vecindex] = {}
                                        str_count[vecindex] = 0
                                    # check if exists in str_mapping
                                    if not vec in str_mapping[vecindex]:
                                        str_mapping[vecindex][vec] = str_count[vecindex]
                                        str_count[vecindex] += 1
                    inputclustervecs = []
                    length = 0
                    binlength = max([str_count[i] for i in str_count], default=0)
                    #if binlength != 0:
                    for index, vector in enumerate(clustervecs): # iterate through value list and convert strings to some encoding -- binary for now 
                        vectorlist = []
                        for vecindex, vecs in enumerate(vector):
                            if len(vecs) > 0 and type(vecs[0]) == str and whattocast == 'binary':
                                # create binary vector
                                bins = [0 for i in range(binlength)]
                                for vec in vecs:
                                    bins[str_mapping[vecindex][vec]] = 1
                                vectorlist += bins
                            else:
                                vectorlist += vecs
                        length = max(length, len(vectorlist))
                        inputclustervecs.append(vectorlist)
                    inputclustervecs = [vector + [-1 for _ in range(length - len(vector))] for vector in inputclustervecs]
                    
                    centers = kmeans_clustering(inputclustervecs, centroids=centroids )
                    for index, center in enumerate(centers):
                        hashcenter = f'cluster_{center}_{clusterop.Type.Name}_{clusterop.Value}'
                        if not ann_configurations[index] in prefind:
                            prefind[ann_configurations[index]] = {hashcenter : center}
                        else:
                            prefind[ann_configurations[index]][hashcenter] = center
                    #else:
                    #    logging.info('Nothing found so no clusters formulated')
                        
                    #    for ann_config in ann_configurations:
                    #        prefind[ann_config] = {"":""}
        criteria = otherlist
        
        for aindex, ann_config in enumerate(ann_configurations):
            print(aindex,ann_config)
            found = []
            networks = ann_config.__getattr__(self.ontology.hasNetwork.name)
            

            logger.info(f"{' ' * 3}ANNConfig: {ann_config}, type: {type(ann_config)}")
            
            # NOTE: ontology.hasNetwork is an ObjectProperty -> returns annett-o-0.1.hasNetwork of type: <class 'owlready2.prop.ObjectPropertyClass'>
            
            # iterate over network
            #for network in networks:

                #print(criteria.has)
                #for crit in criteria:
                #    print(has)
                #    found += get_instance_property_values(ann_config, has)

             #   logger.info(f"{' '  * 5}Network: {network}, type: {type(network)}")

              #  layers = network.__getattr__(self.ontology.hasLayer.name)
               # logger.info(f"{' ' * 5}Layers: {layers}, type: {type(layers)}")

            for crit in criteria:
                items = []

                #print(classmapping,propertymapping)

                '''for value in crit.Value:
                    val = ontology[value.Name]
                    if val in classmapping:
                        print('here1',classmapping)
                    if val in propertymapping:
                        print('here2',propertymapping)
                        stack = [ann_config]
                        for j in propertymapping[val]:
                            newstack = []
                            for plate in stack:
                                newstack += plate.__getattr__(j.name)
                            stack = newstack
                            #print(stack)
                            #input()
                            #nn_configurations = get_class_instances(self.ontology.ANNConfiguration)
                        for plate in stack:
                            print('plate: ',plate)
                            print(plate.__getattr__(value.Name))

                    #print(value.Name)
                    input()
                '''
                newitems = find_instances(ann_config,ontology,crit) #find_instance_properties_new(ann_config, query=crit, found=[ [] for index, value in enumerate(crit.Value)])

                #print(len(items))
                #input()
                # flatten found
                items += [ item for itemlist in newitems for item in itemlist]
                '''if crit.HasType != '':

                    # BFS search for has properties 
                    #items += find_instance_properties(network, has_property=[crit.HasType], equals=[], found=[])

                if crit.Type != '':
                    searchType = crit.Type

                    # need to implement more search types ......
                    if searchType == 'layer_num_units' or searchType == 'layer':
                        #query_instance_properties(network, crit)
                        
                        for layer in layers:
                            items += query_instance_properties(layer, crit)
                            # NOTE: Here we can access the class (ie for layer the subclass we care about) in two ways, we use .is_a[0] more typically
                            logger.info(f"{' ' * 7}Layer: {layer}, type: {type(layer)}")
                            logger.info(f"{' ' * 7}Layer: {layer}, type: {layer.is_a}")

                            subclass = layer.is_a[0]
                            logger.info(f"{' ' * 9}Subclass: {subclass}, type: {type(subclass)}")'''
                for item in items:
                    item['hash'] = item[crit.HashOn]

                found += items
            #print('found',found)
            for index, data in enumerate(found): 
                found[index]['annconfig'] = ann_config
            if ann_config in prefind:
                for pkey in prefind[ann_config]:
                    found.append({'annconfig':ann_config, 'hash':pkey, 'type': 'int', 'value': prefind[ann_config][pkey]})
            hashvalue = set([item['hash'] for item in found])
            hashvalue = hashvalue = ','.join( str(hash) for hash in hashvalue)
            #print('hash: ', hashvalue)
 
            if not hashvalue in hashmap:
                hashmap[hashvalue] = { ann_config : found }
            else:
                hashmap[hashvalue][ann_config] = found
            #print(hashmap)
            #annconfignodes.append(TaxonomyNode(ann_config.name))

            for network in networks:
                
                task_characterizations = network.__getattr__(self.ontology.hasTaskType.name)
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
        ann_configurations = get_class_instances(self.ontology.ANNConfiguration)

        logger.info(f"ANNConfigurations: {ann_configurations}, type: {type(ann_configurations)}")
        splits = [ann_configurations]
        topnode = TaxonomyNode(name='Top of Taxonomy',criteria=None, annConfigs = [annconfig.name for annconfig in ann_configurations])
        
        # construct a category for eac
        facetedTaxonomy = { level.Name+f'_level_{index}' : {} for index, level in enumerate(self.levels)  }
        nodes = [topnode]
        for level_index, level in enumerate(self.levels):
            newsplits = []
            newnodes = []

            for index,split in enumerate(splits):
                found = self.create_level(split, level.Searchs, self.ontology) # only supporting has properties right now
                for key in found:
                    
                    split = list(found[key].keys())

                    childnode = TaxonomyNode(level.Name,  criteria=level, splitProperties=found[key], splitKey=key if len(key) > 0 else 'empty', annConfigs = found[key].keys())
                    print(level_index,key,found[key].keys())
                    inserting = level.Name + f'_level_{level_index}'
                    facetedTaxonomy[inserting]['criteria'] = level

                    if not key in facetedTaxonomy[inserting]:
                        facetedTaxonomy[inserting][key] = list(found[key].keys())
                    else:
                        facetedTaxonomy[inserting][key] += list(found[key].keys())
                    
                    print(f'key: {key}', len(found[key]), len(key), key == ' ',len(key))
                    if faceted: # alway expand if faceted -- bring them to other grouping
                        nodes[index].children.append(childnode)
                        newnodes.append(childnode)
                        newsplits = [ann_configurations]
                    elif not (len(key) == 0 and len(found[key]) == 1): # don't expand leafs -- when not faceted
                        nodes[index].children.append(childnode)
                        newnodes.append(childnode)
                        newsplits.append(split)

            splits = newsplits
            nodes = newnodes
            print(nodes)

        if format == 'json':
            output = topnode.to_json()
        elif format == 'graphml':
            output = topnode.to_graphml()
        else:
            output = topnode.to_rdf()
        return topnode, facetedTaxonomy, output







def main():
    logging.getLogger().setLevel(logging.WARNING)
    logger.info("Loading ontology.")
    #ontology_path = f"./data/owl/{C.ONTOLOGY.FILENAME}" 
    #ontology_path = f"./data/Annett-o/annett-o-test.owl"
    #ontology_path = f"./data/Annett-o/annett-o-0.1.owl"

    ontology_path = f"./data/owl/fairannett-o.owl" 
    # Example Criteria...
    #op = SearchOperator(HasType=HasLoss )#, equals=[{'type':'name', 'value':'simple_classification_L2'}])
    op = SearchOperator(Type=TypeOperator(name='layer_num_units'),Value=[ValueOperator(Name="layer_num_units",Value=[600,3001],Op="range"), ValueOperator(Name='hasLayer',Op="has")],Op='none',Name='layer_num_units', HashOn='found' )#, equals=[{'type':'name', 'value':'simple_classification_L2'}])
    criteria = Criteria(Name='Layer Num Units')
    criteria.add(op)

    #op2 = SearchOperator(HasType=HasTaskType )
    #criteria2 = Criteria(Name='HasTaskType')
    #criteria2.add(op2)
   
    op3 = SearchOperator(Op='cluster',Type=TypeOperator(Name='kmeans', Arguments=['4','binary']), Value=[ValueOperator(Name='layer_num_units',Op='name')], HasType='')

    criteria3 = Criteria(Name='KMeans Clustering')
    criteria3.add(op3)

    #op3 = SearchOperator(has=[HasLoss] )
    #criteria3 = Criteria()
    #criteria3.add(op3)
    
    criterias = [criteria]
    print('before load')
    ontology = load_ontology(ontology_path=ontology_path)
    print('after load')

    #print(ontology.load())
    logger.info(ontology.instances)
    logger.info("Ontology loaded.")

    logger.info("Creating taxonomy from Annetto annotations.")
    taxonomy_creator = TaxonomyCreator(ontology,criteria=criterias)

    format='json'

    topnode, facetedTaxonomy, output = taxonomy_creator.create_taxonomy(format=format,faceted=True)

    print (json.dumps(serialize(facetedTaxonomy)))

    with open('test.json', 'w') as handle:
        handle.write(json.dumps(serialize(facetedTaxonomy)))

    # print(output)
    # with open('test.xml','w') as handle:
    #     handle.write(output)

    # if format == 'graphml':
    #     visualizeTaxonomy(output)
    #     print(output)
    # logger.info("Finished creating taxonomy.")


if __name__ == "__main__":
    main()
