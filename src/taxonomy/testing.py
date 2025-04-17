from owlready2 import get_ontology, Thing,ThingClass,default_world, DataPropertyClass
from collections import deque
from utils.owl_utils import get_highest_subclass_ancestor


def find_property_chain_to_property(ontology, start, target_prop, max_depth=10):
    """
    Traverse starting at 'start' (an Owlready2 individual) following only properties
    whose names start with "has". The function searches for a step where the property
    is exactly equal to 'target_prop' (an Owlready2 property object) and returns a list
    of property chains (each chain is a list of property names) that lead to such a triple.
    
    Parameters:
      start: An Owlready2 individual (e.g., an ANNConfiguration instance).
      target_prop: The target property (Owlready2 property object) to look for.
      max_depth: Maximum number of steps (to avoid infinite recursion).
      
    Returns:
      A list of chains, where each chain is a list of property names (strings) that, when
      applied in order from the start, reach a triple using target_prop.
    """
    target_iri = str(target_prop.iri)
    queue = deque([(start, [])])
    found_chains = []  
    # We'll track visited nodes to avoid cycles:
    visited = [(start,[])]
    
    while queue:
        current, chain = queue.popleft()
        # Stop if we reached maximum allowed depth:
        if len(chain) >= max_depth:
            continue

        # For each object property defined in the ontology (we assume these are the ones we care about).
        if type(current) != ThingClass:
            continue
        for prop in list(ontology.object_properties()):
            # Only follow properties that start with 'has'
            #if not prop.name.startswith("has"):
            #    continue
            searchDomain = [item for dom in prop.domain for item in dom.is_a] + prop.domain
            # If this property is the one we are looking for, record the chain.
            if current in searchDomain:
                # Only add the chain if there is at least one value (i.e. the triple exists)
                targetSearchDomain = target_prop.domain #[item for dom in target_prop.domain for item in dom.is_a] + target_prop.range 
                targetSearchDomain = [get_highest_subclass_ancestor(item) for item in targetSearchDomain]
                if get_highest_subclass_ancestor(current) in targetSearchDomain:
                    #input('really found')
                    #print(chain,current,target_prop)
                    if not chain in found_chains:
                        found_chains.append(chain)

                # Otherwise, we follow the property edge if there are any values.
                try:
                    next_values = prop.range #[item for dom in prop.range for item in dom.is_a] + prop.range
                    #new_chain = chain + [prop.name]
                except Exception:
                    next_values = []
                new_chain = list(chain + [str(prop.iri)])
                for next_node in next_values:
                    if prop not in visited:
                        visited.append(prop) #next_node)
                        try:
                            queue.append((next_node, new_chain))
                        except:
                            pass
    return found_chains

def find_inverse_path(onto, target):
    annconfig = onto.ANNConfiguration
    
    #previous = [item for dom in target.domain for item in dom.is_a] + target.domain
    previous = []
    #for item in target.domain:
    #    if type(item) == ThingClass:
    #        previous += item.is_a
    if isinstance(target, DataPropertyClass):
        #cla = target.domain[0]
        #print(cla)
        #for prop in onto.object_properties():
        #    if not cla in prop.domain and cla in prop.range and not cla in previous and prop != target:
        #        print(prop)
        previous += target.domain
        #print(cla.ancestors(),cla,target.ancestors(),target)
        #input('blabal')
    else:
        previous += target.domain
    previous = [get_highest_subclass_ancestor(item) if type(item) == ThingClass else item for item in previous]
    if annconfig in previous:
        return [[str(target.iri)]]
    queue = deque([(i, []) for i in previous])
    visited = []
    #input()
    finalChains = []
    answer = []
    while queue:
        previous, chain = queue.popleft()
        if type(previous) != ThingClass:
            continue
        for prop in list(onto.object_properties()):
            searchRange = []
            searchRange += prop.domain

            searchRange = [get_highest_subclass_ancestor(item) if type(item) == ThingClass else item for item in searchRange ] + prop.domain
            #input('test')
            if previous in searchRange:
                #searchRange = [item for dom in previous.domain for item in dom.is_a] + previous.domain

                #input()
                new_chain = list(chain)
                if annconfig in prop.range:
                    #print(new_chain,previous,prop)
                    #input('testing')
                    if not chain in finalChains:
                        finalChains.append(chain)

                if not prop in chain:

                    new_chain.append(prop)


                #if len(new_chain) > 1:
                #    print('finding',new_chain, searchRange)
                #    input()
                searchRange = prop.domain #[item for dom in prop.domain for item in dom.is_a] + prop.domain
                #print(searchRange)
                #input()
                for dom in searchRange:
                    if not dom in visited:
                        visited.append(dom)
                        queue.append((dom,new_chain))
        #if previous == annconfig:
        #    print(finalChains,previous)
        #    input('poop')
        #    break
    output = []
    for item in finalChains:
        adding = True
        '''if len(item) > 0:
            item.reverse()
            print(item,item[0].domain,item[-1].range, target.range[0])
            input('java')
            if target.domain[0] == onto.ANNConfiguration and item[0] == target:
                adding = True
            if item[0].domain[0] == onto.ANNConfiguration and (item[-1].range[0] == target.domain[0]):
                adding = True
                print('here')
                input(test)
        '''
        item.reverse()
        if adding:
                item = [str(ite.iri) for ite in item]
                output.append(item)
    return output

def query_annconfig_property_chain(onto, property_chain, ann_config_iri=None, filter_condition=None):
    """
    Build and execute a SPARQL query that retrieves the target values by traversing the property chain 
    from ANNConfiguration to the target property.
    
    Parameters:
      onto:      The Owlready2 ontology instance.
      property_chain: A list of property IRIs (strings) to follow in order.
         For example, for "layer_num_units", if the chain is:
             ANNConfiguration --hasNetwork--> Network --hasLayer--> Layer --layer_num_units--> Literal,
         you would call:
            property_chain = [
                "http://w3id.org/annett-o/hasNetwork",
                "http://w3id.org/annett-o/hasLayer",
                "http://w3id.org/annett-o/layer_num_units"
            ]
      ann_config_iri: (optional) If provided (as string), restrict the query to that specific ANNConfiguration.
      
    Returns:
      A list of tuples from the SPARQL query, each containing:
         (ANNConfiguration, concatenated target value(s))
    """
    # Define a prefix for the ontology (adjust the namespace as needed)
    prefix = "PREFIX ann: <http://w3id.org/annett-o/>\n"
    
    # Restrict to a specific ANNConfiguration if provided,
    # otherwise, get all individuals of type ann:ANNConfiguration.
    if ann_config_iri:
        ann_config_clause = f"VALUES ?annConfig {{ <{ann_config_iri}> }} .\n"
    else:
        ann_config_clause = "?annConfig a ann:ANNConfiguration .\n"
    # Build the triple pattern string.
    # We'll start with ?annConfig, then for each property in the chain, create a new variable.
    patterns = ""
    # Set the current variable to ?annConfig.
    current_var = "?annConfig"
    # Iterate over property chain
    for i, prop in enumerate(property_chain):
        # For all but the last property we assign intermediate variables; last is ?value.
        next_var = f"?node{i+1}" if i < len(property_chain) - 1 else "?value"
        patterns += f"{current_var} <{prop}> {next_var} .\n"
        current_var = next_var

    filter_clause = ""
    if filter_condition:
        filter_clause = f"FILTER({filter_condition})\n"

    # Build the full query.
    query = prefix + f"""
    SELECT ?annConfig (GROUP_CONCAT(?value; separator=", ") AS ?values)
    WHERE {{
        {ann_config_clause}
        {patterns}
        {filter_clause}
    }}
    GROUP BY ?annConfig
    """
    graph = default_world.as_rdflib_graph()
    # Execute the query using Owlready2's SPARQL engine.
    results = list(onto.world.sparql(query))
    results = list(graph.query(query))
    print(results)
    print(query)
    return results
# ----------------- USAGE EXAMPLE ----------------- #
if __name__ == "__main__":
    # Load your ontology
    #onto = get_ontology("./data/owl/annett-o-0.2.owl").load()
    onto = get_ontology("./data/owl/fairannett-o.owl").load()
    
    # Get an ANNConfiguration individual. Adjust the search criteria as needed.
    ann_config = onto.ANNConfiguration #onto.search_one(is_a=onto.ANNConfiguration)
    
    if ann_config is None:
        print("No ANNConfiguration individual found.")
    else:
        # Suppose we want to find the property chain to the property 'layer_num_units'
        # (which might be a data property defined on layers).
        # You may have loaded it as, say, onto.layer_num_units:
        for i in list(onto.data_properties()) + list(onto.object_properties()):
            #if i != onto.hasTrainingOptimizer: #hasTrainingStrategy: #hasActivationFunction:
            #    continue
            #print(i)
            target_prop = i#onto.layer_num_units
            chains1 = find_property_chain_to_property(onto, ann_config, target_prop, max_depth=10)

            chains = find_inverse_path(onto, target_prop)
            if len(chains1) > 0:
                
                print("Found property chains from annconfig to the target property:")
                for chain in chains1:
                    print(" -> ".join(chain))
                print(i,chains)
                for c in chains1:

                    output=query_annconfig_property_chain(onto,c+[str(i.iri)])
                    if len(output) > 0:
                        print(output)
                        #input()
                #print(i,len(chains))
                print(chains1,chains)
                #input('here')
            else:
                print(chains)
                print(i)
                print(chains1,chains)
                #input('not here')

