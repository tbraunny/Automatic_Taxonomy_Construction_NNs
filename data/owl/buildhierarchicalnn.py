from owlready2 import *





def get_instances(onto,owlclass):
    annconfig_class = onto.search_one(iri=owlclass)
    if annconfig_class:
        instances = list(annconfig_class.instances())
    else:
        return []
    print('Instances',instances)
    return instances

def get_subclass(onto,owlclass):
    annconfig_class = onto.search_one(iri=owlclass)
    print(annconfig_class)
    if annconfig_class:
        instances = list(annconfig_class.subclasses())
    else:
        return []
    print('Sub classes',instances)
    return instances

def get_descendents(onto,owlclass):
    annconfig_class = onto.search_one(iri=owlclass)
    if annconfig_class:
        instances = list(annconfig_class.descendants())
    else:
        return []
    print('Descendents',instances)
    return instances

def buildTaxonomy(owlFile='test.owl', classes=[], splittingCriteria=[]):
    
    onto = get_ontology(owlFile).load()
    #sync_reasoner()
    #if onto:
    #    'AnnConfig'
    for child in onto.classes():
        pass
        #print(child.name)
        #if 'Layer' in child.name:
        #    print(child.name,list(child.subclasses()))
        #if 'Conv' in child.name:
        #    print('======')
        #    print(child.name)
        #    print(child.iri)
        #    conv2d_axioms = set(onto.get_parents_of(child))
        #    print(conv2d_axioms)
    for i in classes:
        #pass
        #print(i)
        get_instances(onto,'*'+i)
        get_subclass(onto,'*/'+i)
        get_descendents(onto,'*/'+i)
    #annconfig_class = onto.search_one(iri="*ANNConfiguration")
    
    # Get instances
    #instances = list(annconfig_class.instances())
    #print("Instances:", instances)
    '''
    # Get subclasses
    subclasses = list(annconfig_class.subclasses())
    print("Subclasses:", subclasses)

    # Get properties
    related_properties = [prop for prop in onto.properties() if annconfig_class in prop.domain or annconfig_class in prop.range]
    print("Related properties:", related_properties)'''
    #print(onto)


if __name__ == '__main__':
    classes = ['Layer','ANNConfiguration','TaskCharacterization', 'Layer','LossFunction']
    buildTaxonomy( 'annett-o-test.owl',classes)
