from owlready2 import *

# Load the ontology
onto = get_ontology("ontology/annett-o-0.1.owl").load()

# Classes
print("Classes:")
for cls in onto.classes():
    print(cls.name)

# Object Properties
print("\nObject Properties:")
for prop in onto.object_properties():
    print(f"{prop.name}: Domain = {prop.domain}, Range = {prop.range}")

# Data Properties
print("\nData Properties:")
for prop in onto.data_properties():
    print(f"{prop.name}: Domain = {prop.domain}, Range = {prop.range}")

# Restrictions
print("\nRestrictions:")
for cls in onto.classes():
    for restriction in cls.is_a:
        if isinstance(restriction, Restriction):
            print(f"{cls.name} has restriction on {restriction.property.name}: {restriction}")

# # Iterate over instances and their properties
# print("\nInstances and Properties:")
# for cls in onto.classes():
#     for instance in cls.instances():
#         print(f"Instance {instance.name} of class {cls.name}")
#         for prop in instance.get_properties():
#             try:
#                 for value in prop[instance]:
#                     print(f"  {prop.name} -> {value}")
#             except ValueError as e:
#                 print(f"Skipped property {prop.name} for instance {instance.name} due to error: {e}")