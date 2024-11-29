# OWL 2 Schema Vocabulary Summary with Explanations

### Prefixes and Vocabulary
- **dc:** <http://purl.org/dc/elements/1.1/> — Dublin Core Metadata Elements, a vocabulary for general-purpose metadata.
- **grddl:** <http://www.w3.org/2003/g/data-view#> — Gleaning Resource Descriptions from Dialects of Languages, used for converting HTML/XML data to RDF.
- **owl:** <http://www.w3.org/2002/07/owl#> — Web Ontology Language (OWL), defines the classes and properties for creating ontologies on the web.
- **rdf:** <http://www.w3.org/1999/02/22-rdf-syntax-ns#> — Resource Description Framework, a model for data interchange on the web.
- **rdfs:** <http://www.w3.org/2000/01/rdf-schema#> — RDF Schema, provides additional vocabulary for RDF, like `subClassOf` and `label`.
- **xml:** <http://www.w3.org/XML/1998/namespace> — XML namespace, often used for structuring data.
- **xsd:** <http://www.w3.org/2001/XMLSchema#> — XML Schema Definition, a language for defining data types (e.g., `xsd:string`, `xsd:nonNegativeInteger`).

### Ontology Metadata
- **dc:title:** `"The OWL 2 Schema vocabulary (OWL 2)"` — Title of the OWL 2 vocabulary.
- **rdfs:comment:** Provides a description of the OWL 2 vocabulary and its purpose.
- **rdfs:isDefinedBy:** Links to documentation of OWL 2 RDF-Based Semantics, the model used for RDF-based ontologies.
- **owl:imports:** Specifies that this ontology imports RDF Schema (rdfs).
- **owl:versionIRI:** Specifies the version IRI (Internationalized Resource Identifier) for this ontology.
- **owl:versionInfo:** Contains versioning information for tracking updates.
- **grddl:namespaceTransformation:** Transformation URL for GRDDL, enabling the conversion of data into RDF.

### Classes and Their Purpose

#### Collection Classes
- **owl:AllDifferent:** Declares that members of a collection are pairwise different individuals.
- **owl:AllDisjointClasses:** Specifies that members of a collection are mutually disjoint classes.
- **owl:AllDisjointProperties:** Indicates that all properties in the collection are disjoint.

#### Annotation Classes
- **owl:Annotation:** Represents annotations in the form of subject, predicate, and object triples.
- **owl:AnnotationProperty:** Properties intended for annotations (e.g., metadata).

#### Property Classes
- **owl:AsymmetricProperty:** Asymmetric properties; if property A relates to B, B cannot relate to A.
- **owl:DatatypeProperty:** Represents properties associated with data values.
- **owl:FunctionalProperty:** Declares that a property can have only one value per individual.
- **owl:InverseFunctionalProperty:** Declares that if two values are linked by the property, they refer to the same individual.
- **owl:IrreflexiveProperty:** An irreflexive property cannot relate an individual to itself.
- **owl:NamedIndividual:** A unique, identifiable individual within the ontology. (Instance of a Class)
- **owl:NegativePropertyAssertion:** Declares that a property does not relate certain individuals.
- **owl:SymmetricProperty:** Indicates a property is symmetric (if A relates to B, B also relates to A).
- **owl:TransitiveProperty:** Declares that a property is transitive (if A relates to B and B to C, A relates to C).

#### Core OWL Classes
- **owl:Class:** Defines OWL classes, which group individuals that share characteristics.
- **owl:DataRange:** Defines ranges for data properties; deprecated in favor of `rdfs:Datatype`.
- **owl:DeprecatedClass:** Marks a class as deprecated.
- **owl:DeprecatedProperty:** Marks a property as deprecated.
- **owl:ObjectProperty:** Properties that connect two OWL individuals.
- **owl:Ontology:** Represents an entire OWL ontology.
- **owl:OntologyProperty:** Properties that apply specifically to ontologies.
- **owl:Restriction:** Defines property restrictions on individuals.
- **owl:Thing:** Universal superclass for all OWL individuals.
- **owl:Nothing:** Represents the empty class (a class with no members).

### Key Properties

#### Annotation Properties
- **owl:deprecated:** Marks an entity as deprecated, signaling it should not be used.
- **owl:versionInfo:** Provides information on the version of an entity.
- **owl:incompatibleWith:** Declares an ontology as incompatible with another.
- **owl:backwardCompatibleWith:** Indicates backward compatibility with another ontology.

#### Property Constraints
- **owl:allValuesFrom:** A universal property restriction indicating that all values must be from a specified class.
- **owl:someValuesFrom:** An existential property restriction requiring at least one value from a specified class.
- **owl:cardinality:** Specifies the exact cardinality (number of allowed values).
- **owl:minCardinality:** Specifies the minimum cardinality.
- **owl:maxCardinality:** Specifies the maximum cardinality.
- **owl:qualifiedCardinality:** Restricts an exact qualified cardinality based on a class.
- **owl:maxQualifiedCardinality:** Limits the maximum qualified cardinality based on a class.
- **owl:minQualifiedCardinality:** Limits the minimum qualified cardinality based on a class.

#### Logical Properties
- **owl:complementOf:** Defines a class as the complement of another.
- **owl:unionOf:** Represents a union of classes or data ranges.
- **owl:intersectionOf:** Represents an intersection of classes or data ranges.
- **owl:equivalentClass:** Specifies equivalence between two classes.
- **owl:disjointWith:** Declares two classes to have no members in common.

#### Special Properties
- **owl:sameAs:** Specifies that two individuals are the same.
- **owl:differentFrom:** Declares two individuals to be different.
- **owl:hasKey:** Defines a unique combination of properties that act as a key for a class.
- **owl:propertyChainAxiom:** Specifies a chain of properties as a sub-property.
- **owl:propertyDisjointWith:** Declares two properties as mutually exclusive.
- **owl:hasSelf:** Restricts a property to a self-reference.
- **owl:hasValue:** Restricts a property to a specific value.

#### Ontology Properties
- **owl:imports:** Allows for importing other ontologies.
- **owl:priorVersion:** Marks the prior version of an ontology.
- **owl:versionIRI:** Provides the version IRI for an ontology.

#### Other
- **owl:topObjectProperty:** General object property that relates every individual.
- **owl:bottomObjectProperty:** Relates no individuals (an empty property).
- **owl:topDataProperty:** Connects every individual to every possible data value.
- **owl:bottomDataProperty:** Relates no individuals to any data value.
- **owl:distinctMembers:** Identifies members of an `owl:AllDifferent` collection.
- **owl:onClass:** Class restriction for qualified restrictions.
- **owl:onDataRange:** Data range restriction for qualified restrictions.
- **owl:onDatatype:** Restricts to a specific datatype.
- **owl:oneOf:** Specifies a set of individual or data values (enumeration).

---

### General Usage Notes
- **Defined Relationships:** OWL classes and properties use RDF elements like `rdfs:subClassOf`, `rdfs:label`, and `rdfs:comment` to define relationships.
- **Domain and Range:** Many properties, like `owl:sameAs` and `owl:equivalentClass`, specify domains (the classes they apply to) and ranges (the types of values they return).

This summary covers the essential OWL 2 vocabulary needed for building and understanding OWL ontologies. Use this as a reference for terms, structure, and relationships in OWL-based data models.
