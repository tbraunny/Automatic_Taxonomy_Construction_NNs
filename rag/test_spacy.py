from owlready2 import *
from utils.constants import Constants as C
from utils.query_rag import RemoteDocumentIndexer
from utils.owl import *
from utils.embedding_model import EmbeddingModel

from sentence_transformers import util

import spacy


def build_synonym_dictionary(ontology):
    """
    Builds a dictionary of synonyms for ontology classes.

    :param ontology: The ontology object.
    :return: Dictionary mapping synonyms to ontology classes.
    """
    synonym_dict = {}
    for cls in ontology.classes():
        synonyms = cls.label if hasattr(cls, 'label') else []
        for synonym in synonyms:
            synonym_dict[synonym.lower()] = cls
    return synonym_dict


def map_entities_with_similarity(embed_model, entities, ontology):
    """
    Maps extracted entities to ontology classes using semantic similarity.

    :param embed_model: Embedding model for computing similarity.
    :param entities: List of extracted entities.
    :param ontology: The ontology object.
    :return: List of mappings from entities to ontology classes.
    """
    ontology_terms = [cls.name for cls in ontology.classes()]
    ontology_embeddings = embed_model.encode(ontology_terms)

    mapped_entities = []
    for entity in entities:
        entity_text = entity['text']
        entity_embedding = embed_model.encode(entity_text)
        # Compute cosine similarity
        similarities = util.cos_sim(entity_embedding, ontology_embeddings)[0]
        # Find the best match
        best_match_idx = similarities.argmax()
        best_match_cls = ontology.classes()[best_match_idx]
        mapped_entities.append({
            'entity': entity,
            'ontology_class': best_match_cls
        })
    return mapped_entities


def extract_entities(text, nlp_model):
    """
    Extracts entities from text using spaCy.

    :param text: Input text to process.
    :param nlp_model: Loaded spaCy model.
    :return: List of extracted entities.
    """
    doc = nlp_model(text)
    entities = []
    for ent in doc.ents:
        entities.append({
            'text': ent.text,
            'label': ent.label_
        })
    return entities


def main():
    # Load the spaCy model
    nlp_model = spacy.load("en_core_sci_sm")
    embed_model = EmbeddingModel().get_model()

    # Load ontology
    ontology_path = f"./data/owl/{C.ONTOLOGY.FILENAME}"
    onto = get_ontology(ontology_path).load()

    # Build synonym dictionary for ontology
    synonym_dict = build_synonym_dictionary(onto)

    # Simulate a response from the RAG query engine
    query_engine = RemoteDocumentIndexer(device_ip='100.105.5.55', port=5000).get_rag_query_engine()
    response = query_engine.query("Describe the neural network architecture used in AlexNet.")
    print
    # Preprocess response text
    text = response.get('text', '')  # Adjust key as per the actual response structure

    # Extract entities using spaCy
    entities = extract_entities(text, nlp_model)
    print("\nExtracted Entities:")
    for entity in entities:
        print(f" - {entity['text']} ({entity['label']})")

    # Map entities to ontology classes using semantic similarity
    mapped_entities = map_entities_with_similarity(embed_model, entities, onto)

    print("\nMapped Entities to Ontology Classes:")
    for mapping in mapped_entities:
        print(f" - Entity: {mapping['entity']['text']} -> Ontology Class: {mapping['ontology_class'].name}")

    # Save updated ontology (optional)
    updated_ontology_path = "./data/owl/updated_ontology.owl"
    onto.save(file=updated_ontology_path)
    print(f"\nUpdated ontology saved to {updated_ontology_path}")


if __name__ == "__main__":
    main()
