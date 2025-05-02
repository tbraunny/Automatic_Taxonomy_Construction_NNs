from src.taxonomy.create_taxonomy import TaxonomyCreator, serialize
from utils.annetto_utils import load_annetto_ontology
from src.taxonomy.querybuilder import * 
from src.taxonomy.utils_tabular import create_tabular_view_from_faceted_taxonomy
from src.taxonomy.llm_generate_criteria import llm_create_taxonomy
from utils.constants import Constants as C
import json
from pathlib import Path


def generate_custom_taxonomy(query: str) -> None:
    ontology_path = C.ONTOLOGY.USER_OWL_FILENAME
    ontology = load_annetto_ontology(return_onto_from_path=ontology_path)

    thecriteria = llm_create_taxonomy(query=query, ontology=ontology)

    taxonomy_creator = TaxonomyCreator(ontology, criteria=thecriteria.criteriagroup)
    format = "json"
    topnode, facetedTaxonomy, output = taxonomy_creator.create_taxonomy(
        format=format, faceted=True
    )

    # # create a dataframe
    df = create_tabular_view_from_faceted_taxonomy(
        taxonomy_str=json.dumps(serialize(facetedTaxonomy)), format="json"
    )

    custom_taxonomy_dir = Path("./data/taxonomy/faceted/custom/")
    custom_taxonomy_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(custom_taxonomy_dir / 'user_custom_taxonomy.csv')

if __name__ == "__main__":
    test_query = """Create a taxonomy that highlights the differences between small and large networks"""

    generate_custom_taxonomy(test_query)