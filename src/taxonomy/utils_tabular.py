import pandas as pd
import json
from collections import defaultdict

import json
import json
import pandas as pd
from collections import defaultdict

import dataframe_image as dfi
from utils.llm_service import load_environment_llm
from typing import List, Tuple

from utils.pydantic_models import RenamedFacetList, RenamedFacetLabel  # if you moved them elsewhere


def readable_facet_criteria(search):
    op = search.get("Op", "")
    type_ = search.get("Type", {})
    value = search.get("Value", [])
    cluster = search.get("Cluster", "")
    name = search.get("Name", "")

    cluster_type = type_.get("Name", "") if isinstance(type_, dict) else str(type_)

    if cluster == "cluster":
        return f"{name} clustered ({cluster_type})"

    if op == "range" and len(value) == 2:
        return f"{name} ∈ [{value[0]}, {value[1]}]"
    if op in {"less", "greater", "leq", "geq"} and value:
        symbols = {"less": "<", "greater": ">", "leq": "≤", "geq": "≥"}
        return f"{name} {symbols[op]} {value[0]}"
    if op == "sequal" and value:
        return f"{name} = {value[0]}"
    if op == "has":
        return f"Has {name}"
    if op == "none":
        return f"{name} unspecified"
    if op == "scomp":
        return f"{name} ~ \"{value[0]}\""
    if op == "name":
        return f"{name} == {value[0]}"
    
    return f"{name} {op} {value}"

def format_model(model_uri: str) -> str:
    return model_uri.replace("http://w3id.org/annett-o/", "")

def format_characteristic(characteristic: str, facet: str) -> str:
    """
    Format the characteristic string to be more readable.
    """
    if "cluster" in facet.lower():
        return " ".join(characteristic.split('_')[:2]).capitalize()
    return characteristic.replace("http://w3id.org/annett-o/", "")

def create_tabular_view_from_faceted_taxonomy(taxonomy_str: str = "", taxonomy_fp: str = "", format: str = "json", style: str='X') -> pd.DataFrame:
    """
    Create a tabular DataFrame from a faceted taxonomy JSON file.
    Each column is a (Readable Facet Name, Subcategory) pair.
    Rows are model URIs.
    """
    if format != "json":
        raise ValueError("Unsupported format. Only 'json' is supported.")
    
    try:
        if type(taxonomy_str) == str and taxonomy_str != "":
            data = json.loads(taxonomy_str)
        if type(taxonomy_fp) == str and taxonomy_fp != "":
            with open(taxonomy_fp, "r") as f:
                data = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed with exception: {e}")
        return {}
    
    

    model_map = defaultdict(dict)
    column_tuples = []

    for facet_key, group in data.items():        
        criteria = group.get("criteria", {})
        searches = criteria.get("Searchs", [])
        descs = [readable_facet_criteria(s) for s in searches]
        readable_facet = criteria.get("Name", facet_key)
        if descs:
            readable_facet = f"{readable_facet} {'; '.join(descs) if descs else ''}"

        splits = group.get('splits',[])
        for characteristic, model_list in splits.items():
            if characteristic == "criteria":
                continue
            if characteristic == "" or characteristic == " ":
                characteristic = 'False'
                #continue # skip empty subcategories
            characteristic = format_characteristic(characteristic, readable_facet)
            col = (readable_facet, characteristic)
            if col not in column_tuples:
                column_tuples.append(col)
            for model_uri in model_list:
                model_map[model_uri][col] = True

    all_models = sorted(model_map.keys())
    all_columns = sorted(column_tuples)

    # Build row data
    row_data = []
    for model in all_models:
        row = [model_map[model].get(col, False) for col in all_columns]
        row_data.append(row)

    # multi_index = pd.MultiIndex.from_tuples(all_columns)
    # Use LLM for renaming
    llm_client = load_environment_llm()
    renamed_columns = rename_columns_with_llm(llm_client, all_columns)
    multi_index = pd.MultiIndex.from_tuples(renamed_columns)

    df = pd.DataFrame(row_data, columns=multi_index, index=[format_model(model_uri) for model_uri in all_models])
    df.index.name = "Model"

    if style == 'X':
        df = df.replace(True, "X").replace(False, "")

    


    return df

from typing import List, Tuple
import json
from utils.pydantic_models import FacetLabel, RenamedFacetLabel, RenamedFacetList

def rename_columns_with_llm(llm_client, columns: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Use LLM to rename MultiIndex column labels into human-friendly names using Pydantic structured output.
    """
    facet_objs = [FacetLabel(high_level=h, low_level=l) for h, l in columns]
    input_json = json.dumps([f.model_dump() for f in facet_objs], indent=2)

    prompt = f"""
You are a helpful assistant for making ontology-based taxonomies more readable.

You will receive a list of technical facet labels, each with:
- `high_level`: a general category of the facet
- `low_level`: a specific subcategory or value

Your job is to return human-friendly names for each, keeping them meaningful but concise.

Return your output as a list of dictionaries, wrapped in an `items` field like this:
{{ "items": [{{ ... }}, {{ ... }}] }}

Each dictionary must include:
- `high_level`: (original high-level)
- `low_level`: (original low-level)
- `readable_high`: a human-readable version of the high-level category
- `readable_low`: a human-readable version of the subcategory

Be descriptive but brief (e.g., avoid jargon unless necessary).

Example Input:
[
  {{ "high_level": "Training & Optimization", "low_level": "hasTrainingStrategy" }}
]

Example Output:
{{ 
  "items": [
    {{ "high_level": "Training & Optimization", "low_level": "hasTrainingStrategy", "readable_high": "Training Strategy", "readable_low": "Strategy Type" }}
  ]
}}

Now here is the list you should rewrite:
{input_json}
    """

    # Structured output with wrapper model
    structured_llm = llm_client.llm.with_structured_output(RenamedFacetList)
    try:
        response: RenamedFacetList = structured_llm.invoke(prompt)
        return [(r.readable_high, r.readable_low) for r in response.items]
    except Exception as e:
        print(f"LLM structured output failed: {e}")
        raise

    

if __name__ == "__main__":
    json_fp = "src/taxonomy/test.json"
    # Create DataFrame
    df = create_tabular_view_from_faceted_taxonomy(taxonomy_fp=json_fp, format="json")

    # Display
    print(df.head())

    dfi.export(df,'/home/richw/Josue/Automatic_Taxonomy_Construction_NNs/src/taxonomy/dataframe.png')