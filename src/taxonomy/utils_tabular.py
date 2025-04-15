import pandas as pd
import json
from collections import defaultdict

import json
import json
import pandas as pd
from collections import defaultdict

import dataframe_image as dfi

def readable_facet_criteria(search):
    op = search.get("Op", "")
    type_ = search.get("Type", "")
    value = search.get("Value", [])
    has_type = search.get("HasType", "")
    hash_on = search.get("HashOn", "")

    if op == "range" and isinstance(value, list) and len(value) == 2:
        return f"in range {value}"
    elif op == "cluster" and isinstance(value, list):
        fields = ', '.join(value)
        return f"by [{fields}] using params {type_}"
    elif has_type and hash_on == "type":
        return f""
    elif type_:
        return f"Includes {type_}"
    return "Unrecognized criteria"

def format_model(model_uri: str) -> str:
    return model_uri.replace("http://w3id.org/annett-o/", "")

def format_characteristic(characteristic: str, facet: str) -> str:
    """
    Format the characteristic string to be more readable.
    """
    if "cluster" in facet.lower():
        return " ".join(characteristic.split('_')[:2]).capitalize()
    return characteristic

def create_tabular_view_from_faceted_taxonomy(taxonomy_fp: str, format: str = "json", style: str='X') -> pd.DataFrame:
    """
    Create a tabular DataFrame from a faceted taxonomy JSON file.
    Each column is a (Readable Facet Name, Subcategory) pair.
    Rows are model URIs.
    """
    if format != "json":
        raise ValueError("Unsupported format. Only 'json' is supported.")
    
    with open(taxonomy_fp, "r") as f:
        data = json.load(f)

    model_map = defaultdict(dict)
    column_tuples = []

    for facet_key, group in data.items():
        criteria = group.get("criteria", {})
        searches = criteria.get("Searchs", [])
        descs = [readable_facet_criteria(s) for s in searches]
        readable_facet = criteria.get("Name", facet_key)
        if descs:
            readable_facet = f"{readable_facet} {'; '.join(descs) if descs else ''}"

        for characteristic, model_list in group.items():
            if characteristic == "criteria":
                continue
            if characteristic == "":
                continue # skip empty subcategories
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

    multi_index = pd.MultiIndex.from_tuples(all_columns)
    df = pd.DataFrame(row_data, columns=multi_index, index=[format_model(model_uri) for model_uri in all_models])
    df.index.name = "Model"

    if style == 'X':
        df = df.replace(True, "X").replace(False, "")

    return df

if __name__ == "__main__":
    json_fp = "/home/richw/Josue/Automatic_Taxonomy_Construction_NNs/src/taxonomy/test.json"
    # Create DataFrame
    df = create_tabular_view_from_faceted_taxonomy(taxonomy_fp=json_fp, format="json")

    # Display
    print(df.head())

    dfi.export(df,'/home/richw/Josue/Automatic_Taxonomy_Construction_NNs/src/taxonomy/dataframe.png')