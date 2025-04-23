import pandas as pd
import json
import re
from collections import defaultdict
import dataframe_image as dfi  # pip install dataframe-image

def readable_facet_criteria(search):
    op = search.get("Op", "")
    type_ = search.get("Type", {})
    value = search.get("Value", [])
    cluster = search.get("Cluster", "")
    name = search.get("Name", "")

    cluster_type = type_.get("Name", "") if isinstance(type_, dict) else str(type_)

    if cluster == "cluster" and value:
        if cluster_type == "kmeans":
            return f"{name} < {value[0]} (kmeans)"
        elif cluster_type == "agg":
            return f"has {name} (agg clustering)"
        elif cluster_type == "graph":
            return f"has {name} (graph clustering)"
        else:
            return f"cluster by {name} using {cluster_type}"
    elif op == "range" and isinstance(value, list) and len(value) == 2:
        return f"{name} in [{value[0]}, {value[1]}]"
    elif op in {"less", "greater", "leq", "geq", "equal"} and value:
        return f"{name} {op} {value[0]}"
    elif op == "sequal" and value:
        return f"{name} = {value[0]}"
    elif op == "has":
        return f"has {name}"
    elif op == "none":
        return f"{name} unspecified"
    elif cluster_type:
        return f"{name} includes {cluster_type}"
    return f"{name} {op} {value}"

def format_model(model_uri: str) -> str:
    return model_uri.replace("http://w3id.org/annett-o/", "")

def format_characteristic(characteristic: str, facet: str) -> str:
    match = re.match(r"cluster_(\\d+)_\\w+_\\[.*?\\]", characteristic)
    if match:
        return f"Cluster {match.group(1)}"
    if characteristic.strip() == "":
        return "False"
    if characteristic.strip() == "True":
        return "True"
    if characteristic.startswith("http"):
        return characteristic.replace("http://w3id.org/annett-o/", "")
    if len(characteristic) > 60:
        return characteristic[:57] + "..."
    return characteristic

def create_tabular_view_from_faceted_taxonomy(taxonomy_str: str = "", taxonomy_fp: str = "", format: str = "json", style: str = 'X') -> pd.DataFrame:
    if format != "json":
        raise ValueError("Unsupported format. Only 'json' is supported.")

    try:
        if taxonomy_str:
            data = json.loads(taxonomy_str)
        elif taxonomy_fp:
            with open(taxonomy_fp, "r") as f:
                data = json.load(f)
        else:
            raise ValueError("Must provide either taxonomy_str or taxonomy_fp")
    except Exception as e:
        raise ValueError(f"Failed to parse taxonomy JSON: {e}")

    model_map = defaultdict(dict)
    column_tuples = []

    for facet_key, group in data.items():
        criteria = group.get("criteria", {})
        searches = criteria.get("Searchs", [])
        facet_label = criteria.get("Name", facet_key)

        short_descriptions = [readable_facet_criteria(s) for s in searches if readable_facet_criteria(s)]
        readable_label = f"{facet_label} ({'; '.join(short_descriptions)})" if short_descriptions else facet_label

        for characteristic, models in group.get("splits", {}).items():
            if not models:
                continue
            subfacet = format_characteristic(characteristic, readable_label)
            col = (readable_label, subfacet)
            if col not in column_tuples:
                column_tuples.append(col)
            for model in models:
                model_map[model][col] = True

    all_models = sorted(model_map.keys())
    all_columns = sorted(column_tuples)

    rows = []
    for model in all_models:
        rows.append([model_map[model].get(col, False) for col in all_columns])

    multi_index = pd.MultiIndex.from_tuples(all_columns)
    df = pd.DataFrame(rows, columns=multi_index, index=[format_model(m) for m in all_models])
    df.index.name = "Model"

    if style == 'X':
        df = df.replace(True, "X").replace(False, "")

    return df

if __name__ == "__main__":
    json_fp = "src/taxonomy/test.json"
    df = create_tabular_view_from_faceted_taxonomy(taxonomy_fp=json_fp, format="json")
    print(df)
    dfi.export(df, "src/taxonomy/dataframe.png")