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
import time
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

def unhash_instance_name(instance_name):
    parts = instance_name.split("_", 1)
    if len(parts) == 2:
        return " ".join(
            word.capitalize() for word in parts[1].replace("-", " ").split()
        )
    return instance_name

def format_model(model_uri: str) -> str:
    model_name = model_uri.replace("http://w3id.org/annett-o/", "")
    
    model_name = unhash_instance_name(model_name)
    return model_name

def format_characteristic(characteristic: str, facet: str) -> str:
    """
    Format the characteristic string to be more readable.
    """
    if "cluster_" in facet.lower():
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
        splits = group.get('splits',[])
        criteria = group.get("criteria", {})
        searches = criteria.get("Searchs", [])
        # descs = [readable_facet_criteria(s) for s in searches]
        # TODO: Update readable criteria??
        readable_facet = criteria.get("Name", facet_key)

        # if descs:
        #     readable_facet = f"{readable_facet} {'; '.join(descs) if descs else ''}"

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
    # llm_client = load_environment_llm()
    # print(f"ORIGINAL_COLUMNS: {all_columns}")
    # renamed_columns = rename_columns_with_llm(llm_client, all_columns)
    # print(f"RENAMED_COLUMNS: {renamed_columns}")

    # renamed_columns = all_columns #TODO: REMOVE THIS
    multi_index = pd.MultiIndex.from_tuples(all_columns)

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
    
    prompt = """You are a helpful assistant tasked with improving the readability of taxonomy labels generated from neural network ontologies.
    IMPORTANT INSTRUCTIONS:

    - You must process every original column provided to you.
    - Do not skip, drop, or omit any columns.
    - Preserve the **exact same order** as the original list.
    - For each original column, produce exactly one corresponding output dictionary in the "items" list.
    - If a label is already clear and readable, you may reuse the original label for the `readable_high` and `readable_low` fields with minimal edits.
    - If unsure how to improve a label, preserve its original meaning exactly rather than changing it unnecessarily.
    - Always fill both `readable_high` and `readable_low`.
    - Cluster labels like "Cluster 0", "Cluster 1", etc. must be preserved exactly unless a clear improvement is possible.

    Each item has:
    - `high_level`: a general facet category
    - `low_level`: a specific property or feature associated with the category

    Your goals:
    1. **Preserve important technical context** such as:
    - Numerical ranges (e.g., "Batch Size 16–64")
    - Specific layer types or activation functions (e.g., "ReLU", "BatchNormalization")
    - Model types (e.g., "Autoencoder", "GAN")
    - Cluster names (e.g., "Cluster 0", "Cluster 1") — keep them recognizable
    2. **Handle logical values carefully**:
    - If `low_level` represents `True`, interpret it as "Present" or "Applied".
    - If `low_level` represents `False`, interpret it as "Not Present" or "Not Applied".
    - If clustering, preserve cluster labels like "Cluster 0", "Cluster 1", etc.
3. **Make labels readable while preserving essential technical detail**:
   - You must always include any meaningful numeric thresholds, parameter values, or ranges present in the original label (e.g., units > 5000, dropout rate between 0.2 and 0.5).
   - If the model architecture is mentioned (e.g., convolutional, transformer), keep those terms intact.
   - Do NOT generalize into vague summaries like "Tiny Model" or "Small Network" unless the facet clearly defines what makes it such, and this is reflected in the readable version.
   - Prefer labels that explain *why* something is "tiny" (e.g., "Fewer than 5 layers and unit count below 32") over simply naming it "Tiny Model".
   - If the original column name includes task types (e.g., "Unsupervised", "Classification"), include those explicitly in the readable version.
    4. **Be consistent** in style:
    - Use title casing (capitalize main words).
    - Prefer "within range", "outside range", "not present", "missing" rather than just "True" or "False".
    Return your output as a list of dictionaries, wrapped in an `items` field like this:
    ```json{{ "items": [{{ ... }}, {{ ... }}] }}```
    ---

    Examples:

    Example 1)
    Original Columns:
    ```json
    {{
    [
    {{
        "high_level": "Units < 64 layer_num_units [{{'Name': 'layer_num_units', 'Op': 'less', 'Value': [64]}}]",
        "low_level": "True"
    }}
    ]
    }}

    }}

    Renamed Columns:
    {{
    "items": [
    {{
        "high_level": "Units < 64",
        "low_level": "True",
        "readable_high": "Layer Unit Count < 64,
        "readable_low": "True"
    }}
    ]
    }}

    Example 2)

    Original Columns:
    {{
    [
    {{
        "high_level": "Has BatchNorm Layer hasLayer [{{'Name': 'hasLayer', 'Op': 'sequal', 'Value': ['BatchNormalization']}}]",
        "low_level": "False"
    }}
    ]
    }}

    Renamed Columns:
    {{
    "items": [
    {{
        "high_level": "Has BatchNorm Layer",
        "low_level": "False",
        "readable_high": "Batch Normalization Layer",
        "readable_low": "False"
    }}
    ]
    }}

    Example 3)

    Original Columns)
    {{
    [
    {{
        "high_level": "Learning Rate < 0.001 learning_rate [{{'Name': 'learning_rate', 'Op': 'less', 'Value': [0.001]}}]",
        "low_level": "False"
    }}
    ]
    }}

    Renamed Columns)
    {{
    "items": [
    {{
        "high_level": "Learning Rate < 0.001",
        "low_level": "False",
        "readable_high": "Learning Rate < 0.001",
        "readable_low": "False"
    }}
    ]
    }}

    Example 4)

    Original Columns:

        {{
    [
    {{
        "high_level": "Small Networks (Units 16-128, Layers < 5) layer_num_units [{{'Name': 'layer_num_units', 'Op': 'range', 'Value': [16, 128]}}]; layer_depth [{{'Name': 'layer_depth', 'Op': 'less', 'Value': [5]}}]",
        "low_level": "True"
    }}
    ]
    }}

    Renamed Columns:
    {{
    "items": [
    {{
        "high_level": "Small Networks (Units 16-128, Layers < 5)",
        "low_level": "True",
        "readable_high": "Network Size: Units 16–128, Depth < 5",
        "readable_low": "True"
    }}
    ]
    }}

    Example 5)

    Original Columns:
    {{
    [
    {{
        "high_level": "KMeans Cluster on Dropout + Units layer_features clustered (kmeans)",
        "low_level": "Cluster 0"
    }}
    ]
    }}

    Renamed Columns:

    {{
    "items": [
    {{
        "high_level": "KMeans Cluster on Dropout and Units",
        "low_level": "Cluster 0",
        "readable_high": "KMeans Cluster on Dropout and Units",
        "readable_low": "Cluster 0"
    }}
    ]
    }}




    NOW: Here is the list you should rewrite:
    ```json
    {}
    ```
    """
    prompt = prompt.format(input_json)

    # Structured output with wrapper model
    structured_llm = llm_client.llm.with_structured_output(RenamedFacetList)
    max_retries = 3
    delay_seconds = 1

    for attempt in range(1, max_retries + 1):
        try:
            response: RenamedFacetList = structured_llm.invoke(prompt)
            return [(r.readable_high, r.readable_low) for r in response.items]
        except Exception as e:
            print(f"Attempt {attempt} failed with error: {e}")
            if attempt == max_retries:
                raise  # Give up after max retries
            time.sleep(delay_seconds)  # Wait a little before retrying

    

if __name__ == "__main__":
    json_fp = "src/taxonomy/test.json"
    # Create DataFrame
    df = create_tabular_view_from_faceted_taxonomy(taxonomy_fp=json_fp, format="json")

    # Display
    print(df.head())

    dfi.export(df,'/home/richw/Josue/Automatic_Taxonomy_Construction_NNs/src/taxonomy/dataframe.png')