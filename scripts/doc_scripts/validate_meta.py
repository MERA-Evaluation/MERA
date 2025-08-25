import argparse
import json
import os


REQUIRED_FIELDS = [
    "dataset_name",
    "modalities",
    "universal_domains",
    "data_example",
    "data_field_descriptions",
    "prompts",
    "metrics",
    "human_benchmark",
    "skills",
    "license",
]
FIRST_ORDER_TYPES = {
    "dataset_name": str,
    "modalities": list,
    "universal_domains": list,
    "data_example": dict,
    "data_field_descriptions": dict,
    "prompts": list,
    "metrics": dict,
    "human_benchmark": dict,
    "skills": list,
    "license": str,
}
DATA_EXAMPLE_TYPES = {"instruction": str, "inputs": dict, "meta": dict}
DATA_DESCRIPTION_TYPES = {
    "instruction": dict,
    "inputs": dict,
    "outputs": dict,
    "meta": dict,
}
FILLED_FIELDS = {
    "dataset_name": [""],
    "modalities": ["", []],
    "data_example": ["", {}],
    "data_field_descriptions": ["", {}],
    "prompts": ["", []],
    "metrics": ["", [], {}],
    "human_benchmark": ["", [], {}],
    "skills": ["", []],
    "license": [""],
}
FILLED_DATA_EXAMPLE = {
    "instruction": ["", -1, "-1", 0, "0"],
    "inputs": ["", {}],
    "outputs": ["", [], {}],
    "meta": ["", {}],
}
FILLED_DATA_DESCRIPTION = {
    "instruction": ["", {}],
    "inputs": ["", {}],
    "outputs": ["", [], {}],
    "meta": ["", {}],
}


class MetaError(Exception):
    pass


def load_json(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def validate_meta_required_fields(meta_template, required_fields):
    # fields that meta_template should have
    diff = set(required_fields) - set(list(meta_template.keys()))
    if len(diff):
        missing = ", ".join(list(diff))
        msg = "Some of the required fields in `meta_template.json` are not found: {missing}".format(
            missing=missing
        )
        raise MetaError(msg)


def find_dataset_has_domains(path_to_data):
    if "test.json" in os.listdir(path_to_data):
        domains_split = False
    else:
        domains_split = True
    return domains_split


def get_domains(path_to_data):
    domains = os.listdir(path_to_data)
    domains = list(filter(lambda x: os.path.isdir(os.path.join(path_to_data, x)), domains))
    domains = list(filter(lambda x: x != "samples", domains))
    return domains


def get_shots_test_data(path_to_data, domains_split):
    if not domains_split:
        # if no split for domains, just take jsons
        shots = load_json(os.path.join(path_to_data, "shots.json"))["data"]
        test = load_json(os.path.join(path_to_data, "test.json"))["data"]
    else:
        # if dataset is split into domains
        shots, test = [], []
        dirs = get_domains(path_to_data)
        # go through each domain to read jsons
        for folder in dirs:
            shots_domain = load_json(os.path.join(path_to_data, folder, "shots.json"))[
                "data"
            ]
            test_domain = load_json(os.path.join(path_to_data, folder, "test.json"))[
                "data"
            ]
            shots.extend(shots_domain)
            test.extend(test_domain)
    return shots, test


def get_domains_list(path_to_data, domains_split):
    if not domains_split:
        domains = []
    else:
        domains = get_domains(path_to_data)
    return domains


def get_synt_source(item):
    for key in item["meta"]:
        if isinstance(item["meta"][key], dict):
            # modality meta
            for modality_key in item["meta"][key]:
                if modality_key == "synt_source":
                    return item["meta"][key][modality_key]
    return [None]


def unpack_filter_list(lst):
    unpack = [elem for pack in lst for elem in pack]
    return list(filter(lambda x: x is not None, unpack))


def validate_meta_types(meta_template, type_dict, item_name):
    for key in type_dict:
        if not isinstance(meta_template[key], type_dict[key]):
            msg = "Field `{key_name}` of `{item_name}` should be of type `{required_type}`, but got `{actual_type}`.".format(
                key_name=key,
                item_name=item_name,
                required_type=str(type_dict[key]),
                actual_type=str(type(meta_template[key])),
            )
            raise MetaError(msg)


def validate_missing_fields(meta_template, required_fields, item_name):
    for key in required_fields:
        if meta_template[key] in required_fields[key] or meta_template[key] is None:
            msg = "You should fill `{field_name}` of `{item_name}`. Now it has `{value}`".format(
                field_name=key, item_name=item_name, value=str(meta_template[key])
            )
            raise MetaError(msg)


def validate_meta_data_example_id(meta):
    if not isinstance(meta["data_example"]["meta"]["id"], int):
        msg = 'Field `id` of `data_example["meta"]` must contain integer value > 0, but got value=`{value}`, type=`{type_name}`.'.format(
            value=meta["data_example"]["meta"]["id"],
            type_name=type(meta["data_example"]["meta"]["id"]),
        )
        raise MetaError(msg)


def validate_number_of_prompts(meta):
    if len(meta["prompts"]) < 10:
        msg = "You are supposed to have at least 10 prompts, but got `{count}` prompts".format(
            count=len(meta["prompts"])
        )
        raise MetaError(msg)


def validate_modalities_meta_dicts(meta):
    for modality_key in ["image", "audio", "video"]:
        if modality_key in meta["data_example"]["meta"]:
            if not isinstance(meta["data_example"]["meta"][modality_key], dict):
                msg = 'Field `{field_name}` of `data_example["meta"]` must be of type `dict`, but got `{name}`.'.format(
                    field_name=modality_key,
                    name=type(meta["data_example"]["meta"][modality_key]),
                )
                raise MetaError(msg)


def check_coincide_fields(dct1, dct2, level, parent=None):
    for key in dct1:
        if key not in dct2:
            if level == 1:
                msg = "Field `{name}` should be both in `data_example` and `data_field_descriptions`.".format(
                    name=key
                )
            else:
                msg = "Field `{name}` of `{parent_name}` should be both in `data_example` and `data_field_descriptions`.".format(
                    name=key, parent_name=parent
                )
            raise MetaError(msg)
        if isinstance(dct1[key], dict):
            check_coincide_fields(dct1[key], dct2[key], level + 1, parent=key)


def has_lang_keys(dct):
    should_have = set(["en", "ru"])
    diff = set(list(dct.keys())) - should_have
    return bool(len(diff))


def is_end_node(dct):
    for key in dct:
        if isinstance(dct[key], dict):
            return False
    return True


def check_description_dicts(dct):
    for key in dct:
        if isinstance(dct[key], dict) and is_end_node(dct[key]):
            has_lang_keys(dct[key])
        elif not isinstance(dct[key], dict):
            msg = "Field `{name}` in `data_field_descriptions` should be of type `dict`, but got `{type_name}`".format(
                name=key, type_name=type(dct[key])
            )
            raise MetaError(msg)
        else:
            check_description_dicts(dct[key])


def validate_metrics_with_hb(meta):
    metrics = meta["metrics"]
    hbs = meta["human_benchmark"]

    metrics = list(metrics.keys())
    hbs = list(hbs.keys())

    diff = set(hbs) - set(metrics)
    if len(diff):
        msg = "There are some new metrics in `human_benchmark` that are not declared in `metrics`. Do not use any metrics that is not stated in `metrics` field."
        raise MetaError(msg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_data",
        default=".",
        type=str,
        help="Path to folder where all data including `meta_template.json` and `readme_template.json` is located",
    )
    return parser.parse_args()


def validate_meta(path_to_data):

    path_to_meta = os.path.join(path_to_data, "raw_dataset_meta.json")

    meta_template = load_json(path_to_meta)

    domains_split = find_dataset_has_domains(path_to_data)

    shots, test = get_shots_test_data(path_to_data, domains_split)

    print("Validating meta fields...")

    validate_meta_required_fields(meta_template, REQUIRED_FIELDS)
    validate_meta_types(meta_template, FIRST_ORDER_TYPES, "meta")
    validate_meta_types(
        meta_template["data_example"], DATA_EXAMPLE_TYPES, 'meta["data_example"]'
    )
    validate_meta_types(
        meta_template["data_field_descriptions"],
        DATA_DESCRIPTION_TYPES,
        'meta["data_field_descriptions"]',
    )
    validate_missing_fields(meta_template, FILLED_FIELDS, "meta")
    validate_missing_fields(
        meta_template["data_example"], FILLED_DATA_EXAMPLE, 'meta["data_example"]'
    )
    validate_missing_fields(
        meta_template["data_field_descriptions"],
        FILLED_DATA_DESCRIPTION,
        'meta["data_field_descriptions"]',
    )
    validate_meta_data_example_id(meta_template)
    validate_number_of_prompts(meta_template)
    validate_modalities_meta_dicts(meta_template)
    validate_metrics_with_hb(meta_template)

    check_coincide_fields(
        meta_template["data_example"], meta_template["data_field_descriptions"], level=1
    )
    check_description_dicts(meta_template["data_field_descriptions"])

    domains = get_domains_list(path_to_data, domains_split)
    synt_sources = sorted(
        set(unpack_filter_list([get_synt_source(item) for item in test + shots]))
    )
    dataset_length = len(test)

    meta = {
        "dataset_name": meta_template["dataset_name"],
        "license": meta_template["license"],
        "dataset_size": dataset_length,
        "description": meta_template["description"],
        "modalities": meta_template["modalities"],
        "skills": meta_template["skills"],
        "domains": domains,
        "universal_domains": meta_template["universal_domains"],
        "synt_source_models": synt_sources,
        "data_example": meta_template["data_example"],
        "data_field_descriptions": meta_template["data_field_descriptions"],
        "prompts": meta_template["prompts"],
        "metrics": meta_template["metrics"],
        "human_benchmark": meta_template["human_benchmark"],
    }

    return meta


if __name__ == "__main__":
    args = parse_args()
    path_to_data = args.path_to_data

    meta = validate_meta(path_to_data)
    print("Validation passed! Saving new `dataset_meta.json`...")
    save_json(os.path.join(path_to_data, "dataset_meta.json"), meta)
