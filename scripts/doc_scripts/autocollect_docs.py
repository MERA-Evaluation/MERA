import os
import json
import argparse
from pathlib import Path

from validate_meta import validate_meta


def rename_manual_readme(dataset_dir, language):
    if not (dataset_dir / "README{}_manual.md".format("_ru" if language == "ru" else "")).is_file():
        readme_old = dataset_dir / "README{}.md".format("_ru" if language == "ru" else "")
        readme_old_name = readme_old.name
        try:
            readme_new = readme_old.parent / f"{readme_old.stem}_manual.md"
            os.rename(readme_old, readme_new)
            print(f"OK! Renamed: {readme_old_name} -> {readme_new.name}")
        except FileNotFoundError:
            print("ERROR: README files must be named `README.md` and `README_ru.md`!")
            exit()
    else:
        print(f"OK! Manual README ({language}) already exists.")


def parse_manual_readme(dataset_dir, language):
    readme_manual = Path(dataset_dir) / f"raw_readme_{language}.json"
    with open(readme_manual) as rf:
        pars_with_headings = json.load(rf)
    print(f"OK! Parsed headings from {readme_manual.name}")
    return pars_with_headings


def parse_dataset_meta(dataset_dir):
    try:
        with open(Path(dataset_dir) / "raw_dataset_meta.json") as rf:
            dataset_meta = json.load(rf)
    except FileNotFoundError:
        print("ERROR: no `raw_dataset_meta.json` file!")
        exit()
    return dataset_meta


def get_card_template(language):
    readme_file_name = "README{}_template.md".format("_ru" if language == "ru" else "")
    try:
        with open(Path("docs/templates") / readme_file_name) as rf:
            template_str = rf.read().strip()
    except FileNotFoundError:
        print(f"ERROR: no `{readme_file_name}` file!")
        exit()
    return template_str


def format_prompts(prompts):
    return prompts[0]


def format_data_field_desc(data_field_descriptions, lang, term_dict, data_ex):

    def get_string_desc(field_name, field_desc, data_ex, level=0, is_leaf=True):
        dtype = f" [{type(data_ex).__name__}]" if not isinstance(data_ex, dict) else ""
        line_end = "" if field_desc[-1] == "." else ":" if not is_leaf else ";"
        return f"{' ' * 4 * level}- `{field_name}`{dtype} — {field_desc}{line_end}"

    def get_field_desc(field_name, field_value, term_dict_value, lang, data_ex, level=0):
        if lang in field_value:
            if field_value[lang] == "default":
                fill_desc = term_dict_value["_desc"][lang]
            else:
                fill_desc = field_value[lang]
            desc = get_string_desc(field_name, fill_desc, level=level, data_ex=data_ex)
        else:
            desc_list = [get_string_desc(field_name, term_dict_value["_desc"][lang], level=level, 
                                         is_leaf=False, data_ex=data_ex)]
            for sub_field, sub_field_value in field_value.items():
                if not lang in sub_field_value or sub_field_value.get(lang, None) == "default":
                    fill_desc = term_dict_value[sub_field]
                else:
                    fill_desc = sub_field_value[lang]
                desc = get_field_desc(sub_field, sub_field_value, fill_desc, data_ex=data_ex[sub_field], lang=lang, level=level+1)
                desc_list.append(desc)
            desc = "\n".join(desc_list)
        return desc

    desc_list = []
    for field, value in data_field_descriptions.items():
        desc = get_field_desc(field, value, term_dict[field], lang=lang, data_ex=data_ex[field], level=0)
        desc_list.append(desc)

    description = "\n".join(desc_list)
    description = description[:-1] + description[-1].replace(";", "")

    return description


def process_json_with_term_dict(json_data, term_dict):
    """
    Process JSON structure by replacing "default" values 
    with descriptions from term_dict.
    """
    def process_field(current_value, field_path):
        # Traverse the term_dict to find the corresponding description
        current_dict = term_dict
        for key in field_path:
            if key in current_dict:
                current_dict = current_dict[key]
        
        # If current_value is a dict with language keys
        if isinstance(current_value, dict):
            processed_value = {}
            for lang, val in current_value.items():
                if val == 'default' and '_desc' in current_dict and lang in current_dict['_desc']:
                    processed_value[lang] = current_dict['_desc'][lang]
                else:
                    processed_value[lang] = val
            return processed_value
        
        return current_value

    def recursive_process(obj, path=None):
        path = path or []
        
        if isinstance(obj, dict) and not "en" in obj:
            return {k: recursive_process(v, path + [k]) for k, v in obj.items()}
        else:
            return process_field(obj, path)
    
    return recursive_process(json_data)


def format_metrics(metrics_list, lang, term_dict):
    metrics_desc_list = []
    for metric, desc in metrics_list.items():
        metric_desc = desc[lang] if desc[lang] != "default" else term_dict["metric_descriptions"][metric][lang]
        metrics_desc_list.append(f"- `{metric}`: {metric_desc}")
    return "\n".join(metrics_desc_list)


hb_method_template_ru = """


### Human baseline

Для сравнения качества ответов моделей с тем, как отвечают люди, были проведены замеры ответов людей по следующей методологии.

{hb_method}"""

hb_method_template_en = """


### Human baseline

To compare the quality of the model responses and human responses, measurements of human responses were taken using the following methodology.

{hb_method}"""


hb_res_template_ru = "\n\nРезультаты оценки:\n\n{hb_res_str}"
hb_res_template_en = "\n\nEvaluation results:\n\n{hb_res_str}"


def format_hb(hb_results, hb_method, lang):

    if hb_method:
        hb_method_template = hb_method_template_ru if lang == "ru" else hb_method_template_en
        hb_method_str = hb_method_template.format(hb_method=hb_method)
        if hb_results:
            hb_res_list = []
            for metric, value in hb_results.items():
                hb_res_list.append(f"- {metric} – {value:.2f}")
            hb_res_template = hb_res_template_ru if lang == "ru" else hb_res_template_en
            hb_res_str = hb_res_template.format(
                hb_res_str="\n".join(hb_res_list)
            )
            return hb_method_str + hb_res_str
        else:
            return hb_method_str
    else:
        return ""


def format_skills(skills):
    return ", ".join(skills)


def format_contributors(custom, lang):
    contr_field = "Авторы" if lang == "ru" else "Contributors"
    contributors = custom.get(contr_field, "")
    if contributors:
        contributors = [c.strip() for c in contributors.split(",")]
        if lang == "ru":
            return f"\n\n{contr_field}: {', '.join(contributors)}"
        else:
            return f"\n\n{contr_field}: {', '.join(contributors)}"
    else:
        return ""


def complete_template(dataset_dir, language, meta, custom, term_dict):

    # set computable variables used in template
    computed = {}
    computed["prompts"] = format_prompts(meta["prompts"])
    computed["data_field_descriptions"] = format_data_field_desc(
        meta["data_field_descriptions"], lang=language, term_dict=term_dict["data_field_descriptions"],
        data_ex=meta["data_example"])
    computed["data_example"] = json.dumps(meta["data_example"], indent=4, ensure_ascii=False)
    computed["metrics"] = format_metrics(meta["metrics"], lang=language, term_dict=term_dict)
    computed["human_benchmark"] = format_hb(meta["human_benchmark"], custom["Human baseline"], lang=language)
    computed["skills"] = format_skills(meta["skills"])
    computed["contributors"] = format_contributors(custom, lang=language)

    # format template
    template_fstr = get_card_template(language)
    compiled_template_fstr = compile(template_fstr, "<template_fstr>", "eval")
    formatted_card = eval(compiled_template_fstr)
    formatted_file_name = "README{}.md".format("_ru" if language == "ru" else "")
    with open(dataset_dir / formatted_file_name, "w") as wf:
        wf.write(formatted_card)
    print(f"OK! Formatted card saved at {formatted_file_name}")


def get_dataset_domains(dataset_dir):
    domains = []
    for path in dataset_dir.iterdir():
        if path.is_dir():
            if (path / "test.json").is_file():
                domains.append(path.name)
    return sorted(domains)


def read_dataset(dataset_dir, dataset_domains):
    if dataset_domains:
        test_data = []
        for domain in dataset_domains:
            with open(Path(dataset_dir) / domain / "test.json") as rf:
                test_data.extend(json.load(rf)["data"])
        dataset = {"test": test_data}
    else:
        with open(Path(dataset_dir) / "test.json") as rf:
            dataset = {"test": json.load(rf)["data"]}
    return dataset


def compose_final_meta(raw_meta, dataset, dataset_domains):

    def get_synt_sources(dataset):
        synt_sources = set()
        modalities = ["image", "audio", "video"]
        for sample in dataset["test"]:
            for mod in modalities:
                sample_mod = sample["meta"].get(mod, None)
                if sample_mod is not None:
                    synt_source = sample_mod.get("synt_source", None)
                    if synt_source is not None:
                        synt_sources.add(synt_source)
        return sorted(synt_sources)

    raw_meta["dataset_size"] = len(dataset)
    raw_meta["domains"] = dataset_domains
    raw_meta["synt_source_models"] = get_synt_sources(dataset)
    return raw_meta


def autocollect_cards_and_meta(dataset_dir):
    with open("docs/templates/term_dictionary.json") as rf:
        term_dict = json.load(rf)

    # validate meta & collect dataset_meta.json
    meta = validate_meta(Path(dataset_dir))
    print("Meta validation passed! Saving new `dataset_meta.json`...")
    meta["data_field_descriptions"] = process_json_with_term_dict(
        meta["data_field_descriptions"], term_dict["data_field_descriptions"])
    for metric in meta["metrics"].keys():
        for lang in ["en", "ru"]:
            meta["metrics"][metric][lang] = term_dict["metrics"][metric][lang]
    with open(Path(dataset_dir) / "dataset_meta.json", "w") as wf:
        json.dump(meta, wf, ensure_ascii=False, indent=4)

    # collect readmes
    languages = ["en", "ru"]
    for language in languages:
        custom = parse_manual_readme(dataset_dir, language)
        complete_template(dataset_dir, language, meta=meta, custom=custom, term_dict=term_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=str, help="dataset directory name inside `datasets/`, e.g. 'dataset_name'")

    args = parser.parse_args()

    dataset_dir = Path("datasets") / args.dataset_dir
    autocollect_cards_and_meta(dataset_dir)
