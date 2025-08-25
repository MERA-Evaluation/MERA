import os
import json


dataset_link_format = '<a href="../{datasets_dir}/{dataset}/README_ru.md">{dataset} ({mods})</a>'

def format_modalities(mods):
    emoji_map = {
        "I": "🖼️",
        "A": "🔊",
        "V": "📽️"
    }
    return " ".join(emoji_map.get(mod, "") for mod in mods) if mods else ""


def flatten_taxonomy(taxonomy, datasets_dir, max_levels=5):
    flattened_rows = []
    
    def traverse(item, level=1, parents=None):
        if parents is None:
            parents = []
        
        if level > max_levels:
            return
        
        for key, value in item.items():
            if isinstance(item, dict) and not key.startswith("_"):
                current_parents = parents + [key] + [""] * (max_levels - level)
                current_parents = current_parents[:max_levels]

                if isinstance(value, dict):
                    nested = value.get("_nest", {})

                    if not nested:
                        # get modalities
                        mod_str = format_modalities(value.get("_mod", []))

                        # get skills
                        skill = [p for p in current_parents if p][-1]
                        skill_datasets = get_datasets_using_skill(skill, datasets_dir)
                        skill_datasets_f = [dataset_link_format.format(
                            datasets_dir=datasets_dir,
                            dataset=dataset,
                            mods=format_modalities(mods),
                        ) for dataset, mods in skill_datasets.items()]

                        # create table row
                        flattened_rows.append(current_parents + [mod_str, "<br>".join(skill_datasets_f)])
                    else:
                        traverse(nested, level + 1, current_parents[:level])

    traverse(taxonomy)
    return flattened_rows

def get_datasets_using_skill(skill, datasets_dir):

    def sort_mods(mods):
        order = ["I", "A", "V"]
        sort_key = lambda mods: [order.index(m) for m in mods]
        mods.sort(key=sort_key)
        return mods

    datasets_using_skill = {}

    for dataset_folder in os.listdir(datasets_dir):
        dataset_meta_path = os.path.join(datasets_dir, dataset_folder, "dataset_meta.json")
        
        if os.path.exists(dataset_meta_path):
            with open(dataset_meta_path, "r") as f:
                dataset_meta = json.load(f)
                skills = dataset_meta.get("skills", [])
                if skill in skills:
                    mods = dataset_meta.get("modalities", [])
                    mods.remove("text")
                    if dataset_folder != "1_example_dataset":
                        datasets_using_skill[dataset_folder] = sort_mods([m[0].upper() for m in mods])

    return datasets_using_skill


def generate_html_table(taxonomy_rows):
    html = ["<table>"]
    html.append("    <thead>")
    html.append("        <tr>")
    html.append("            <th>L1</th>")
    html.append("            <th>L2</th>")
    html.append("            <th>L3</th>")
    html.append("            <th>L4</th>")
    html.append("            <th>Skills</th>")
    html.append("            <th>Modalities</th>")
    html.append("            <th>Datasets</th>")
    html.append("            <th></th>")
    html.append("        </tr>")
    html.append("    </thead>")
    html.append("    <tbody>")
    
    def get_unique_row_count(rows, check_levels):
        return sum(1 for row in rows if all(row[i] == "" or row[i] != "" for i in check_levels))
    
    processed_keys = set()
    i = 0
    while i < len(taxonomy_rows):
        row = taxonomy_rows[i]
        
        row_html = ["        <tr>"]
        
        for level in range(5):
            key = row[level]
            
            if not key:
                continue
            
            unique_key = tuple(row[:level+1])
            
            if unique_key not in processed_keys:
                rowspan = get_unique_row_count(
                    [r for r in taxonomy_rows[i:] if all(r[j] == row[j] for j in range(level+1))], 
                    range(level+1)
                )

                if level == 4:
                    key = "<i>{key}</i>".format(key=key)
                
                if rowspan > 1:
                    row_html.append(f"            <td rowspan=\"{rowspan}\">{key}</td>")
                else:
                    row_html.append(f"            <td>{key}</td>")
                
                processed_keys.add(unique_key)
        
        row_html.append(f"            <td>{row[-2]}</td>")  # Modalities column
        row_html.append(f"            <td>{row[-1]}</td>")  # Datasets column
        
        row_html.append("        </tr>")
        
        html.extend(row_html)
        
        i += 1
        while i < len(taxonomy_rows) and tuple(taxonomy_rows[i][:5]) in processed_keys:
            i += 1
    
    html.append("    </tbody>")
    html.append("</table>")
    
    return "\n".join(html)


def taxonomy_to_html_table(taxonomy_json, datasets_dir):
    if isinstance(taxonomy_json, str):
        taxonomy = json.loads(taxonomy_json)
    else:
        taxonomy = taxonomy_json
    
    flattened_rows = flatten_taxonomy(taxonomy, datasets_dir)
    
    return generate_html_table(flattened_rows)


if __name__ == "__main__":
    datasets_directory = "datasets"
    
    with open("skills/skill_taxonomy.json") as rf:
        example_taxonomy = json.load(rf)
    
    skills_table = taxonomy_to_html_table(example_taxonomy, datasets_directory)
    
    with open("skills/skills_tax_template.md") as rf:
        skills_tax_template = rf.read().strip()

    compiled_template_fstr = compile(skills_tax_template, "<skills_tax_template>", "eval")
    skills_tax_docs = eval(compiled_template_fstr)
    
    with open("docs/skills_tax.md", "w") as wf:
        wf.write(skills_tax_docs)
