# cv_loader.py

import orjson

from pathlib import Path


def _load_list_of_names(path: Path, name_keys=("name",)) -> set:
    if not path.exists():
        return set()
    with open(path, "rb") as f:
        data = orjson.loads(f.read())
    allowed = set()
    for obj in data:
        for k in name_keys:
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                allowed.add(v.strip().lower())
    return allowed

def _load_subcategories(path: Path) -> tuple[set, dict]:
    if not path.exists():
        return set(), {}
    with open(path, "rb") as f:
        data = orjson.loads(f.read())
    all_names, parent_map = set(), {}
    for obj in data:
        name = (obj.get("name") or "").strip()
        if not name:
            continue
        name_l = name.lower()
        all_names.add(name_l)
        parents = set()
        for p in (obj.get("parent_category") or []):
            if isinstance(p, str) and p.strip():
                parents.add(p.strip().lower())
        parent_map[name_l] = parents
    return all_names, parent_map

def load_controlled_vocabs(base_dir: Path) -> dict:
    files = {
        "category": base_dir / "data_model.category.json",
        "themes": base_dir / "data_model.themes.json",
        "topics": base_dir / "data_model.topics.json",
        "subcategories": base_dir / "data_model.subcategories.json",
        "languages": base_dir / "data_model.languages.json",
        "locations": base_dir / "data_model.locations.json",
        "license": base_dir / "data_model.license.json",
        "intended_purposes": base_dir / "data_model.intended_purposes.json",
        "project_type": base_dir / "data_model.project_type.json",
    }
    cv = {}
    cv["category"] = _load_list_of_names(files["category"], ("name",))
    cv["themes"] = _load_list_of_names(files["themes"], ("name",))
    cv["topics"] = _load_list_of_names(files["topics"], ("name",))
    cv["subcategories_all"], cv["subcat_parents"] = _load_subcategories(files["subcategories"])
    cv["languages"] = _load_list_of_names(files["languages"], ("name", "english_name"))
    cv["locations"] = _load_list_of_names(files["locations"], ("name",))
    cv["license"] = _load_list_of_names(files["license"], ("name",))
    cv["intended_purposes"] = _load_list_of_names(files["intended_purposes"], ("name",))
    cv["project_type"] = _load_list_of_names(files["project_type"], ("name",))
    return cv
