"""Pull one canonical openFDA prescription label per drug, chunk it by section,
and build the brand <-> generic <-> RxCUI synonym map from label metadata.

Run once:  uv run python -m app.rag.ingest
Writes data/labels.jsonl, data/synonyms.json (and embeddings via app.rag.search).
"""
import json
import re
import time

import httpx

from app.cache import DiskCache
from app.config import CACHE_DIR, LABELS_PATH, OPENFDA_API_KEY, SYNONYMS_PATH

URL = "https://api.fda.gov/drug/label.json"
_cache = DiskCache(CACHE_DIR / "openfda")

# ~50 commonly asked-about drugs across oncology, metabolic, cardiovascular,
# immunology and CNS. Canonical names are the active moiety, without salts.
DRUGS = [
    # oncology
    "pembrolizumab", "nivolumab", "atezolizumab", "durvalumab", "trastuzumab", "bevacizumab",
    "osimertinib", "ibrutinib", "lenalidomide", "rituximab", "olaparib", "palbociclib", "imatinib",
    # metabolic
    "semaglutide", "tirzepatide", "liraglutide", "metformin", "empagliflozin", "dapagliflozin",
    "sitagliptin", "insulin glargine",
    # cardiovascular
    "warfarin", "apixaban", "rivaroxaban", "dabigatran", "atorvastatin", "rosuvastatin", "lisinopril",
    "amlodipine", "metoprolol", "clopidogrel", "amiodarone",
    # immunology
    "adalimumab", "etanercept", "ustekinumab", "dupilumab", "tofacitinib", "upadacitinib", "methotrexate",
    # CNS
    "sertraline", "escitalopram", "bupropion", "gabapentin", "levetiracetam", "aripiprazole",
    "quetiapine", "lithium", "valproic acid",
    # other
    "isotretinoin", "ciprofloxacin", "levothyroxine", "omeprazole", "prednisone",
]

SECTIONS = {
    "boxed_warning": "Boxed warning",
    "indications_and_usage": "Indications and usage",
    "dosage_and_administration": "Dosage and administration",
    "dosage_forms_and_strengths": "Dosage forms and strengths",
    "contraindications": "Contraindications",
    "warnings_and_cautions": "Warnings and precautions",
    "warnings": "Warnings",
    "precautions": "Precautions",
    "adverse_reactions": "Adverse reactions",
    "drug_interactions": "Drug interactions",
    "use_in_specific_populations": "Use in specific populations",
    "pregnancy": "Pregnancy",
    "mechanism_of_action": "Mechanism of action",
    "overdosage": "Overdosage",
}

# Words that vary between label variants of the same active moiety
SALT_WORDS = {
    "sodium", "hydrochloride", "hcl", "potassium", "calcium", "magnesium", "maleate", "tartrate",
    "succinate", "besylate", "mesylate", "citrate", "carbonate", "sulfate", "fumarate", "phosphate",
    "dihydrate", "monohydrate", "trihydrate", "hydrobromide", "acetate", "etexilate", "extended",
    "release", "delayed", "injection", "tablets", "tablet", "and", "-aaaa", "dimeglumine",
}
# Some labels (e.g. Tecentriq, Tagrisso) carry no harmonized `openfda` block, so
# metadata search misses them. Found by product data text instead; brands from here.
UNHARMONIZED_BRANDS = {"atezolizumab": ["Tecentriq"], "osimertinib": ["Tagrisso"]}
CHUNK_CHARS = 1500
OVERLAP = 200
RX = 'openfda.product_type:"HUMAN PRESCRIPTION DRUG"'


def _get(params: dict) -> dict:
    if OPENFDA_API_KEY:
        params = {**params, "api_key": OPENFDA_API_KEY}
    cache_params = {k: v for k, v in params.items() if k != "api_key"}
    cached = _cache.get(URL, cache_params)
    if cached is not None:
        return cached
    for attempt in range(4):
        resp = httpx.get(URL, params=params, timeout=30)
        if resp.status_code == 404:  # openFDA returns 404 for "no matches"
            data = {"results": []}
            break
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(5 * (attempt + 1))
            continue
        resp.raise_for_status()
        data = resp.json()
        break
    else:
        raise RuntimeError("openFDA rate limit: retries exhausted")
    _cache.put(URL, cache_params, data)
    time.sleep(0.3)
    return data


def _moiety(generic: str) -> str:
    """'WARFARIN SODIUM' -> 'warfarin'; 'Insulin Glargine-aglr' -> 'insulin glargine'."""
    g = re.sub(r"-[a-z]{4}\b", "", generic.lower())  # biosimilar suffixes
    tokens = [t for t in re.split(r"[\s,]+", g) if t and t not in SALT_WORDS]
    return " ".join(tokens)


def generic_variants(drug: str) -> list[str]:
    data = _get({"search": f'openfda.generic_name:"{drug}" AND {RX}', "count": "openfda.generic_name.exact", "limit": 100})
    return [r["term"] for r in data.get("results", []) if _moiety(r["term"]) == drug]


def brand_names(variants: list[str]) -> list[str]:
    q = " OR ".join(f'"{v}"' for v in variants)
    data = _get({"search": f"openfda.generic_name.exact:({q}) AND {RX}", "count": "openfda.brand_name.exact", "limit": 100})
    return [r["term"] for r in data.get("results", [])]


def canonical_label(variants: list[str]) -> dict | None:
    """The most recent prescription label with the most sections we index."""
    q = " OR ".join(f'"{v}"' for v in variants)
    data = _get({"search": f"openfda.generic_name.exact:({q}) AND {RX}", "sort": "effective_time:desc", "limit": 10})
    labels = data.get("results", [])
    if not labels:
        return None
    return max(labels, key=lambda l: (sum(1 for s in SECTIONS if s in l), l.get("effective_time", "")))


def unharmonized_label(drug: str) -> dict | None:
    data = _get({"search": f'spl_product_data_elements:"{drug}"', "sort": "effective_time:desc", "limit": 10})
    labels = [l for l in data.get("results", []) if "indications_and_usage" in l]
    if not labels:
        return None
    return max(labels, key=lambda l: (sum(1 for s in SECTIONS if s in l), l.get("effective_time", "")))


def clean_brands(drug: str, brands: list[str]) -> list[str]:
    """Drop repackager 'brands' that are just the generic name, and case duplicates."""
    words = set(drug.split())
    seen, out = set(), []
    for b in brands:
        low = b.lower()
        if words & set(re.split(r"[\s/,()-]+", low)) or low in seen:
            continue
        seen.add(low)
        out.append(b)
    return out


def chunk_text(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= CHUNK_CHARS:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_CHARS, len(text))
        if end < len(text):  # prefer to break at a sentence boundary
            cut = text.rfind(". ", start + CHUNK_CHARS // 2, end)
            end = cut + 1 if cut != -1 else end
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - OVERLAP
    return chunks


def label_chunks(drug: str, label: dict) -> list[dict]:
    out = []
    date = label.get("effective_time")
    for field, section_name in SECTIONS.items():
        if field not in label:
            continue
        body = " ".join(label[field])
        for i, piece in enumerate(chunk_text(body)):
            out.append({
                "id": f"label:{drug.replace(' ', '_')}:{field}:{i}",
                "drug": drug,
                "section": field,
                "section_name": section_name,
                "label_id": label["id"],
                "set_id": label.get("set_id"),
                "effective_time": date,
                "url": f"https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid={label.get('set_id')}",
                # Prefix gives each chunk its context: structure-aware chunking
                "text": f"{drug.title()} — {section_name}: {piece}",
            })
    return out


def main() -> None:
    chunks: list[dict] = []
    synonyms: dict[str, dict] = {}
    for drug in DRUGS:
        variants = generic_variants(drug)
        if variants:
            label = canonical_label(variants)
            brands = clean_brands(drug, sorted(brand_names(variants)))
        else:
            label = unharmonized_label(drug)
            brands = UNHARMONIZED_BRANDS.get(drug, [])
        if label is None:
            print(f"!! no label for {drug}")
            continue
        rxcui = label.get("openfda", {}).get("rxcui", [])
        synonyms[drug] = {"generic": drug, "variants": variants, "brands": brands, "rxcui": rxcui}
        cs = label_chunks(drug, label)
        chunks.extend(cs)
        print(f"{drug:22s} {len(cs):3d} chunks  label {label.get('effective_time')}  brands: {', '.join(brands[:5])}")

    LABELS_PATH.write_text("\n".join(json.dumps(c) for c in chunks) + "\n")
    SYNONYMS_PATH.write_text(json.dumps(synonyms, indent=1, sort_keys=True))
    print(f"\n{len(synonyms)} drugs, {len(chunks)} chunks -> {LABELS_PATH.name}, {SYNONYMS_PATH.name}")


if __name__ == "__main__":
    main()
