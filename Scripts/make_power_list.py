import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

HIERARCHICAL = {
    # Royalty / high nobility
    "emperors", "empresses", "tsars", "pharaohs",
    "princes", "princesses", "dukes", "duchesses",
    "earls", "countesses", "barons", "baronesses", "lords", "nobles",
    # Military ranks
    "admirals", "generals", "marshals", "commanders", "colonels",
    "majors", "captains", "lieutenants", "sergeants", "corporals",
    "commodores", "recruits", "cadets", "troopers", "gunners", "soldiers",
    # Religious hierarchy
    "popes", "archbishops", "bishops", "abbots",
    "monks", "friars", "priests", "deacons", "novices",
    "pilgrims", "hermits", "disciples",
    # Civil / political
    "chancellors", "governors", "senators", "magistrates", "judges",
    "philosophers", "scholars", "scribes", "clerks", "professors",
    # Low social status
    "slaves", "serfs", "peasants", "commoners", "beggars",
    "paupers", "laborers", "servants", "captives", "prisoners",
    # Court roles
    "knights", "squires",
}

# Load curated pairs (still has shared_domain)
rows = list(csv.DictReader(open(PROJECT_ROOT / "Data" / "curated_1000_plural.csv", encoding="utf-8")))
role_keys = {(r["word1"], r["word2"]) for r in rows if r["shared_domain"] == "roles"}

power_keys = {(w1, w2) for w1, w2 in role_keys
              if w1 in HIERARCHICAL and w2 in HIERARCHICAL}

print(f"Role pairs:       {len(role_keys)}")
print(f"Hierarchical pairs: {len(power_keys)}")
for w1, w2 in sorted(power_keys):
    print(f"  {w1} + {w2}")

# Write power_list.csv using columns from coded CSV
coded = list(csv.DictReader(open(PROJECT_ROOT / "Data" / "curated_1000_plural_coded.csv", encoding="utf-8")))
coded_lookup = {(r["word1"], r["word2"]): r for r in coded}

out_rows = [coded_lookup[(w1, w2)] for w1, w2 in power_keys if (w1, w2) in coded_lookup]
fieldnames = list(coded[0].keys())
out_path = PROJECT_ROOT / "Data" / "power_list.csv"
with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(out_rows)
print(f"\nWrote {len(out_rows)} pairs to {out_path.name}")
