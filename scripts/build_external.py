"""Build the post-release datasets (block E): datasets/external/<name>.csv and datasets/external.csv.

The raw files are downloaded by hand from the URLs in datasets/external.csv into RAW_DIR. Row
filters, deduplication and target recoding happen here; drop lists go into external.csv, where the
loader applies them, so the CSVs keep those columns.

Needs pandas, openpyxl and pyreadstat, which are not project dependencies.
Usage, from the repo root: python scripts/build_external.py
"""
import os
import re
import warnings

import pandas as pd
import pyreadstat

warnings.filterwarnings("ignore")

RAW_DIR = os.path.expanduser("~/.sdb-runs/research")
RAW = f"{RAW_DIR}/block_e/raw"
DRY = f"{RAW_DIR}/dryad"
KAG = f"{RAW_DIR}/kaggle/malpdf-efs-malicious-pdf-enhanced-feature-set/FS43"
OUT = "datasets/external"
os.makedirs(OUT, exist_ok=True)
rows = []


def emit(name, df, source, url, first_public, target, positive, drop, license_, lower=False):
    for col in df.select_dtypes(include="category"):
        df[col] = df[col].astype(str)
    if lower:
        for col in df.select_dtypes(include="object"):
            df[col] = df[col].str.strip().str.lower()
    cats = [c for c in df.select_dtypes(include="object").columns if c != target and c not in drop]
    df.to_csv(f"{OUT}/{name}.csv", index=False)
    rows.append(dict(dataset=name, source=source, url=url, first_public=first_public, target=target,
                     positive=positive, drop_columns=";".join(drop), categorical=";".join(cats), license=license_))
    print(f"{name}: {df.shape}, target {df[target].value_counts().to_dict()}, categorical {cats}")


# Dengue deaths, Mexico 2021-2025 (Zenodo 23242722)
d = pd.read_csv(f"{RAW}/23242722/dengue_mexico_2021_2025_minimal.csv")
emit("dengue-mexico-deaths", d, "Zenodo 23242722", "https://zenodo.org/records/23242722", "2026-10-08",
     "death", "1", ["cvegeo", "municipality"], "CC-BY-4.0")

# Non-performing loans, P2P lending (Zenodo 23138626); header row 1 is a label row
n = pd.read_csv(f"{RAW}/23138626/dataset_comparative_study_of_predictive_models_for_non-performing_loan_prediction_in_a_p2p_lending_company - data to use-final.csv", skiprows=1)
n = n[n["npl_status"].notna()].copy()  # trailing blank rows in the source
# LightGBM rejects feature names with JSON-special characters (":", ",", "[" ...)
n.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", c).strip("_") for c in n.columns]
n["npl_status"] = n["npl_status"].astype(int)
emit("p2p-npl", n, "Zenodo 23138626", "https://zenodo.org/records/23138626", "2026-10-04",
     "npl_status", "1", [], "CC-BY-4.0")

# Biotech program outcomes at 36 months (Zenodo 23201979); evaluable rows only
b = pd.read_csv(f"{RAW}/23201979/historical_biotech_benchmark_core.csv")
b = b[b["outcome_evaluable"] == "Yes"].copy()
b["y36"] = b["y36"].astype(int)
bdrop = ["record_id", "company", "program", "nct_id", "indication", "index_date", "historical_reconstruction_status",
         "inclusion_probability", "design_weight", "outcome_36m_status", "outcome_event_date", "outcome_evaluable",
         "outcome_adjudication_confidence", "outcome_horizon_36m_end_date", "index_source_locator"]
emit("biotech-y36", b, "Zenodo 23201979", "https://zenodo.org/records/23201979", "2026-10-07",
     "y36", "1", bdrop, "CC-BY-4.0")

# Cholera result, Tanzania (Zenodo 22695115); Outcome is the death after the result
c = pd.read_csv(f"{RAW}/22695115/CholeraClima-TZ.csv")
emit("cholera-tz-result", c, "Zenodo 22695115", "https://zenodo.org/records/22695115", "2026-09-10",
     "result", "1", ["Outcome", "Date", "Concatenated_wards", "Concatenated_districts"], "CC-BY-4.0", lower=True)

# MalPDF EFS, FS43 (Kaggle dooschoi): benign vs malicious (old and recent malicious stacked)
parts = []
for f in ["feature43_benign.csv", "feature43_old_malicious.csv", "feature43_recent_malicious.csv"]:
    parts.append(pd.read_csv(f"{KAG}/{f}"))
m = pd.concat(parts, ignore_index=True)
m["class"] = m["class"].str.strip().str.lower().replace({"old_malicious": "malicious", "recent_malicious": "malicious"})
emit("malpdf-fs43-binary", m, "Kaggle dooschoi/malpdf-efs-malicious-pdf-enhanced-feature-set (FS43)",
     "https://www.kaggle.com/datasets/dooschoi/malpdf-efs-malicious-pdf-enhanced-feature-set", "2026-10-14",
     "class", "malicious", ["header"], "CC-BY-4.0")

# Meconium aspiration, neonates (Zenodo 23767049); outcome is death, leakage columns dropped per the record review
s = pd.read_stata(f"{RAW}/23767049/S1_Data.dta")
s["Outcome"] = s["Outcome"].astype(str)
mdrop = ["index", "hospital_iD", "Suction", "Typs_of_suctioning", "Bag_mask", "Oxygen_suporet", "PNA", "Sepsis", "Hypo",
         "Other_comp"]  # complication field, same timing class as PNA, Sepsis, Hypo (review 2026-11-01)
emit("meconium-aspiration", s, "Zenodo 23767049", "https://zenodo.org/records/23767049", "2026-10-27",
     "Outcome", "Yes", mdrop, "CC-BY-4.0")

# Low back pain during WFH (Dryad 6wwpzgn5m); the LBP self-report is the target
lbp, _ = pyreadstat.read_sav(f"{DRY}/immobility_lbp/files/lbp2_eng_update.sav", apply_value_formats=True)
emit("immobility-lbp", lbp, "Dryad 10.5061/dryad.6wwpzgn5m", "https://doi.org/10.5061/dryad.6wwpzgn5m", "2026-09-21",
     "lbp", "Yes", ["hnp", "age1", "job1", "smoking1"], "CC0-1.0")


# DRRisk, diabetic retinopathy from EHR (Dryad jwstqjqrm): the three *_original splits stacked; exact duplicates dropped
DRR = f"{DRY}/diabetic_retinopathy"
dr = pd.concat([pd.read_csv(f"{DRR}/DRRisk_{x}_original.csv") for x in ["training", "test", "external_validation"]])
emit("drrisk-retinopathy", dr.drop_duplicates(), "Dryad 10.5061/dryad.jwstqjqrm", "https://doi.org/10.5061/dryad.jwstqjqrm",
     "2026-09-21", "RETINOPATHY", "Y", [], "CC0-1.0")

# Steroid-induced ocular hypertension after vitrectomy (Dryad ghx3ffc3x); post-outcome columns dropped
st = pd.read_excel(f"{DRY}/steroid_induced/data.xlsx")
emit("steroid-ocular-hypertension", st, "Dryad 10.5061/dryad.ghx3ffc3x", "https://doi.org/10.5061/dryad.ghx3ffc3x",
     "2026-10-05", "case", "1", ["id", " ", "max_iop", "when_highiop", "no_of_eyedrop", "mannitol", "acpuncture", "last"],
     "CC0-1.0")

# Aedes aegypti pupal morphometrics, sex (Dryad z08kprrx0); generation and replication are not predictors
ae = pd.read_csv(f"{DRY}/sex_cephalothorax/Measurements.csv", encoding="utf-8-sig")
emit("aedes-pupal-sex", ae, "Dryad 10.5061/dryad.z08kprrx0", "https://doi.org/10.5061/dryad.z08kprrx0",
     "2026-10-14", "Sex", "Male", ["Gen", "Replication"], "CC0-1.0")

# Child obesity risk factors, Beijing (Dryad gf1vhhn44); birthheight is constant
ch = pd.read_csv(f"{DRY}/child_obesity_beijing/20_22.csv")
emit("child-obesity-beijing", ch, "Dryad 10.5061/dryad.gf1vhhn44", "https://doi.org/10.5061/dryad.gf1vhhn44",
     "2026-07-21", "obesity", "1", ["birthheight"], "CC0-1.0")

# Cervical smear abnormality, Kisangani (Dryad 5qfttdzkt); target abnormal Pap smear (0/1, 2 = invalid smear dropped);
# the cytology categories are the target itself
hp = pd.read_excel(f"{DRY}/hpv_genotypes/files/YN_datafile.xlsx", sheet_name="df NEEMA")
hp = hp[hp["Abnormal Pap smear"].isin([0, 1])].copy()
emit("hpv-abnormal-pap", hp, "Dryad 10.5061/dryad.5qfttdzkt", "https://doi.org/10.5061/dryad.5qfttdzkt",
     "2026-05-26", "Abnormal Pap smear", "1",
     ["Numbering", "NILM", "Inflammatory smear", "LSIL", "invalid smears", "ASC-US", "AGC-NOS, AGC-eccNeo",
      "ASC-H, HSIL", "\u2265ASC-US", "no lesions", "minor abnomality", "major abnormality"], "CC0-1.0")

# Sinonasal disease in horses, CT (Dryad np5hqc09w); one row per (clinic, case number), the first kept;
# target primary (1) vs dental (2) sinusitis; diagnosis, treatment and outcome columns dropped
eq = pd.read_csv(f"{DRY}/equine_ct_1600/files/Multicentre_Equine_Head_CT_Study_Data_Repository.csv",
                 encoding="utf-8-sig").drop_duplicates(["sheet", "Case number"], keep="first")
eq = eq[eq["PrimSin1DenSin2"].isin([1, 2])].copy()
eq["PrimSin1DenSin2"] = eq["PrimSin1DenSin2"].astype(int)
eqdrop = ["sheet", "Case number", "Signalment", "Final Diagnosis", "Primary Sinusitis", "Dental sinusitis",
          "Oromax fistula diastema", "Oromax fistula following extraction", "Mycotic sinusitis", "Sinus cyst", "PEH",
          "Traumatic Sinusitis", "Sinonasal tumour type", "Other tumour type free text", "Sinonasaltumour1y0n",
          "Other causes of sinusitis ", "Multiple Diagnoses", "Treatment", "Standing sedation", "UnderGA",
          "Dental extraction Triadan", "Oral extraction", "MTE", "Standard surgical transbuccal technique", "Pin repulsion",
          "Standard Repulsion", "Dental segmentation used", "Extraction meth NR", "Anydentalextract", " Nasofrontal Flap",
          "Maxillary Flap", "Sinoscopic deb MSB fenestration", "Sinoscopic lavage", "Sinus lavage per nasum via fistula",
          "Sinus lavage per nasum via sinonasal ostium", "Surgical sinonasal fistula creation", "AnySinusTx",
          "conservative treatment only", "Weeks of followup after treatment", "Number of treatments", "outcome", "Free Text",
          # apical infection of a tooth on CT: near-definition of the dental class (review 2026-11-01)
          "Ipsilateral apic infect Non-fractured tooth Triadan", "Ipsilateral apic infect fractured tooth Triadan",
          "Contralateral apic infect Nonfractured tooth Triadan", "Contralateral apic Infect fractured tooth Triadan"]
emit("equine-sinonasal-primary", eq, "Dryad 10.5061/dryad.np5hqc09w", "https://doi.org/10.5061/dryad.np5hqc09w",
     "2026-09-03", "PrimSin1DenSin2", "1", eqdrop, "CC0-1.0")

# 30-day adverse events after CAP, France (Harvard Dataverse TJ64VI); the target is the composite
# adv_evt_30; the 30-day delayed outcomes and the components of the target are dropped
cap = pd.read_csv(f"{RAW}/cap-france/r3p.tab", sep="\t")
cap["adv_evt_30"] = cap["adv_evt_30"].astype(int)
emit("cap-france-30d-adverse", cap, "Harvard Dataverse 10.7910/DVN/TJ64VI", "https://doi.org/10.7910/DVN/TJ64VI",
     "2026-06-01", "adv_evt_30", "1",
     ["patient", "deces_30j", "rehosp_nonprog_30", "any_readm_30", "readm_unp_comp", "any_readm_comp", "del_readm_30",
      "del_unp_read_30", "del_dc_30", "del_adv_evt_30", "del_unp_cen", "del_readm_cen"], "CC0-1.0")

pd.DataFrame(rows).to_csv("datasets/external.csv", index=False)
print("wrote datasets/external.csv with", len(rows), "datasets")
