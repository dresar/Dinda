import json
import os
import re
from typing import Dict, Tuple

import numpy as np
import pandas as pd


INPUT_CSV = r"c:\Users\eka\Downloads\DINDA\Database R1 - Aneka Jaya Tunas Agro.csv"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed")

COLUMNS = [
    "RETAILER NAME (R1)",
    "JENIS PESTISIDA",
    "PACKAGING",
    "VOLUME (L/KG)",
]

PETA_JENIS_PESTISIDA = {
    "insektisida": 1,
    "herbisida": 2,
    "fungisida": 3,
    "zat pengatur tumbuh": 4,
}


def _norm_text(x: object) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip().lower()


def _parse_volume_to_float(x: object) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return float("nan")
    s = str(x).strip()
    s = s.replace(",", ".")
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return float("nan")
    return float(m[0])


def _encode_retailer(series: pd.Series) -> Tuple[pd.Series, Dict[int, str]]:
    nilai = series.astype(str).fillna("").map(lambda v: v.strip())
    uniq = sorted(set(nilai.tolist()))
    peta = {name: i + 1 for i, name in enumerate(uniq)}
    balik = {i: name for name, i in peta.items()}
    return nilai.map(peta).astype(int), balik


def _encode_jenis(series: pd.Series) -> Tuple[pd.Series, Dict[int, str]]:
    norm = series.map(_norm_text)
    peta = dict(PETA_JENIS_PESTISIDA)
    next_code = max(peta.values(), default=0) + 1
    out = []
    for v in norm.tolist():
        if v in peta:
            out.append(peta[v])
        else:
            if v not in peta:
                peta[v] = next_code
                next_code += 1
            out.append(peta[v])
    balik = {code: name for name, code in peta.items()}
    return pd.Series(out, index=series.index).astype(int), balik


def _encode_packaging(series: pd.Series) -> Tuple[pd.Series, Dict[int, str]]:
    nilai = series.astype(str).fillna("").map(lambda v: v.strip())
    uniq = sorted(set(nilai.tolist()))
    peta = {name: i + 1 for i, name in enumerate(uniq)}
    balik = {i: name for name, i in peta.items()}
    return nilai.map(peta).astype(int), balik


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    hilang = [c for c in COLUMNS if c not in df.columns]
    if hilang:
        raise SystemExit(f"Kolom wajib tidak ditemukan: {', '.join(hilang)}")

    df = df[COLUMNS].copy()
    df["VOLUME (L/KG)"] = df["VOLUME (L/KG)"].map(_parse_volume_to_float)

    df = df.dropna(subset=COLUMNS)
    df = df.drop_duplicates()

    retailer_kode, retailer_map = _encode_retailer(df["RETAILER NAME (R1)"])
    jenis_kode, jenis_map = _encode_jenis(df["JENIS PESTISIDA"])
    packaging_kode, packaging_map = _encode_packaging(df["PACKAGING"])

    out_df = pd.DataFrame(
        {
            "retailer_kode": retailer_kode.values.astype(int),
            "jenis_pestisida_kode": jenis_kode.values.astype(int),
            "packaging_kode": packaging_kode.values.astype(int),
            "volume": df["VOLUME (L/KG)"].astype(float).values,
        }
    )

    out_df = out_df.replace([np.inf, -np.inf], np.nan).dropna()

    out_path = os.path.join(OUT_DIR, "dataset_numerik.csv")
    out_df.to_csv(out_path, index=False)

    out_path_noheader = os.path.join(OUT_DIR, "dataset_numerik_noheader.csv")
    out_df.to_csv(out_path_noheader, index=False, header=False)

    meta = {
        "sumber": INPUT_CSV,
        "kolom_sumber": COLUMNS,
        "kolom_output": ["retailer_kode", "jenis_pestisida_kode", "packaging_kode", "volume"],
        "jumlah_baris_output": int(out_df.shape[0]),
        "encoding": {
            "retailer_kode": {"tipe": "label_encoding_urut_alfabet", "map_kode_ke_nama": retailer_map},
            "jenis_pestisida_kode": {"tipe": "mapping_penelitian_plus_dinamis", "map_kode_ke_label_norm": jenis_map},
            "packaging_kode": {"tipe": "label_encoding_urut_alfabet", "map_kode_ke_packaging": packaging_map},
            "volume": {"tipe": "float", "catatan": "ekstrak angka pertama dari string, koma->titik"},
        },
    }
    meta_path = os.path.join(OUT_DIR, "kamus_encoding.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("OK")
    print("output_csv:", out_path)
    print("output_csv_noheader:", out_path_noheader)
    print("encoding_json:", meta_path)
    print("rows:", int(out_df.shape[0]))


if __name__ == "__main__":
    main()

