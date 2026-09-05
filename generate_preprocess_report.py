import os
from datetime import datetime
from typing import List

import pandas as pd

from clustering_logic import prapemrosesan_dataset


INPUT_CSV = r"c:\Users\eka\Downloads\DINDA\Database R1 - Aneka Jaya Tunas Agro.csv"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed")


def _md_table(df: pd.DataFrame, max_rows: int = 10) -> str:
    view = df.head(max_rows)
    cols = view.columns.tolist()
    lines: List[str] = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in view.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            s = "" if pd.isna(v) else str(v)
            s = s.replace("\n", " ").replace("\r", " ")
            vals.append(s)
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(OUT_DIR, f"preprocessed_report_{stamp}.csv")
    report_md = os.path.join(OUT_DIR, f"preprocess_diff_report_{stamp}.md")

    log = []

    def cb(msg: str) -> None:
        log.append(msg)

    raw = pd.read_csv(INPUT_CSV)
    raw_kolom = raw.columns.tolist()
    raw_n = int(raw.shape[0])

    df_pre, ringkas = prapemrosesan_dataset(INPUT_CSV, out_csv, progress_cb=cb)
    pre_n = int(df_pre.shape[0])
    pre_kolom = df_pre.columns.tolist()
    kolom_baru = [c for c in pre_kolom if c not in raw_kolom]

    fokus_raw = raw[
        [
            c
            for c in [
                "YEAR",
                "MONTH",
                "DISTRIBUTOR",
                "AREA",
                "RETAILER NAME (R1)",
                "REGENCY/DISTRICT",
                "PRODUCT",
                "JENIS PESTISIDA",
                "PACKAGING",
                "VOLUME (L/KG)",
            ]
            if c in raw.columns
        ]
    ].copy()
    fokus_pre = df_pre[
        [
            c
            for c in [
                "YEAR",
                "MONTH",
                "DISTRIBUTOR",
                "distributor_kode",
                "AREA",
                "area_kode",
                "REGENCY/DISTRICT",
                "regency_kode",
                "PRODUCT",
                "product_kode",
                "RETAILER NAME (R1)",
                "JENIS PESTISIDA",
                "jenis_pestisida_kode",
                "PACKAGING",
                "packaging_kode_angka",
                "VOLUME (L/KG)",
                "volume_normalisasi",
            ]
            if c in df_pre.columns
        ]
    ].copy()

    ring = {
        "jumlah_awal": ringkas.jumlah_awal,
        "jumlah_setelah_bersih": ringkas.jumlah_setelah_bersih,
        "jumlah_dihapus_na": ringkas.jumlah_dihapus_na,
        "jumlah_dihapus_duplikat": ringkas.jumlah_dihapus_duplikat,
        "jumlah_missing_diisi": ringkas.jumlah_missing_diisi,
        "volume_min": ringkas.volume_min,
        "volume_max": ringkas.volume_max,
        "volume_norm_min": ringkas.volume_norm_min,
        "volume_norm_max": ringkas.volume_norm_max,
        "jumlah_unik_distributor": ringkas.jumlah_unik_distributor,
        "jumlah_unik_area": ringkas.jumlah_unik_area,
        "jumlah_unik_regency": ringkas.jumlah_unik_regency,
        "jumlah_unik_product": ringkas.jumlah_unik_product,
    }

    lines: List[str] = []
    lines.append(f"# Diff Report Prapemrosesan ({stamp})")
    lines.append("")
    lines.append(f"- Sumber: `{INPUT_CSV}`")
    lines.append(f"- Output: `{out_csv}`")
    lines.append("")
    lines.append("## Ringkasan")
    lines.append("")
    lines.append(
        f"- {ring['jumlah_setelah_bersih']} baris diproses dari {ring['jumlah_awal']} baris awal "
        f"(NA terhapus: {ring['jumlah_dihapus_na']}, diisi placeholder: {ring['jumlah_missing_diisi']}, duplikasi terhapus: {ring['jumlah_dihapus_duplikat']})."
    )
    lines.append(f"- Min–maks volume: {ring['volume_min']} → {ring['volume_max']}")
    lines.append(f"- Rentang volume normalisasi: {ring['volume_norm_min']} → {ring['volume_norm_max']}")
    lines.append("")
    lines.append("## Kolom baru hasil transformasi")
    lines.append("")
    lines.append("- " + (", ".join(kolom_baru) if kolom_baru else "(tidak ada)"))
    lines.append("")
    lines.append("## Preview Data Mentah (contoh)")
    lines.append("")
    lines.append(_md_table(fokus_raw, max_rows=10))
    lines.append("")
    lines.append("## Preview Data Setelah Prapemrosesan (contoh)")
    lines.append("")
    lines.append(_md_table(fokus_pre, max_rows=10))
    lines.append("")
    lines.append("## Log pipeline (tail)")
    lines.append("")
    for m in log[-15:]:
        lines.append(f"- {m}")
    lines.append("")

    with open(report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("OK")
    print("raw_rows:", raw_n)
    print("pre_rows:", pre_n)
    print("report:", report_md)
    print("preprocessed_csv:", out_csv)


if __name__ == "__main__":
    main()
