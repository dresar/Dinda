import os

import pandas as pd

from clustering_logic import (
    bangun_tabel_karakteristik_cluster,
    buat_plot_sebaran_cluster,
    jalankan_hac,
    jalankan_kmedoids,
    prapemrosesan_dataset,
    ringkasan_kualitas_sederhana,
)


def main() -> None:
    base = os.path.dirname(os.path.abspath(__file__))
    upload = os.path.join(base, "uploads")
    static_gen = os.path.join(base, "static", "generated")
    os.makedirs(upload, exist_ok=True)
    os.makedirs(static_gen, exist_ok=True)

    raw = os.path.join(upload, "_smoke_raw.csv")
    pre = os.path.join(upload, "_smoke_pre.csv")
    dendro = os.path.join(static_gen, "_smoke_dendro.png")
    sebaran_hac = os.path.join(static_gen, "_smoke_sebaran_hac.png")
    sebaran_km = os.path.join(static_gen, "_smoke_sebaran_km.png")

    df = pd.DataFrame(
        {
            "RETAILER NAME (R1)": ["A", "B", "C", "D", "E", "F", "G", "H"],
            "JENIS PESTISIDA": [
                "Insektisida",
                "Herbisida",
                "Fungisida",
                "Insektisida",
                "Herbisida",
                "Fungisida",
                "Zat Pengatur Tumbuh",
                "Insektisida",
            ],
            "PACKAGING": ["Botol", "Botol", "Sachet", "Sachet", "Botol", "Jerigen", "Jerigen", "Botol"],
            "VOLUME (L/KG)": [10, 12, 2, 3, 13, 30, 25, 11],
        }
    )
    df.to_csv(raw, index=False)

    log = []
    _, ring = prapemrosesan_dataset(raw, pre, progress_cb=log.append)
    df2 = pd.read_csv(pre)

    label_hac, _ = jalankan_hac(df2, jumlah_cluster=3, output_dendrogram_path=dendro)
    df2["cluster_hac"] = label_hac
    buat_plot_sebaran_cluster(df2, "cluster_hac", sebaran_hac, "Sebaran Cluster HAC (Smoke Test)")

    label_km, medoids, _ = jalankan_kmedoids(df2, k=3, max_iter=10)
    df2["cluster_kmedoids"] = label_km
    df2["is_medoid"] = df2.index.isin(medoids)
    buat_plot_sebaran_cluster(df2, "cluster_kmedoids", sebaran_km, "Sebaran Cluster K-Medoids (Smoke Test)")

    print("pre_rows:", df2.shape[0])
    print("ringkasan:", ring.ke_json())
    print("medoids:", medoids)
    print("hac_cluster_sizes:")
    print(bangun_tabel_karakteristik_cluster(df2, "cluster_hac")[["cluster", "jumlah_anggota"]].to_string(index=False))
    print("km_cluster_sizes:")
    print(
        bangun_tabel_karakteristik_cluster(df2, "cluster_kmedoids")[["cluster", "jumlah_anggota"]].to_string(
            index=False
        )
    )
    print("metrik_hac:", ringkasan_kualitas_sederhana(df2, "cluster_hac"))
    print("metrik_km:", ringkasan_kualitas_sederhana(df2, "cluster_kmedoids"))
    print("log_tail:", log[-3:])
    print("ok")


if __name__ == "__main__":
    main()

