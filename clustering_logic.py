import json
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import cdist
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
matplotlib.use("Agg")
KolomWajib = [
    "RETAILER NAME (R1)",
    "JENIS PESTISIDA",
    "PACKAGING",
    "VOLUME (L/KG)",
]

KolomKDD = [
    "YEAR",
    "MONTH",
    "DISTRIBUTOR",
    "AREA",
    "RETAILER NAME (R1)",
    "ADDRESS R1",
    "REGENCY/DISTRICT",
    "PRODUCT",
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
def _norm_teks(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip().lower()


def _parse_angka_pertama(value: object) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    s = str(value).strip().replace(",", ".")
    m = pd.Series([s]).str.extract(r"(-?\d+(?:\.\d+)?)", expand=False).iloc[0]
    if m is None or (isinstance(m, float) and math.isnan(m)):
        return float("nan")
    try:
        return float(m)
    except Exception:
        return float("nan")


def _pastikan_kolom(df: pd.DataFrame) -> pd.DataFrame:
    rename = {c: str(c).strip() for c in df.columns}
    df = df.rename(columns=rename)
    hilang = [c for c in KolomWajib if c not in df.columns]
    if hilang:
        raise ValueError(f"Kolom CSV tidak lengkap. Hilang: {', '.join(hilang)}")
    return df


def _minmax(series: pd.Series) -> Tuple[pd.Series, float, float]:
    x = series.astype(float)
    x_min = float(np.nanmin(x.values))
    x_max = float(np.nanmax(x.values))
    if x_max == x_min:
        return x.apply(lambda _: 0.0), x_min, x_max
    return (x - x_min) / (x_max - x_min), x_min, x_max


def _label_encode(series: pd.Series) -> Tuple[pd.Series, Dict[str, int]]:
    nilai = series.astype(str).fillna("").map(lambda v: str(v).strip())
    uniq = sorted(set(nilai.tolist()))
    peta = {v: i + 1 for i, v in enumerate(uniq)}
    return nilai.map(peta).astype(int), peta


@dataclass
class RingkasanPrapemrosesan:
    jumlah_awal: int
    jumlah_setelah_bersih: int
    jumlah_dihapus_na: int
    jumlah_dihapus_duplikat: int
    jumlah_missing_diisi: int
    peta_jenis_pestisida: Dict[str, int]
    volume_min: float
    volume_max: float
    volume_norm_min: float
    volume_norm_max: float
    kolom_dibersihkan: List[str]
    kolom_baru: List[str]
    jumlah_unik_distributor: int
    jumlah_unik_area: int
    jumlah_unik_regency: int
    jumlah_unik_product: int

    def ke_json(self) -> str:
        return json.dumps(
            {
                "jumlah_awal": self.jumlah_awal,
                "jumlah_setelah_bersih": self.jumlah_setelah_bersih,
                "jumlah_dihapus_na": self.jumlah_dihapus_na,
                "jumlah_dihapus_duplikat": self.jumlah_dihapus_duplikat,
                "jumlah_missing_diisi": self.jumlah_missing_diisi,
                "peta_jenis_pestisida": self.peta_jenis_pestisida,
                "volume_min": self.volume_min,
                "volume_max": self.volume_max,
                "volume_norm_min": self.volume_norm_min,
                "volume_norm_max": self.volume_norm_max,
                "kolom_dibersihkan": self.kolom_dibersihkan,
                "kolom_baru": self.kolom_baru,
                "jumlah_unik_distributor": self.jumlah_unik_distributor,
                "jumlah_unik_area": self.jumlah_unik_area,
                "jumlah_unik_regency": self.jumlah_unik_regency,
                "jumlah_unik_product": self.jumlah_unik_product,
            },
            ensure_ascii=False,
        )


def prapemrosesan_dataset(
    input_csv_path: str,
    output_csv_path: str,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Tuple[pd.DataFrame, RingkasanPrapemrosesan]:
    if progress_cb:
        progress_cb("Membaca file CSV...")
    df = pd.read_csv(input_csv_path)
    df = _pastikan_kolom(df)

    jumlah_awal = int(df.shape[0])

    if progress_cb:
        progress_cb("Membersihkan missing values pada kolom KDD yang tersedia...")
    kolom_dibersihkan = [c for c in KolomKDD if c in df.columns]
    jumlah_missing_diisi = 0
    placeholder = "TIDAK DIKETAHUI"
    for col in kolom_dibersihkan:
        if col in {"YEAR", "MONTH", "VOLUME (L/KG)"}:
            continue
        if df[col].isna().any():
            jumlah_missing_diisi += int(df[col].isna().sum())
            df[col] = df[col].fillna(placeholder)
    jumlah_dihapus_na = 0

    if progress_cb:
        progress_cb("Menghapus data duplikat...")
    sebelum_dup = int(df.shape[0])
    df = df.drop_duplicates()
    jumlah_dihapus_duplikat = sebelum_dup - int(df.shape[0])

    if progress_cb:
        progress_cb("Mengoreksi tipe data numerik (YEAR, MONTH, VOLUME)...")
    if "YEAR" in df.columns:
        df["YEAR"] = df["YEAR"].map(_parse_angka_pertama)
    if "MONTH" in df.columns:
        df["MONTH"] = df["MONTH"].map(_parse_angka_pertama)
    df["VOLUME (L/KG)"] = df["VOLUME (L/KG)"].map(_parse_angka_pertama)
    sebelum_na2 = int(df.shape[0])
    drop_numeric = ["VOLUME (L/KG)"]
    if "YEAR" in df.columns:
        drop_numeric.append("YEAR")
    if "MONTH" in df.columns:
        drop_numeric.append("MONTH")
    df = df.dropna(subset=drop_numeric)
    jumlah_dihapus_na += sebelum_na2 - int(df.shape[0])
    if df.empty:
        raise ValueError("Seluruh data terbuang setelah pembersihan missing value dan koreksi tipe data.")

    if progress_cb:
        progress_cb("Menormalkan teks kolom kategorikal (trim)...")
    for col in ["DISTRIBUTOR", "AREA", "REGENCY/DISTRICT", "PRODUCT", "RETAILER NAME (R1)", "ADDRESS R1", "JENIS PESTISIDA", "PACKAGING"]:
        if col in df.columns:
            df[col] = df[col].astype(str).map(lambda v: str(v).strip())

    if progress_cb:
        progress_cb("Melakukan encoding JENIS PESTISIDA sesuai pedoman penelitian...")
    jenis_norm = df["JENIS PESTISIDA"].apply(_norm_teks)
    kode_jenis = []
    peta_dinamis = dict(PETA_JENIS_PESTISIDA)
    kode_berikutnya = max(peta_dinamis.values(), default=0) + 1
    for raw, norm in zip(df["JENIS PESTISIDA"].astype(str).tolist(), jenis_norm.tolist()):
        if norm in peta_dinamis:
            kode_jenis.append(peta_dinamis[norm])
        else:
            peta_dinamis[norm] = kode_berikutnya
            kode_jenis.append(kode_berikutnya)
            kode_berikutnya += 1
    df["jenis_pestisida_kode"] = kode_jenis

    if progress_cb:
        progress_cb("Melakukan encoding label PACKAGING menjadi kode angka...")
    packaging_str = df["PACKAGING"].astype(str).fillna("")
    df["packaging_kode_angka"], _ = _label_encode(packaging_str)

    if progress_cb:
        progress_cb("Melakukan encoding label untuk kolom kategorikal lainnya...")
    jumlah_unik_distributor = 0
    jumlah_unik_area = 0
    jumlah_unik_regency = 0
    jumlah_unik_product = 0
    if "DISTRIBUTOR" in df.columns:
        df["distributor_kode"], peta_distributor = _label_encode(df["DISTRIBUTOR"])
        jumlah_unik_distributor = len(peta_distributor)
    if "AREA" in df.columns:
        df["area_kode"], peta_area = _label_encode(df["AREA"])
        jumlah_unik_area = len(peta_area)
    if "REGENCY/DISTRICT" in df.columns:
        df["regency_kode"], peta_reg = _label_encode(df["REGENCY/DISTRICT"])
        jumlah_unik_regency = len(peta_reg)
    if "PRODUCT" in df.columns:
        df["product_kode"], peta_prod = _label_encode(df["PRODUCT"])
        jumlah_unik_product = len(peta_prod)

    if progress_cb:
        progress_cb("Melakukan Min-Max Normalization untuk kolom volume...")
    df["volume_normalisasi"], volume_min, volume_max = _minmax(df["VOLUME (L/KG)"])
    volume_norm_min = float(np.nanmin(df["volume_normalisasi"].values))
    volume_norm_max = float(np.nanmax(df["volume_normalisasi"].values))

    df = df.reset_index(drop=True)
    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    df.to_csv(output_csv_path, index=False)

    kolom_baru = [
        c
        for c in [
            "jenis_pestisida_kode",
            "packaging_kode_angka",
            "distributor_kode",
            "area_kode",
            "regency_kode",
            "product_kode",
            "volume_normalisasi",
        ]
        if c in df.columns
    ]
    ringkasan = RingkasanPrapemrosesan(
        jumlah_awal=jumlah_awal,
        jumlah_setelah_bersih=int(df.shape[0]),
        jumlah_dihapus_na=int(jumlah_dihapus_na),
        jumlah_dihapus_duplikat=int(jumlah_dihapus_duplikat),
        jumlah_missing_diisi=int(jumlah_missing_diisi),
        peta_jenis_pestisida=peta_dinamis,
        volume_min=float(volume_min),
        volume_max=float(volume_max),
        volume_norm_min=float(volume_norm_min),
        volume_norm_max=float(volume_norm_max),
        kolom_dibersihkan=kolom_dibersihkan,
        kolom_baru=kolom_baru,
        jumlah_unik_distributor=int(jumlah_unik_distributor),
        jumlah_unik_area=int(jumlah_unik_area),
        jumlah_unik_regency=int(jumlah_unik_regency),
        jumlah_unik_product=int(jumlah_unik_product),
    )
    if progress_cb:
        progress_cb("Prapemrosesan selesai.")
    return df, ringkasan


def _ambil_fitur(df: pd.DataFrame) -> np.ndarray:
    return df[["jenis_pestisida_kode", "packaging_kode_angka", "volume_normalisasi"]].astype(float).values


def jalankan_hac(
    df: pd.DataFrame,
    jumlah_cluster: int,
    output_dendrogram_path: str,
    progress_cb: Optional[Callable[[str], None]] = None,
    truncate_p: int = 50,
) -> Tuple[np.ndarray, str]:
    if progress_cb:
        progress_cb("Menyiapkan fitur untuk HAC...")
    X = _ambil_fitur(df)

    if progress_cb:
        progress_cb("Menghitung linkage (Single Linkage, Euclidean Distance)...")
    Z = linkage(X, method="single", metric="euclidean")

    if progress_cb:
        progress_cb("Membentuk label cluster dari dendrogram...")
    label = fcluster(Z, t=int(jumlah_cluster), criterion="maxclust")

    if progress_cb:
        progress_cb("Membuat visualisasi dendrogram...")
    os.makedirs(os.path.dirname(output_dendrogram_path) or ".", exist_ok=True)
    plt.figure(figsize=(14, 7))
    dendrogram(
        Z,
        truncate_mode="lastp",
        p=int(truncate_p),
        show_leaf_counts=True,
        leaf_rotation=90,
        leaf_font_size=10,
        color_threshold=None,
    )
    plt.title("Dendrogram HAC (Single Linkage)")
    plt.xlabel("Ringkasan Cluster (Truncated)")
    plt.ylabel("Jarak (Euclidean)")
    plt.tight_layout()
    plt.savefig(output_dendrogram_path, dpi=160)
    plt.close()

    if progress_cb:
        progress_cb("HAC selesai.")
    return label.astype(int), json.dumps({"jumlah_cluster": int(jumlah_cluster)}, ensure_ascii=False)


def jalankan_kmedoids(
    df: pd.DataFrame,
    k: int,
    max_iter: int = 30,
    random_state: int = 42,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Tuple[np.ndarray, List[int], str]:
    X = _ambil_fitur(df)
    n = int(X.shape[0])
    k = int(k)
    if k < 2 or k >= n:
        raise ValueError("Nilai k harus minimal 2 dan lebih kecil dari jumlah data.")

    if progress_cb:
        progress_cb("Sedang menghitung matriks jarak Euclidean...")
    D = cdist(X, X, metric="euclidean")

    rng = np.random.RandomState(int(random_state))

    if progress_cb:
        progress_cb("Inisialisasi medoid awal...")
    medoids = _inisialisasi_kmedoids_pp(D, k, rng)

    if progress_cb:
        progress_cb("Memulai iterasi pertukaran medoid (PAM)...")

    for it in range(1, int(max_iter) + 1):
        if progress_cb:
            progress_cb(f"Iterasi {it}: Menghitung assignment cluster...")
        label, biaya, nearest_idx, d_nearest, d_second = _assign_dan_biaya(D, medoids)

        if progress_cb:
            progress_cb(f"Iterasi {it}: Sedang melakukan pertukaran medoid...")
        terbaik = None
        terbaik_biaya = biaya

        medoid_set = set(medoids.tolist())
        kandidat = [i for i in range(n) if i not in medoid_set]

        for pos in range(k):
            for h in kandidat:
                biaya_baru = _biaya_swap(D, medoids, pos, h, nearest_idx, d_nearest, d_second)
                if biaya_baru < terbaik_biaya:
                    terbaik_biaya = biaya_baru
                    terbaik = (pos, h)

        if terbaik is None:
            if progress_cb:
                progress_cb(f"Iterasi {it}: Tidak ada pertukaran yang memperbaiki biaya. Berhenti.")
            break

        perbaikan = float(biaya - terbaik_biaya)
        if perbaikan <= 1e-12:
            if progress_cb:
                progress_cb(f"Iterasi {it}: Perbaikan biaya sangat kecil. Berhenti.")
            break

        pos, h = terbaik
        if progress_cb:
            progress_cb(f"Iterasi {it}: Pertukaran terbaik ditemukan (perbaikan {perbaikan:.6f}).")
        medoids[pos] = int(h)

    label, biaya, _, _, _ = _assign_dan_biaya(D, medoids)
    ringkas = {
        "k": int(k),
        "max_iter": int(max_iter),
        "random_state": int(random_state),
        "total_biaya": float(biaya),
        "medoid_index": [int(i) for i in medoids.tolist()],
    }
    if progress_cb:
        progress_cb("K-Medoids selesai.")
    return label.astype(int), [int(i) for i in medoids.tolist()], json.dumps(ringkas, ensure_ascii=False)


def _inisialisasi_kmedoids_pp(D: np.ndarray, k: int, rng: np.random.RandomState) -> np.ndarray:
    n = int(D.shape[0])
    medoids = [int(rng.randint(0, n))]
    while len(medoids) < k:
        dist_ke_terdekat = np.min(D[:, medoids], axis=1)
        dist_ke_terdekat[medoids] = 0.0
        prob = dist_ke_terdekat ** 2
        total = float(np.sum(prob))
        if total <= 0:
            kandidat = [i for i in range(n) if i not in set(medoids)]
            medoids.append(int(rng.choice(kandidat)))
            continue
        prob = prob / total
        medoids.append(int(rng.choice(np.arange(n), p=prob)))
        medoids = list(dict.fromkeys(medoids))
    return np.array(medoids[:k], dtype=int)


def _assign_dan_biaya(
    D: np.ndarray, medoids: np.ndarray
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    dist = D[:, medoids]
    nearest_pos = np.argmin(dist, axis=1).astype(int)
    d_nearest = dist[np.arange(dist.shape[0]), nearest_pos]
    dist_sorted = np.sort(dist, axis=1)
    d_second = dist_sorted[:, 1] if dist_sorted.shape[1] > 1 else dist_sorted[:, 0]
    label = nearest_pos + 1
    biaya = float(np.sum(d_nearest))
    return label, biaya, nearest_pos, d_nearest, d_second


def _biaya_swap(
    D: np.ndarray,
    medoids: np.ndarray,
    pos_diganti: int,
    kandidat: int,
    nearest_pos: np.ndarray,
    d_nearest: np.ndarray,
    d_second: np.ndarray,
) -> float:
    best_other = np.where(nearest_pos == int(pos_diganti), d_second, d_nearest)
    d_kandidat = D[:, int(kandidat)]
    d_baru = np.minimum(best_other, d_kandidat)
    return float(np.sum(d_baru))


def bangun_tabel_karakteristik_cluster(
    df: pd.DataFrame, kolom_label: str
) -> pd.DataFrame:
    if kolom_label not in df.columns:
        raise ValueError(f"Kolom label cluster tidak ditemukan: {kolom_label}")

    def modus(series: pd.Series) -> str:
        if series.empty:
            return "-"
        return str(series.mode(dropna=True).iloc[0]) if not series.mode(dropna=True).empty else "-"

    ringkas = (
        df.groupby(kolom_label)
        .agg(
            jumlah_anggota=("RETAILER NAME (R1)", "count"),
            jenis_dominan=("JENIS PESTISIDA", modus),
            packaging_dominan=("PACKAGING", modus),
            volume_rata2=("VOLUME (L/KG)", "mean"),
            volume_min=("VOLUME (L/KG)", "min"),
            volume_maks=("VOLUME (L/KG)", "max"),
        )
        .reset_index()
        .rename(columns={kolom_label: "cluster"})
    )
    q1 = float(df["VOLUME (L/KG)"].quantile(0.33))
    q2 = float(df["VOLUME (L/KG)"].quantile(0.66))

    def label_volume(v: float) -> str:
        if v <= q1:
            return "volume rendah"
        if v <= q2:
            return "volume sedang"
        return "volume tinggi"

    ringkas["interpretasi"] = ringkas.apply(
        lambda r: f"Cluster {int(r['cluster'])} didominasi {r['jenis_dominan']} dengan {label_volume(float(r['volume_rata2']))}.",
        axis=1,
    )
    ringkas["volume_rata2"] = ringkas["volume_rata2"].round(4)
    ringkas["volume_min"] = ringkas["volume_min"].round(4)
    ringkas["volume_maks"] = ringkas["volume_maks"].round(4)
    return ringkas.sort_values("cluster").reset_index(drop=True)


def buat_plot_sebaran_cluster(
    df: pd.DataFrame,
    kolom_label: str,
    output_path: str,
    judul: str,
) -> str:
    if kolom_label not in df.columns:
        raise ValueError(f"Kolom label cluster tidak ditemukan: {kolom_label}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    x = df["volume_normalisasi"].astype(float).values
    y = df["jenis_pestisida_kode"].astype(float).values
    label = df[kolom_label].astype(int).values

    plt.figure(figsize=(12, 6))
    plt.scatter(x, y, c=label, cmap="tab20", s=18, alpha=0.85)
    handles: List[Line2D] = []
    if "is_medoid" in df.columns:
        is_medoid = df["is_medoid"].astype(bool).values
        if bool(np.any(is_medoid)):
            plt.scatter(
                x[is_medoid],
                y[is_medoid],
                s=90,
                marker="X",
                c="#ef4444",
                edgecolors="#ffffff",
                linewidths=0.8,
                alpha=0.95,
            )
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="X",
                    color="w",
                    markerfacecolor="#ef4444",
                    markeredgecolor="#ffffff",
                    markersize=10,
                    label="Medoid (K-Medoids)",
                )
            )
    plt.title(judul)
    plt.xlabel("Volume (Normalisasi Min–Maks)")
    plt.ylabel("Kode Jenis Pestisida")
    if handles:
        plt.legend(handles=handles, loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def buat_plot_ukuran_cluster_batang(
    df: pd.DataFrame, kolom_label: str, output_path: str, judul: str
) -> str:
    if kolom_label not in df.columns:
        raise ValueError(f"Kolom label cluster tidak ditemukan: {kolom_label}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ukuran = df[kolom_label].astype(int).value_counts().sort_index()
    x = ukuran.index.astype(int).tolist()
    y = ukuran.values.astype(int).tolist()
    warna = [plt.get_cmap("tab20")((i - 1) % 20) for i in x]
    plt.figure(figsize=(10, 4.8))
    bars = plt.bar([str(i) for i in x], y, color=warna, edgecolor="#0f172a", linewidth=0.4)
    for b, v in zip(bars, y):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height(), str(v), ha="center", va="bottom", fontsize=9)
    plt.title(judul)
    plt.xlabel("Cluster")
    plt.ylabel("Jumlah anggota")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def buat_plot_komposisi_jenis_stacked(
    df: pd.DataFrame, kolom_label: str, output_path: str, judul: str
) -> str:
    if kolom_label not in df.columns:
        raise ValueError(f"Kolom label cluster tidak ditemukan: {kolom_label}")
    if "JENIS PESTISIDA" not in df.columns:
        raise ValueError("Kolom JENIS PESTISIDA tidak ditemukan.")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tab = pd.crosstab(df[kolom_label].astype(int), df["JENIS PESTISIDA"].astype(str))
    tab = tab.sort_index()
    jenis = tab.columns.tolist()
    warna_map = {
        "insektisida": "#60a5fa",
        "herbisida": "#a78bfa",
        "fungisida": "#34d399",
        "zat pengatur tumbuh": "#fbbf24",
    }
    colors = []
    for j in jenis:
        k = _norm_teks(j)
        colors.append(warna_map.get(k, "#94a3b8"))
    bottom = np.zeros(tab.shape[0], dtype=float)
    plt.figure(figsize=(10.5, 5.2))
    for i, j in enumerate(jenis):
        vals = tab[j].values.astype(float)
        plt.bar([str(x) for x in tab.index.tolist()], vals, bottom=bottom, color=colors[i], label=str(j))
        bottom += vals
    plt.title(judul)
    plt.xlabel("Cluster")
    plt.ylabel("Jumlah produk/record")
    plt.legend(loc="best", fontsize=8, frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def buat_plot_boxplot_volume(
    df: pd.DataFrame, kolom_label: str, output_path: str, judul: str
) -> str:
    if kolom_label not in df.columns:
        raise ValueError(f"Kolom label cluster tidak ditemukan: {kolom_label}")
    if "VOLUME (L/KG)" not in df.columns:
        raise ValueError("Kolom VOLUME (L/KG) tidak ditemukan.")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    data = []
    labels = []
    for c in sorted(df[kolom_label].astype(int).unique().tolist()):
        vals = df.loc[df[kolom_label].astype(int) == int(c), "VOLUME (L/KG)"].astype(float).values
        if vals.size == 0:
            continue
        data.append(vals)
        labels.append(str(int(c)))
    plt.figure(figsize=(10.5, 5.0))
    plt.boxplot(
        data,
        labels=labels,
        showmeans=True,
        meanline=True,
        medianprops={"color": "#f43f5e", "linewidth": 1.2},
        meanprops={"color": "#22c55e", "linewidth": 1.2},
        whiskerprops={"color": "#cbd5e1"},
        capprops={"color": "#cbd5e1"},
        boxprops={"color": "#cbd5e1"},
        flierprops={"marker": "o", "markerfacecolor": "#94a3b8", "markeredgecolor": "none", "markersize": 3, "alpha": 0.6},
    )
    plt.title(judul)
    plt.xlabel("Cluster")
    plt.ylabel("Volume (L/KG)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def ringkasan_kualitas_sederhana(df: pd.DataFrame, kolom_label: str) -> Dict[str, float]:
    if kolom_label not in df.columns:
        raise ValueError(f"Kolom label cluster tidak ditemukan: {kolom_label}")
    X = _ambil_fitur(df)
    label = df[kolom_label].astype(int).values
    cluster_unik = np.unique(label)
    ukuran = np.array([np.sum(label == c) for c in cluster_unik], dtype=float)
    centroid = np.vstack([X[label == c].mean(axis=0) for c in cluster_unik])
    idx = {c: i for i, c in enumerate(cluster_unik.tolist())}
    pusat = np.vstack([centroid[idx[c]] for c in label])
    disp = np.linalg.norm(X - pusat, axis=1)

    sil = silhouette_score(X, label) if len(cluster_unik) > 1 else -1.0
    dbi = davies_bouldin_score(X, label) if len(cluster_unik) > 1 else 999.0
    chi = calinski_harabasz_score(X, label) if len(cluster_unik) > 1 else 0.0

    return {
        "jumlah_cluster": float(len(cluster_unik)),
        "rata2_dispersi": float(np.mean(disp)),
        "std_ukuran_cluster": float(np.std(ukuran)),
        "min_ukuran_cluster": float(np.min(ukuran)),
        "maks_ukuran_cluster": float(np.max(ukuran)),
        "silhouette": float(sil),
        "dbi": float(dbi),
        "chi": float(chi),
    }
