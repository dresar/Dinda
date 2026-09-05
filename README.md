# DINDA Analytics - Sistem Clustering Data Pestisida

Selamat datang di dokumentasi resmi **DINDA Analytics**. Sistem ini dirancang untuk melakukan analisis pengelompokan (clustering) pada data distribusi pestisida menggunakan algoritma **Hierarchical Agglomerative Clustering (HAC)** dan **K-Medoids**.

Sistem ini membantu analis data untuk memahami pola distribusi, karakteristik kelompok (cluster), dan memberikan wawasan strategis berbasis data.

---

## 📚 Daftar Isi Dokumentasi

Dokumentasi ini dibagi menjadi 3 bagian utama untuk memudahkan Anda:

1.  **[README.md](README.md)** (File ini): Pengenalan sistem, fitur utama, dan teknologi yang digunakan.
2.  **[INSTALLATION.md](INSTALLATION.md)**: Panduan langkah demi langkah untuk menginstal dan menjalankan aplikasi di Windows.
3.  **[MANUAL.md](MANUAL.md)**: Panduan penggunaan aplikasi, penjelasan detail setiap formulir, tabel, dan cara membaca hasil analisis.

---

## 🚀 Fitur Utama

*   **Upload Data CSV**: Mendukung format data CSV standar untuk analisis.
*   **Prapemrosesan Otomatis**:
    *   Pembersihan data (cleaning).
    *   Penanganan *missing values* (data kosong).
    *   Encoding variabel kategorikal (Label Encoding).
    *   Normalisasi data (Min-Max Scaling).
*   **Algoritma Clustering**:
    *   **HAC (Hierarchical Agglomerative Clustering)**: Metode Ward dengan visualisasi Dendrogram.
    *   **K-Medoids**: Algoritma partisi yang tangguh terhadap outlier (menggunakan PAM-like approach).
*   **Visualisasi Interaktif**:
    *   Dendrogram (Pohon klaster).
    *   Scatter Plot (Sebaran data 2D dengan PCA).
    *   Bar Chart (Ukuran klaster).
    *   Stacked Bar Chart (Komposisi jenis).
    *   Boxplot (Distribusi volume).
*   **Metrik Evaluasi**:
    *   Silhouette Score.
    *   Davies-Bouldin Index (DBI).
    *   Calinski-Harabasz Index (CHI).
*   **AI Assistant (Powered by Gemini)**:
    *   Pembuatan ringkasan eksekutif otomatis.
    *   Chatbot interaktif untuk bertanya seputar hasil analisis data Anda.

---

## 🛠️ Teknologi yang Digunakan

Sistem ini dibangun menggunakan teknologi modern yang handal:

*   **Backend**: Python 3.10+ dengan framework **Flask**.
*   **Database**: **SQLite** (Ringan, tanpa konfigurasi server rumit).
*   **Data Science**:
    *   **Pandas**: Manipulasi dan analisis data tabular.
    *   **Scikit-learn**: Algoritma Machine Learning (HAC, Metrik).
    *   **Scikit-learn-extra**: Algoritma K-Medoids.
    *   **Matplotlib & Seaborn**: Pembuatan visualisasi statis.
*   **Frontend**: HTML5, JavaScript, dan **Tailwind CSS** untuk antarmuka yang modern dan responsif.
*   **AI Integration**: **Google Gemini API** untuk analisis cerdas.

---

## 📋 Prasyarat Singkat

Sebelum menginstal, pastikan komputer Anda memiliki:
*   Sistem Operasi: Windows 10/11 (Disarankan).
*   Python: Versi 3.10 atau lebih baru.
*   Akses Internet: Untuk mengunduh library dan mengakses API Gemini.

👉 **Lanjut ke [Panduan Instalasi (INSTALLATION.md)](INSTALLATION.md) untuk mulai menggunakan sistem.**
