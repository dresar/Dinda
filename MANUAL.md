# Panduan Penggunaan DINDA Analytics (MANUAL)

Dokumen ini adalah panduan lengkap cara menggunakan sistem DINDA Analytics. Kami akan menjelaskan setiap halaman, formulir, tabel, dan bagaimana menafsirkan hasil analisis Anda.

---

## 📋 Alur Kerja Utama

Proses analisis di sistem ini mengikuti 4 tahap utama yang berurutan:
1.  **[Unggah Data](#1-unggah-data)**: Memasukkan data mentah.
2.  **[Prapemrosesan](#2-prapemrosesan-preprocessing)**: Membersihkan dan menyiapkan data.
3.  **[Clustering](#3-clustering-pengelompokan)**: Menjalankan algoritma pengelompokan.
4.  **[Hasil Analisis](#4-hasil-analisis)**: Melihat visualisasi dan evaluasi klaster.

---

## 1. Unggah Data

Halaman ini adalah pintu masuk analisis Anda.

### Formulir Unggah
*   **Pilih File**: Klik tombol "Choose File" untuk memilih file dari komputer Anda.
*   **Format**: Hanya menerima file **CSV (.csv)**.
*   **Kolom Wajib**: Pastikan file CSV Anda memiliki kolom-kolom berikut (nama kolom harus persis/mirip):
    *   `RETAILER`: Nama toko/pengecer.
    *   `JENIS`: Kategori produk (misal: HERBISIDA, INSEKTISIDA).
    *   `PACKAGING`: Kemasan (misal: BOTOL, KALENG, DRUM).
    *   `VOLUME`: Volume produk (angka, misal: 100, 500, 1000).
    *   `DISTRIBUTOR_KODE`: Kode unik distributor.
    *   `AREA_KODE`: Kode area pemasaran.
    *   `REGENCY_KODE`: Kode kabupaten/kota.
    *   `PRODUCT_KODE`: Kode produk.

**Tips**: Data yang tidak lengkap akan dibersihkan di tahap berikutnya, namun struktur kolom harus sesuai.

---

## 2. Prapemrosesan (Preprocessing)

Setelah mengunggah, Anda akan diarahkan ke halaman Prapemrosesan. Di sini data mentah diubah menjadi data siap analisis.

### Proses Otomatis
Sistem melakukan langkah-langkah berikut secara otomatis saat Anda klik **"Jalankan Prapemrosesan"**:
1.  **Pembersihan Data (Cleaning)**: Menghapus baris yang memiliki nilai kosong (NaN) pada kolom kritis (VOLUME).
2.  **Pengisian Nilai Kosong**: Mengisi nilai kosong pada kolom kategorikal dengan label "TIDAK DIKETAHUI".
3.  **Encoding Kategorikal**: Mengubah teks menjadi angka (misal: "HERBISIDA" -> 1, "INSEKTISIDA" -> 2).
4.  **Normalisasi Volume**: Mengubah nilai volume ke skala 0-1 (Min-Max Scaling) agar adil saat dibandingkan dengan fitur lain.

### Tabel Preview
*   **Preview Data Mentah**: Menampilkan 10 baris pertama data asli yang Anda unggah.
*   **Preview Data Setelah Prapemrosesan**: Menampilkan data hasil transformasi. Perhatikan kolom baru seperti `volume_normalisasi`, `jenis_pestisida_kode`, dll.

### Ringkasan Statistik
Tabel ringkasan di sebelah kanan menunjukkan:
*   Jumlah baris awal vs akhir.
*   Jumlah data yang dihapus (karena duplikat atau error).
*   Rentang nilai volume (Min-Max).

---

## 3. Clustering (Pengelompokan)

Di halaman ini, Anda mengatur parameter algoritma.

### Formulir Parameter
Anda bisa mengatur parameter untuk dua algoritma sekaligus:

#### A. Parameter HAC (Hierarchical Agglomerative Clustering)
*   **Jumlah Cluster (k)**: Berapa banyak kelompok yang ingin Anda bentuk?
    *   *Saran*: Mulai dengan **3 atau 4**. Angka ini bisa dilihat dari Dendrogram nanti.
    *   *Default*: 4.

#### B. Parameter K-Medoids
*   **Jumlah Cluster (k)**: Sama seperti HAC, tentukan jumlah kelompok target.
    *   *Saran*: Samakan dengan HAC untuk perbandingan yang adil.
    *   *Default*: 4.
*   **Max Iterations**: Batas maksimum pengulangan algoritma untuk mencari pusat klaster terbaik.
    *   *Default*: 25 (Sudah cukup untuk sebagian besar kasus).

#### C. Opsi Eksekusi
*   **Jalankan**: Pilih algoritma mana yang ingin dijalankan.
    *   *Keduanya (Recommended)*: Menjalankan HAC dan K-Medoids untuk perbandingan.
    *   *Hanya HAC* atau *Hanya K-Medoids*.

**Tombol "Mulai Proses Clustering"**: Klik untuk memulai. Proses ini bisa memakan waktu beberapa detik hingga menit tergantung ukuran data.

---

## 4. Hasil Analisis

Halaman ini menampilkan output lengkap dari proses clustering.

### A. Metrik Evaluasi (Tabel Skor)
Tabel ini membandingkan kualitas kedua algoritma:
1.  **Silhouette Score**: Rentang -1 sampai 1. Semakin **tinggi** (mendekati 1), semakin baik pemisahan antar klaster.
2.  **DBI (Davies-Bouldin Index)**: Semakin **rendah** nilainya, semakin baik (klaster padat dan terpisah jauh).
3.  **CHI (Calinski-Harabasz Index)**: Semakin **tinggi**, semakin baik definisi klasternya.

### B. Visualisasi (Grafik)
*   **Dendrogram (HAC Only)**: Diagram pohon yang menunjukkan hierarki penggabungan data. Potong garis vertikal terpanjang untuk menentukan jumlah klaster optimal.
*   **Scatter Plot (PCA)**: Peta sebaran data dalam 2 dimensi. Titik dengan warna sama berarti satu kelompok.
*   **Bar Chart Ukuran Cluster**: Menunjukkan jumlah anggota (member) di setiap klaster. Apakah seimbang atau timpang?
*   **Stacked Bar Komposisi Jenis**: Menunjukkan proporsi jenis produk (misal: Herbisida vs Insektisida) dalam setiap klaster.
*   **Boxplot Volume**: Menunjukkan distribusi volume penjualan di setiap klaster. Apakah ada klaster khusus volume besar/kecil?

### C. Tabel Karakteristik Cluster
Tabel detail yang menjelaskan "profil" setiap klaster.
*   **Kolom Rata-rata**: Menunjukkan nilai rata-rata fitur numerik di klaster tersebut.
*   **Dominan**: Menunjukkan kategori yang paling sering muncul (modus).
    *   *Contoh*: Jika Cluster 0 memiliki "Dominan Jenis: HERBISIDA", berarti mayoritas anggota Cluster 0 adalah produk Herbisida.

### D. Fitur AI (Chat & Ringkasan)
*   **Ringkasan Otomatis**: Klik tombol untuk meminta AI membuatkan laporan singkat tentang hasil analisis.
*   **Chatbot**: Ketik pertanyaan Anda di kolom chat.
    *   *Contoh Pertanyaan*:
        *   "Jelaskan perbedaan Cluster 0 dan Cluster 1 pada metode HAC."
        *   "Mana metode yang lebih baik berdasarkan Silhouette Score?"
        *   "Apa strategi pemasaran untuk Cluster dengan volume penjualan rendah?"

---

## 5. Reset Data (Hapus Sesi)
Jika ingin memulai analisis baru dengan data berbeda:
1.  Klik menu **"Reset Data"** di navigasi atas (biasanya ikon sampah atau menu dropdown).
2.  Tindakan ini akan **menghapus semua file**, database sementara, dan history chat.
3.  Anda akan kembali ke halaman Dashboard bersih.

---

## Tips Tambahan
*   **Reload Halaman**: Jika proses terasa macet, coba refresh halaman. Sistem memiliki fitur *auto-recovery* untuk melanjutkan proses yang terputus.
*   **Cek Log**: Di bagian bawah setiap halaman proses, terdapat kotak "Log Proses" yang mencatat setiap langkah sistem. Cek di sini jika ada error.

Selamat menganalisis! 🚀
