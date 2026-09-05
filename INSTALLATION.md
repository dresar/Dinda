# Panduan Instalasi DINDA Analytics di Windows

Dokumen ini menjelaskan langkah-langkah lengkap untuk menginstal dan menjalankan sistem DINDA Analytics pada sistem operasi Windows.

---

## 1. Persiapan Lingkungan (Prerequisites)

Sebelum memulai, pastikan Anda sudah memiliki hal-hal berikut:

### a. Python
Anda harus menginstal Python versi **3.10** atau lebih baru.
1.  Unduh Python dari website resmi: [python.org/downloads](https://www.python.org/downloads/).
2.  Saat instalasi, **PENTING**: Centang opsi **"Add Python to PATH"** sebelum klik "Install Now".
3.  Cek instalasi dengan membuka Command Prompt (CMD) atau PowerShell dan ketik:
    ```bash
    python --version
    ```

### b. Google Gemini API Key (Opsional tapi Disarankan)
Untuk menggunakan fitur AI (Ringkasan Otomatis dan Chatbot), Anda memerlukan API Key.
1.  Kunjungi [Google AI Studio](https://aistudio.google.com/).
2.  Buat API Key baru.
3.  Simpan kuncinya untuk nanti.

---

## 2. Instalasi Aplikasi

Ikuti langkah-langkah berikut di folder proyek DINDA yang sudah Anda unduh/ekstrak.

### Langkah 1: Buka Terminal
1.  Buka folder `DINDA` di File Explorer.
2.  Klik kanan di ruang kosong, pilih **"Open in Terminal"** (atau tekan `Shift + Klik Kanan` lalu pilih "Open PowerShell window here").

### Langkah 2: Buat Virtual Environment (Disarankan)
Virtual environment menjaga agar library proyek tidak bercampur dengan library sistem lain.
Ketik perintah berikut di terminal:
```powershell
python -m venv venv
```

### Langkah 3: Aktifkan Virtual Environment
Ketik perintah berikut:
```powershell
.\venv\Scripts\activate
```
*Jika berhasil, Anda akan melihat tulisan `(venv)` di awal baris terminal Anda.*

### Langkah 4: Instal Dependencies
Instal semua library yang dibutuhkan menggunakan `requirements.txt`:
```powershell
pip install -r requirements.txt
```
*Tunggu hingga proses download dan instalasi selesai.*

---

## 3. Konfigurasi Sistem

### Langkah 1: Setup File `.env`
1.  Cari file bernama `.env` di dalam folder proyek. Jika tidak ada, buat file baru bernama `.env`.
2.  Buka file tersebut dengan Notepad atau teks editor.
3.  Isi konfigurasi berikut:

```env
# Kunci rahasia untuk sesi Flask (Bisa diisi acak)
FLASK_SECRET_KEY=kunci_rahasia_super_aman_123

# Konfigurasi AI (Ubah ke "true" untuk mengaktifkan)
AI=true

# Google Gemini API Key (Wajib jika AI=true)
GOOGLE_API_KEY=masukkan_api_key_google_anda_disini

# Model Gemini (Biarkan default)
GEMINI_MODEL=gemini-1.5-flash
```
4.  Simpan file.

---

## 4. Menjalankan Aplikasi

Setelah semua siap, jalankan aplikasi dengan perintah:

```powershell
python app.py
```

Jika berhasil, Anda akan melihat output seperti ini:
```
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
```

### Akses Aplikasi
Buka browser (Chrome, Edge, atau Firefox) dan kunjungi alamat:
**[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 🚧 Masalah Umum (Troubleshooting)

**1. "python is not recognized..."**
*   Solusi: Python belum ditambahkan ke PATH. Instal ulang Python dan pastikan centang "Add Python to PATH".

**2. "ModuleNotFoundError: No module named..."**
*   Solusi: Anda mungkin lupa mengaktifkan virtual environment atau belum menjalankan `pip install -r requirements.txt`.

**3. Fitur AI tidak jalan**
*   Solusi: Cek file `.env`. Pastikan `AI=true` dan `GOOGLE_API_KEY` sudah diisi dengan benar. Restart aplikasi setelah mengubah file `.env`.

---

👉 **Setelah aplikasi berjalan, baca [Panduan Penggunaan (MANUAL.md)](MANUAL.md) untuk memahami cara menggunakan sistem.**
