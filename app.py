import json
import os
import sqlite3
import shutil
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from werkzeug.utils import secure_filename

from clustering_logic import (
    KolomWajib,
    bangun_tabel_karakteristik_cluster,
    buat_plot_boxplot_volume,
    buat_plot_komposisi_jenis_stacked,
    buat_plot_sebaran_cluster,
    buat_plot_ukuran_cluster_batang,
    jalankan_hac,
    jalankan_kmedoids,
    prapemrosesan_dataset,
    ringkasan_kualitas_sederhana,
)


import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
STATIC_GEN_DIR = os.path.join(BASE_DIR, "static", "generated")
DB_PATH = os.path.join(BASE_DIR, "app.db")
PER_PAGE_PREVIEW = 10
CHAT_HISTORY = {}


def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dinda-secret-key")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(STATIC_GEN_DIR, exist_ok=True)
    _init_db()

    @app.context_processor
    def inject_ctx() -> Dict[str, Any]:
        job_id = session.get("job_id")
        job = _get_job(job_id) if job_id else None
        ai_enabled = os.getenv("AI", "false").lower() == "true"
        return {"job_aktif": job, "ai_enabled": ai_enabled}

    @app.route("/")
    def dashboard():
        job_aktif = _get_job(session.get("job_id")) if session.get("job_id") else None
        ringkas = _dashboard_stats()
        return render_template("dashboard.html", job=job_aktif, ringkas=ringkas)

    @app.route("/dokumentasi")
    def dokumentasi():
        return render_template("dokumentasi.html")

    @app.route("/reset")
    def reset_data():
        try:
            # 1. Hapus Session User
            session.clear()
            
            # 2. Hapus File Fisik (Uploads)
            if os.path.exists(UPLOAD_DIR):
                shutil.rmtree(UPLOAD_DIR)
                os.makedirs(UPLOAD_DIR, exist_ok=True)
            
            # 3. Hapus File Fisik (Generated Images)
            if os.path.exists(STATIC_GEN_DIR):
                shutil.rmtree(STATIC_GEN_DIR)
                os.makedirs(STATIC_GEN_DIR, exist_ok=True)

            # 4. Reset Database (Hapus dan Re-init)
            # Tutup koneksi jika ada yang menggantung (biasanya handled by helpers)
            if os.path.exists(DB_PATH):
                try:
                    os.remove(DB_PATH)
                except PermissionError:
                    # Fallback: Truncate table jobs jika file terkunci
                    con = _conn()
                    con.execute("DELETE FROM jobs")
                    con.execute("DELETE FROM sqlite_sequence WHERE name='jobs'")
                    con.commit()
                    con.close()
            
            # Re-initialize DB (create tables if deleted)
            _init_db()

            # 5. Clear In-Memory Chat History
            CHAT_HISTORY.clear()
            
            flash("Semua data (Uploads, Hasil, History) berhasil dihapus total.", "success")
        except Exception as e:
            flash(f"Gagal mereset data: {str(e)}", "error")
            
        return redirect(url_for("dashboard"))

    @app.route("/api/generate_summary", methods=["POST"])
    def generate_summary():
        job_id = session.get("job_id")
        if not job_id:
            return jsonify({"error": "Job tidak ditemukan"}), 404
        
        # Gunakan API Key dan Model HANYA dari .env
        api_key = os.getenv("GOOGLE_API_KEY")
        model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        ai_enabled = os.getenv("AI", "false").lower() == "true"

        if not ai_enabled:
            return jsonify({"error": "Fitur AI dinonaktifkan."}), 403

        if not api_key:
            return jsonify({"error": "Konfigurasi server belum lengkap (API Key)"}), 500

        job = _get_job(job_id)
        if not job:
            return jsonify({"error": "Job tidak valid"}), 404

        # Kumpulkan data untuk prompt
        metrik = json.loads(job["metrics_json"]) if job.get("metrics_json") else {}
        
        karakter_hac = []
        if job.get("hac_result_path") and os.path.exists(job["hac_result_path"]):
            df_hac = pd.read_csv(job["hac_result_path"])
            karakter_hac = bangun_tabel_karakteristik_cluster(df_hac, "cluster_hac").to_dict(orient="records")

        karakter_km = []
        if job.get("kmedoids_result_path") and os.path.exists(job["kmedoids_result_path"]):
            df_km = pd.read_csv(job["kmedoids_result_path"])
            karakter_km = bangun_tabel_karakteristik_cluster(df_km, "cluster_kmedoids").to_dict(orient="records")

        # Bangun Prompt
        prompt = f"""
        Kamu adalah asisten data analyst yang santai tapi cerdas. Tugasmu adalah membuat ringkasan analisis perbandingan hasil clustering HAC (Hierarchical Agglomerative Clustering) vs K-Medoids.
        
        PENTING: Jawablah hanya dalam bentuk PARAGRAF BERNOMOR (1., 2., 3., dst). Jangan gunakan heading markdown (##) yang berlebihan. Gunakan bold (**) untuk menekankan poin penting.
        Bahasa Indonesia yang digunakan harus non-formal (santai, mudah dimengerti), namun tetap profesional.

        Data Metrik Kualitas (Semakin tinggi Silhouette/CHI semakin baik, semakin rendah DBI semakin baik):
        - HAC: {json.dumps(metrik.get('hac', {}), indent=2)}
        - K-Medoids: {json.dumps(metrik.get('kmedoids', {}), indent=2)}

        Karakteristik Cluster HAC (Rata-rata fitur per cluster):
        {json.dumps(karakter_hac, indent=2)}

        Karakteristik Cluster K-Medoids (Rata-rata fitur per cluster):
        {json.dumps(karakter_km, indent=2)}

        Tolong jelaskan dalam poin-poin berikut:
        1. **Perbandingan Performa**: Jelaskan mana yang lebih baik secara statistik (Silhouette, DBI, CHI) dan kenapa, dalam satu atau dua paragraf.
        2. **Karakteristik Cluster**: Ceritakan profil cluster yang terbentuk (misal: "Cluster A isinya toko kecil...", "Cluster B isinya...").
        3. **Rekomendasi**: Saran strategi bisnis singkat berdasarkan hasil clustering ini.

        Ingat: Format output harus rapi, font enak dibaca, gunakan paragraf yang mengalir.
        """

        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return jsonify({"summary": response.text})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/chat_ai", methods=["POST"])
    def chat_ai():
        job_id = session.get("job_id")
        if not job_id:
            return jsonify({"error": "Job tidak ditemukan"}), 404
        
        # Check AI enabled
        ai_enabled = os.getenv("AI", "false").lower() == "true"
        if not ai_enabled:
            return jsonify({"error": "Fitur AI dinonaktifkan."}), 403

        api_key = os.getenv("GOOGLE_API_KEY")
        model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        
        if not api_key:
             return jsonify({"error": "Konfigurasi server belum lengkap (API Key)"}), 500

        data = request.json
        user_message = data.get("message", "")
        if not user_message:
            return jsonify({"error": "Pesan tidak boleh kosong"}), 400

        job = _get_job(job_id)
        if not job:
            return jsonify({"error": "Job tidak valid"}), 404

        # --- Manajemen History Chat (Hapus setelah 5 menit inaktif) ---
        now = datetime.now()
        cleanup_threshold = timedelta(minutes=5)
        
        # Bersihkan history lama
        to_remove = []
        for jid, history in CHAT_HISTORY.items():
            if now - history['last_updated'] > cleanup_threshold:
                to_remove.append(jid)
        for jid in to_remove:
            del CHAT_HISTORY[jid]

        # Inisialisasi history jika belum ada untuk job ini
        if job_id not in CHAT_HISTORY:
            CHAT_HISTORY[job_id] = {'messages': [], 'last_updated': now}
        
        # Update timestamp terakhir akses
        CHAT_HISTORY[job_id]['last_updated'] = now

        # Gather context data (hanya dipakai jika history kosong/baru mulai)
        metrik = json.loads(job["metrics_json"]) if job.get("metrics_json") else {}
        
        # Preprocessing Context
        preprocess_summary = json.loads(job["preprocess_summary_json"]) if job.get("preprocess_summary_json") else {}
        
        # Columns info
        raw_cols = []
        pre_cols = []
        if job.get("uploaded_path") and os.path.exists(job["uploaded_path"]):
             try:
                 raw_cols = pd.read_csv(job["uploaded_path"], nrows=0).columns.tolist()
             except: pass
        if job.get("preprocessed_path") and os.path.exists(job["preprocessed_path"]):
             try:
                 pre_cols = pd.read_csv(job["preprocessed_path"], nrows=0).columns.tolist()
             except: pass

        karakter_hac = []
        if job.get("hac_result_path") and os.path.exists(job["hac_result_path"]):
            df_hac = pd.read_csv(job["hac_result_path"])
            karakter_hac = bangun_tabel_karakteristik_cluster(df_hac, "cluster_hac").to_dict(orient="records")

        karakter_km = []
        if job.get("kmedoids_result_path") and os.path.exists(job["kmedoids_result_path"]):
            df_km = pd.read_csv(job["kmedoids_result_path"])
            karakter_km = bangun_tabel_karakteristik_cluster(df_km, "cluster_kmedoids").to_dict(orient="records")

        # System Context
        context_prompt = f"""
        Kamu adalah asisten data analyst untuk proyek clustering pestisida (HAC vs K-Medoids).
        
        Konteks Data Project:
        1. Prapemrosesan:
           - Ringkasan: {json.dumps(preprocess_summary, indent=2)}
           - Kolom Data Mentah: {raw_cols}
           - Kolom Data Preprocessed: {pre_cols}
           - Catatan: Missing value pada kolom kategorikal telah diisi dengan placeholder 'TIDAK DIKETAHUI'. Data dengan missing value pada kolom numerik (YEAR, MONTH, VOLUME) telah dihapus.
        
        2. Hasil Clustering:
           - Metrik Kualitas (Silhouette, DBI, CHI): {json.dumps(metrik, indent=2)}
           - Karakteristik HAC: {json.dumps(karakter_hac, indent=2)}
           - Karakteristik K-Medoids: {json.dumps(karakter_km, indent=2)}
        
        Instruksi:
        Jawablah pertanyaan user berdasarkan data di atas dengan bahasa Indonesia yang santai, jelas, dan membantu.
        Jika pertanyaan teknis tentang metode, jelaskan dengan sederhana.
        Jika pertanyaan di luar konteks data ini, jawab dengan sopan bahwa kamu fokus pada hasil analisis ini.
        Jawab dalam format Markdown jika perlu (bold, list, dll).
        """

        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            
            # Siapkan history untuk Gemini
            stored_messages = CHAT_HISTORY[job_id]['messages']
            gemini_history = []
            for msg in stored_messages:
                gemini_history.append({
                    "role": msg["role"],
                    "parts": msg["parts"]
                })

            # Mulai chat session
            chat = model.start_chat(history=gemini_history)

            response = None
            if not stored_messages:
                # Jika chat baru, sertakan konteks di pesan pertama
                full_message = context_prompt + f"\n\nUser bertanya: {user_message}"
                response = chat.send_message(full_message)
                
                # Simpan ke history server
                CHAT_HISTORY[job_id]['messages'].append({'role': 'user', 'parts': [full_message]})
                CHAT_HISTORY[job_id]['messages'].append({'role': 'model', 'parts': [response.text]})
            else:
                # Jika lanjutan, cukup kirim pesan user (konteks sudah ada di history)
                response = chat.send_message(user_message)
                
                # Simpan ke history server
                CHAT_HISTORY[job_id]['messages'].append({'role': 'user', 'parts': [user_message]})
                CHAT_HISTORY[job_id]['messages'].append({'role': 'model', 'parts': [response.text]})

            return jsonify({"reply": response.text})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/favicon.ico")
    def favicon():
        return ("", 204)

    @app.route("/unggah", methods=["GET", "POST"])
    def unggah_data():
        if request.method == "POST":
            if "file" not in request.files:
                flash("Tidak ada file yang dipilih.", "error")
                return redirect(url_for("unggah_data"))
            file = request.files["file"]
            if not file or not file.filename:
                flash("Nama file tidak valid.", "error")
                return redirect(url_for("unggah_data"))
            if not file.filename.lower().endswith(".csv"):
                flash("Format file harus CSV.", "error")
                return redirect(url_for("unggah_data"))

            nama_aman = secure_filename(file.filename)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stored = f"{stamp}_{nama_aman}"
            path = os.path.join(UPLOAD_DIR, stored)
            file.save(path)

            job_id = _buat_job(nama_aman, stored, path)
            session["job_id"] = job_id
            _append_log(job_id, f"File berhasil diunggah: {nama_aman}")
            _append_log(job_id, "Menunggu prapemrosesan.")
            flash("Upload berhasil. Lanjutkan ke prapemrosesan.", "success")
            return redirect(url_for("prapemrosesan"))

        return render_template("unggah.html", kolom_wajib=KolomWajib)

    @app.route("/prapemrosesan", methods=["GET", "POST"])
    def prapemrosesan():
        job_id = session.get("job_id")
        if not job_id:
            flash("Silakan unggah data terlebih dahulu.", "error")
            return redirect(url_for("unggah_data"))
        job = _get_job(job_id)
        if not job:
            flash("Job tidak ditemukan.", "error")
            session.pop("job_id", None)
            return redirect(url_for("unggah_data"))

        if request.method == "POST":
            if job["status"] in ("preprocessing", "clustering"):
                flash("Proses masih berjalan. Silakan tunggu.", "error")
                return redirect(url_for("prapemrosesan"))
            t = threading.Thread(target=_task_prapemrosesan, args=(job_id,), daemon=True)
            t.start()
            flash("Prapemrosesan dimulai.", "success")
            return redirect(url_for("prapemrosesan"))

        # Gunakan per_page=10 untuk preview
        PER_PAGE_PREVIEW_SMALL = 10
        page_raw = int(request.args.get("page_raw", "1") or 1)
        page_pre = int(request.args.get("page_pre", "1") or 1)
        
        preview_raw = _preview_csv_paged(job["uploaded_path"], page=page_raw, per_page=PER_PAGE_PREVIEW_SMALL) if job.get("uploaded_path") else None
        preview_pre = _preview_csv_paged(job["preprocessed_path"], page=page_pre, per_page=PER_PAGE_PREVIEW_SMALL) if job.get("preprocessed_path") else None
        
        ringkasan = json.loads(job["preprocess_summary_json"]) if job.get("preprocess_summary_json") else None
        pre_ok = False
        kolom_tambahan = []
        if preview_raw and preview_pre:
            raw_cols = set(preview_raw["columns"])
            pre_cols = set(preview_pre["columns"])
            kolom_tambahan = sorted(list(pre_cols - raw_cols))
            pre_ok = all(k in pre_cols for k in ["jenis_pestisida_kode", "packaging_kode_angka", "volume_normalisasi"])

        perbandingan = _build_comparison_rows(preview_raw, preview_pre, limit=25) if (preview_raw and preview_pre) else None
        
        # Jika request AJAX untuk status check atau partial update
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
             return jsonify({
                "status": job["status"],
                "step": job.get("step"),
                "has_preview_pre": bool(preview_pre),
                "preview_pre_html": render_template("_partials/preview_pre_table.html", preview_pre=preview_pre, pre_ok=pre_ok, kolom_tambahan=kolom_tambahan) if preview_pre else None,
                "ringkasan_html": render_template("_partials/ringkasan_card.html", ringkasan=ringkasan) if ringkasan else None
             })

        return render_template(
            "prapemrosesan.html",
            job=job,
            preview_raw=preview_raw,
            preview_pre=preview_pre,
            ringkasan=ringkasan,
            pre_ok=pre_ok,
            kolom_tambahan=kolom_tambahan,
            perbandingan=perbandingan,
        )

    @app.route("/clustering", methods=["GET", "POST"])
    def clustering():
        job_id = session.get("job_id")
        if not job_id:
            flash("Silakan unggah data terlebih dahulu.", "error")
            return redirect(url_for("unggah_data"))
        job = _get_job(job_id)
        if not job:
            flash("Job tidak ditemukan.", "error")
            session.pop("job_id", None)
            return redirect(url_for("unggah_data"))
        if not job.get("preprocessed_path"):
            flash("Prapemrosesan belum dilakukan.", "error")
            return redirect(url_for("prapemrosesan"))

        if request.method == "POST":
            if job["status"] == "clustering":
                flash("Clustering sedang berjalan. Silakan tunggu.", "error")
                return redirect(url_for("clustering"))

            jumlah_cluster_hac = int(request.form.get("jumlah_cluster_hac", "4") or 4)
            k_kmedoids = int(request.form.get("k_kmedoids", "4") or 4)
            max_iter = int(request.form.get("max_iter", "25") or 25)

            jalankan = request.form.get("jalankan", "keduanya")
            run_hac = jalankan in ("hac", "keduanya")
            run_kmedoids = jalankan in ("kmedoids", "keduanya")

            _set_job_step(job_id, "clustering", "Menyiapkan proses clustering...")
            t = threading.Thread(
                target=_task_clustering,
                args=(job_id, run_hac, run_kmedoids, jumlah_cluster_hac, k_kmedoids, max_iter),
                daemon=True,
            )
            t.start()
            flash("Proses clustering dimulai.", "success")
            return redirect(url_for("clustering"))

        dendro_url = None
        if job.get("hac_dendrogram_path") and os.path.exists(job["hac_dendrogram_path"]):
            rel = os.path.relpath(job["hac_dendrogram_path"], os.path.join(BASE_DIR, "static"))
            dendro_url = url_for("static", filename=rel.replace("\\", "/"))

        return render_template("clustering.html", job=job, dendro_url=dendro_url)

    @app.route("/hasil")
    def hasil():
        job_id = session.get("job_id")
        if not job_id:
            flash("Silakan unggah data terlebih dahulu.", "error")
            return redirect(url_for("unggah_data"))
        job = _get_job(job_id)
        if not job:
            flash("Job tidak ditemukan.", "error")
            session.pop("job_id", None)
            return redirect(url_for("unggah_data"))
            
        # Cek apakah API Key sudah ada di environment variable
        api_key_configured = bool(os.getenv("GOOGLE_API_KEY"))
        ai_enabled = os.getenv("AI", "false").lower() == "true"

        _ensure_visualisasi_tambahan(int(job_id), job)
        job = _get_job(job_id) or job

        hasil_hac = None
        hasil_km = None
        karakter_hac = None
        karakter_km = None
        sebaran_hac_url = None
        sebaran_km_url = None
        hac_size_bar_url = None
        km_size_bar_url = None
        hac_jenis_url = None
        km_jenis_url = None
        hac_box_url = None
        km_box_url = None
        dendro_url = None
        metrik = json.loads(job["metrics_json"]) if job.get("metrics_json") else None

        if job.get("hac_dendrogram_path") and os.path.exists(job["hac_dendrogram_path"]):
            rel = os.path.relpath(job["hac_dendrogram_path"], os.path.join(BASE_DIR, "static"))
            dendro_url = url_for("static", filename=rel.replace("\\", "/"))

        if job.get("hac_result_path") and os.path.exists(job["hac_result_path"]):
            df_hac = pd.read_csv(job["hac_result_path"])
            hasil_hac = df_hac.head(15).to_dict(orient="records")
            karakter_hac = bangun_tabel_karakteristik_cluster(df_hac, "cluster_hac").to_dict(orient="records")

        if job.get("hac_scatter_path") and os.path.exists(job["hac_scatter_path"]):
            rel = os.path.relpath(job["hac_scatter_path"], os.path.join(BASE_DIR, "static"))
            sebaran_hac_url = url_for("static", filename=rel.replace("\\", "/"))
        if job.get("hac_size_bar_path") and os.path.exists(job["hac_size_bar_path"]):
            rel = os.path.relpath(job["hac_size_bar_path"], os.path.join(BASE_DIR, "static"))
            hac_size_bar_url = url_for("static", filename=rel.replace("\\", "/"))
        if job.get("hac_jenis_stacked_path") and os.path.exists(job["hac_jenis_stacked_path"]):
            rel = os.path.relpath(job["hac_jenis_stacked_path"], os.path.join(BASE_DIR, "static"))
            hac_jenis_url = url_for("static", filename=rel.replace("\\", "/"))
        if job.get("hac_volume_boxplot_path") and os.path.exists(job["hac_volume_boxplot_path"]):
            rel = os.path.relpath(job["hac_volume_boxplot_path"], os.path.join(BASE_DIR, "static"))
            hac_box_url = url_for("static", filename=rel.replace("\\", "/"))

        if job.get("kmedoids_result_path") and os.path.exists(job["kmedoids_result_path"]):
            df_km = pd.read_csv(job["kmedoids_result_path"])
            hasil_km = df_km.head(15).to_dict(orient="records")
            karakter_km = bangun_tabel_karakteristik_cluster(df_km, "cluster_kmedoids").to_dict(orient="records")

        if job.get("kmedoids_scatter_path") and os.path.exists(job["kmedoids_scatter_path"]):
            rel = os.path.relpath(job["kmedoids_scatter_path"], os.path.join(BASE_DIR, "static"))
            sebaran_km_url = url_for("static", filename=rel.replace("\\", "/"))
        if job.get("kmedoids_size_bar_path") and os.path.exists(job["kmedoids_size_bar_path"]):
            rel = os.path.relpath(job["kmedoids_size_bar_path"], os.path.join(BASE_DIR, "static"))
            km_size_bar_url = url_for("static", filename=rel.replace("\\", "/"))
        if job.get("kmedoids_jenis_stacked_path") and os.path.exists(job["kmedoids_jenis_stacked_path"]):
            rel = os.path.relpath(job["kmedoids_jenis_stacked_path"], os.path.join(BASE_DIR, "static"))
            km_jenis_url = url_for("static", filename=rel.replace("\\", "/"))
        if job.get("kmedoids_volume_boxplot_path") and os.path.exists(job["kmedoids_volume_boxplot_path"]):
            rel = os.path.relpath(job["kmedoids_volume_boxplot_path"], os.path.join(BASE_DIR, "static"))
            km_box_url = url_for("static", filename=rel.replace("\\", "/"))

        return render_template(
            "hasil.html",
            job=job,
            hasil_hac=hasil_hac,
            hasil_km=hasil_km,
            karakter_hac=karakter_hac,
            karakter_km=karakter_km,
            sebaran_hac_url=sebaran_hac_url,
            sebaran_km_url=sebaran_km_url,
            hac_size_bar_url=hac_size_bar_url,
            km_size_bar_url=km_size_bar_url,
            hac_jenis_url=hac_jenis_url,
            km_jenis_url=km_jenis_url,
            hac_box_url=hac_box_url,
            km_box_url=km_box_url,
            dendro_url=dendro_url,
            metrik=metrik,
            api_key_configured=api_key_configured,
            ai_enabled=ai_enabled,
        )

    @app.route("/api/job/<int:job_id>/status")
    def api_status(job_id: int):
        job = _get_job(job_id)
        if not job:
            return jsonify({"error": "Job tidak ditemukan"}), 404
        return jsonify(
            {
                "job_id": job["id"],
                "status": job["status"],
                "step": job.get("step") or "",
                "updated_at": job.get("updated_at") or "",
                "last_error": job.get("last_error") or "",
            }
        )

    @app.route("/api/job/<int:job_id>/logs")
    def api_logs(job_id: int):
        job = _get_job(job_id)
        if not job:
            return jsonify({"error": "Job tidak ditemukan"}), 404
        teks = job.get("log_text") or ""
        baris = [l for l in teks.splitlines() if l.strip()]
        return jsonify({"job_id": job["id"], "status": job["status"], "step": job.get("step") or "", "logs": baris[-200:]})

    @app.route("/export/<algo>/<fmt>")
    def export(algo: str, fmt: str):
        job_id = session.get("job_id")
        if not job_id:
            flash("Tidak ada job aktif.", "error")
            return redirect(url_for("dashboard"))
        job = _get_job(job_id)
        if not job:
            flash("Job tidak ditemukan.", "error")
            return redirect(url_for("dashboard"))

        path = None
        nama = None
        if algo == "hac":
            path = job.get("hac_result_path")
            nama = "hasil_hac"
        elif algo == "kmedoids":
            path = job.get("kmedoids_result_path")
            nama = "hasil_kmedoids"
        elif algo == "gabungan":
            path = _buat_file_gabungan(job)
            nama = "hasil_gabungan"
        else:
            flash("Jenis ekspor tidak dikenali.", "error")
            return redirect(url_for("hasil"))

        if not path or not os.path.exists(path):
            flash("Hasil belum tersedia.", "error")
            return redirect(url_for("hasil"))

        if fmt == "csv":
            return send_file(path, as_attachment=True, download_name=f"{nama}.csv", mimetype="text/csv")

        if fmt == "xlsx":
            df = pd.read_csv(path)
            out = os.path.join(UPLOAD_DIR, f"{nama}_{job_id}.xlsx")
            df.to_excel(out, index=False)
            return send_file(out, as_attachment=True, download_name=f"{nama}.xlsx")

        flash("Format ekspor tidak dikenali.", "error")
        return redirect(url_for("hasil"))

    return app


def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def _init_db() -> None:
    con = _conn()
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            stored_filename TEXT NOT NULL,
            uploaded_path TEXT NOT NULL,
            preprocessed_path TEXT,
            hac_result_path TEXT,
            kmedoids_result_path TEXT,
            hac_dendrogram_path TEXT,
            hac_scatter_path TEXT,
            kmedoids_scatter_path TEXT,
            metrics_json TEXT,
            status TEXT NOT NULL,
            step TEXT,
            log_text TEXT,
            preprocess_summary_json TEXT,
            hac_params_json TEXT,
            kmedoids_params_json TEXT,
            last_error TEXT
        )
        """
    )
    _ensure_column(cur, "jobs", "hac_scatter_path", "TEXT")
    _ensure_column(cur, "jobs", "kmedoids_scatter_path", "TEXT")
    _ensure_column(cur, "jobs", "hac_size_bar_path", "TEXT")
    _ensure_column(cur, "jobs", "kmedoids_size_bar_path", "TEXT")
    _ensure_column(cur, "jobs", "hac_jenis_stacked_path", "TEXT")
    _ensure_column(cur, "jobs", "kmedoids_jenis_stacked_path", "TEXT")
    _ensure_column(cur, "jobs", "hac_volume_boxplot_path", "TEXT")
    _ensure_column(cur, "jobs", "kmedoids_volume_boxplot_path", "TEXT")
    _ensure_column(cur, "jobs", "viz_version", "INTEGER")
    _ensure_column(cur, "jobs", "metrics_json", "TEXT")
    con.commit()
    con.close()


def _ensure_column(cur: sqlite3.Cursor, table: str, col: str, col_type: str) -> None:
    cur.execute(f"PRAGMA table_info({table})")
    kolom = {r[1] for r in cur.fetchall()}
    if col not in kolom:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _buat_job(original_filename: str, stored_filename: str, uploaded_path: str) -> int:
    con = _conn()
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO jobs (created_at, updated_at, original_filename, stored_filename, uploaded_path, status, step, log_text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (_now(), _now(), original_filename, stored_filename, uploaded_path, "uploaded", "Upload selesai", ""),
    )
    con.commit()
    job_id = int(cur.lastrowid)
    con.close()
    return job_id


def _get_job(job_id: Optional[int]) -> Optional[Dict[str, Any]]:
    if not job_id:
        return None
    con = _conn()
    cur = con.cursor()
    cur.execute("SELECT * FROM jobs WHERE id = ?", (int(job_id),))
    row = cur.fetchone()
    con.close()
    return dict(row) if row else None


def _update_job(job_id: int, fields: Dict[str, Any]) -> None:
    if not fields:
        return
    fields = dict(fields)
    fields["updated_at"] = _now()
    cols = ", ".join([f"{k} = ?" for k in fields.keys()])
    vals = list(fields.values()) + [int(job_id)]
    con = _conn()
    cur = con.cursor()
    cur.execute(f"UPDATE jobs SET {cols} WHERE id = ?", vals)
    con.commit()
    con.close()


def _append_log(job_id: int, pesan: str) -> None:
    job = _get_job(job_id)
    if not job:
        return
    prefix = datetime.now().strftime("%H:%M:%S")
    baris = f"[{prefix}] {pesan}"
    teks_lama = job.get("log_text") or ""
    teks_baru = (teks_lama + "\n" + baris).strip()
    _update_job(job_id, {"log_text": teks_baru})


def _set_job_step(job_id: int, status: str, step: str) -> None:
    _update_job(job_id, {"status": status, "step": step})
    _append_log(job_id, step)


def _preview_csv_paged(path: str, page: int, per_page: int) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    page = max(int(page), 1)
    per_page = max(int(per_page), 1)
    df = pd.read_csv(path)
    # Sembunyikan kolom yang tidak ingin ditampilkan di preview
    hidden_cols = ["packaging_kode_huruf"]
    df = df.drop(columns=[c for c in hidden_cols if c in df.columns], errors="ignore")
    total = int(df.shape[0])
    total_pages = max((total + per_page - 1) // per_page, 1)
    if page > total_pages:
        page = total_pages
    start = (page - 1) * per_page
    end = start + per_page
    slice_df = df.iloc[start:end].copy()
    return {
        "columns": slice_df.columns.tolist(),
        "rows": slice_df.to_dict(orient="records"),
        "page": int(page),
        "per_page": int(per_page),
        "total_rows": int(total),
        "total_pages": int(total_pages),
    }


def _build_comparison_rows(
    preview_raw: Dict[str, Any], preview_pre: Dict[str, Any], limit: int = 20
) -> Dict[str, Any]:
    raw_rows = preview_raw.get("rows") or []
    pre_rows = preview_pre.get("rows") or []
    n = min(int(limit), len(raw_rows), len(pre_rows))

    def g(row: Dict[str, Any], key: str) -> Any:
        return row.get(key, "")

    out = []
    for i in range(n):
        r = raw_rows[i]
        p = pre_rows[i]
        out.append(
            {
                "idx": i + 1 + (int(preview_pre.get("page", 1)) - 1) * int(preview_pre.get("per_page", 100)),
                "retailer": g(r, "RETAILER NAME (R1)"),
                "jenis_raw": g(r, "JENIS PESTISIDA"),
                "jenis_kode": g(p, "jenis_pestisida_kode"),
                "packaging_raw": g(r, "PACKAGING"),
                "packaging_angka": g(p, "packaging_kode_angka"),
                "volume_raw": g(r, "VOLUME (L/KG)"),
                "volume_norm": g(p, "volume_normalisasi"),
                "distributor_kode": g(p, "distributor_kode"),
                "area_kode": g(p, "area_kode"),
                "regency_kode": g(p, "regency_kode"),
                "product_kode": g(p, "product_kode"),
            }
        )
    return {"rows": out, "page": preview_pre.get("page", 1)}


def _task_prapemrosesan(job_id: int) -> None:
    job = _get_job(job_id)
    if not job:
        return
    try:
        _set_job_step(job_id, "preprocessing", "Prapemrosesan dimulai...")
        input_path = job["uploaded_path"]
        out_path = os.path.join(UPLOAD_DIR, f"job_{job_id}_preprocessed.csv")

        def cb(msg: str) -> None:
            _set_job_step(job_id, "preprocessing", msg)
            time.sleep(0.05)

        df, ringkas = prapemrosesan_dataset(input_path, out_path, progress_cb=cb)
        _update_job(job_id, {"preprocessed_path": out_path, "preprocess_summary_json": ringkas.ke_json()})
        _set_job_step(job_id, "preprocessed", f"Prapemrosesan selesai. Total data bersih: {int(df.shape[0])}.")
    except Exception as e:
        _update_job(job_id, {"status": "error", "last_error": str(e), "step": "Terjadi kesalahan saat prapemrosesan."})
        _append_log(job_id, f"Error: {e}")


def _task_clustering(
    job_id: int,
    run_hac: bool,
    run_kmedoids: bool,
    jumlah_cluster_hac: int,
    k_kmedoids: int,
    max_iter: int,
) -> None:
    job = _get_job(job_id)
    if not job or not job.get("preprocessed_path"):
        return
    try:
        _set_job_step(job_id, "clustering", "Membaca data hasil prapemrosesan...")
        df = pd.read_csv(job["preprocessed_path"])

        def cb(msg: str) -> None:
            _set_job_step(job_id, "clustering", msg)
            time.sleep(0.05)

        if run_hac:
            cb("Menjalankan HAC (Single Linkage)...")
            dendro_path = os.path.join(STATIC_GEN_DIR, f"dendrogram_job_{job_id}.png")
            label_hac, hac_params = jalankan_hac(df, jumlah_cluster_hac, dendro_path, progress_cb=cb)
            df_hac = df.copy()
            df_hac["cluster_hac"] = label_hac
            cb("Membuat peta distribusi cluster (HAC)...")
            scatter_hac = os.path.join(STATIC_GEN_DIR, f"sebaran_hac_job_{job_id}.png")
            buat_plot_sebaran_cluster(df_hac, "cluster_hac", scatter_hac, "Sebaran Cluster HAC (Volume vs Jenis)")
            cb("Membuat diagram batang ukuran cluster (HAC)...")
            hac_size_bar = os.path.join(STATIC_GEN_DIR, f"ukuran_cluster_hac_job_{job_id}.png")
            buat_plot_ukuran_cluster_batang(df_hac, "cluster_hac", hac_size_bar, "Ukuran Cluster HAC (Jumlah Anggota)")
            cb("Membuat komposisi jenis pestisida per cluster (HAC)...")
            hac_jenis = os.path.join(STATIC_GEN_DIR, f"komposisi_jenis_hac_job_{job_id}.png")
            buat_plot_komposisi_jenis_stacked(df_hac, "cluster_hac", hac_jenis, "Komposisi Jenis Pestisida per Cluster (HAC)")
            cb("Membuat boxplot sebaran volume per cluster (HAC)...")
            hac_box = os.path.join(STATIC_GEN_DIR, f"boxplot_volume_hac_job_{job_id}.png")
            buat_plot_boxplot_volume(df_hac, "cluster_hac", hac_box, "Sebaran Volume per Cluster (HAC)")
            out_hac = os.path.join(UPLOAD_DIR, f"job_{job_id}_hasil_hac.csv")
            df_hac.to_csv(out_hac, index=False)
            _update_job(
                job_id,
                {
                    "hac_result_path": out_hac,
                    "hac_dendrogram_path": dendro_path,
                    "hac_scatter_path": scatter_hac,
                    "hac_size_bar_path": hac_size_bar,
                    "hac_jenis_stacked_path": hac_jenis,
                    "hac_volume_boxplot_path": hac_box,
                    "hac_params_json": hac_params,
                },
            )

        if run_kmedoids:
            cb("Menjalankan K-Medoids (PAM)...")
            label_km, medoids, km_params = jalankan_kmedoids(
                df, k=k_kmedoids, max_iter=max_iter, random_state=42, progress_cb=cb
            )
            df_km = df.copy()
            df_km["cluster_kmedoids"] = label_km
            df_km["is_medoid"] = df_km.index.isin(medoids)
            cb("Membuat peta distribusi cluster (K-Medoids)...")
            scatter_km = os.path.join(STATIC_GEN_DIR, f"sebaran_kmedoids_job_{job_id}.png")
            buat_plot_sebaran_cluster(df_km, "cluster_kmedoids", scatter_km, "Sebaran Cluster K-Medoids (Volume vs Jenis)")
            cb("Membuat diagram batang ukuran cluster (K-Medoids)...")
            km_size_bar = os.path.join(STATIC_GEN_DIR, f"ukuran_cluster_kmedoids_job_{job_id}.png")
            buat_plot_ukuran_cluster_batang(df_km, "cluster_kmedoids", km_size_bar, "Ukuran Cluster K-Medoids (Jumlah Anggota)")
            cb("Membuat komposisi jenis pestisida per cluster (K-Medoids)...")
            km_jenis = os.path.join(STATIC_GEN_DIR, f"komposisi_jenis_kmedoids_job_{job_id}.png")
            buat_plot_komposisi_jenis_stacked(
                df_km, "cluster_kmedoids", km_jenis, "Komposisi Jenis Pestisida per Cluster (K-Medoids)"
            )
            cb("Membuat boxplot sebaran volume per cluster (K-Medoids)...")
            km_box = os.path.join(STATIC_GEN_DIR, f"boxplot_volume_kmedoids_job_{job_id}.png")
            buat_plot_boxplot_volume(df_km, "cluster_kmedoids", km_box, "Sebaran Volume per Cluster (K-Medoids)")
            out_km = os.path.join(UPLOAD_DIR, f"job_{job_id}_hasil_kmedoids.csv")
            df_km.to_csv(out_km, index=False)
            _update_job(
                job_id,
                {
                    "kmedoids_result_path": out_km,
                    "kmedoids_scatter_path": scatter_km,
                    "kmedoids_size_bar_path": km_size_bar,
                    "kmedoids_jenis_stacked_path": km_jenis,
                    "kmedoids_volume_boxplot_path": km_box,
                    "kmedoids_params_json": km_params,
                },
            )

        cb("Menghitung ringkasan kualitas sederhana (dispersi & sebaran ukuran cluster)...")
        job2 = _get_job(job_id)
        metrik: Dict[str, Any] = {}
        if job2 and job2.get("hac_result_path") and os.path.exists(job2["hac_result_path"]):
            df_h = pd.read_csv(job2["hac_result_path"])
            metrik["hac"] = ringkasan_kualitas_sederhana(df_h, "cluster_hac")
        if job2 and job2.get("kmedoids_result_path") and os.path.exists(job2["kmedoids_result_path"]):
            df_k = pd.read_csv(job2["kmedoids_result_path"])
            metrik["kmedoids"] = ringkasan_kualitas_sederhana(df_k, "cluster_kmedoids")
        if metrik:
            _update_job(job_id, {"metrics_json": json.dumps(metrik, ensure_ascii=False)})

        _set_job_step(job_id, "done", "Clustering selesai. Silakan buka Hasil Analisis.")
    except Exception as e:
        _update_job(job_id, {"status": "error", "last_error": str(e), "step": "Terjadi kesalahan saat clustering."})
        _append_log(job_id, f"Error: {e}")


def _ensure_visualisasi_tambahan(job_id: int, job: Dict[str, Any]) -> None:
    try:
        if int(job.get("viz_version") or 0) != 2:
            fields_v: Dict[str, Any] = {"viz_version": 2}
            if job.get("hac_result_path") and os.path.exists(job["hac_result_path"]):
                df_h0 = pd.read_csv(job["hac_result_path"])
                if "cluster_hac" in df_h0.columns:
                    p = os.path.join(STATIC_GEN_DIR, f"sebaran_hac_job_{job_id}_v2.png")
                    buat_plot_sebaran_cluster(df_h0, "cluster_hac", p, "Sebaran Cluster HAC (Volume vs Jenis)")
                    fields_v["hac_scatter_path"] = p
            if job.get("kmedoids_result_path") and os.path.exists(job["kmedoids_result_path"]):
                df_k0 = pd.read_csv(job["kmedoids_result_path"])
                if "cluster_kmedoids" in df_k0.columns:
                    p = os.path.join(STATIC_GEN_DIR, f"sebaran_kmedoids_job_{job_id}_v2.png")
                    buat_plot_sebaran_cluster(
                        df_k0, "cluster_kmedoids", p, "Sebaran Cluster K-Medoids (Volume vs Jenis)"
                    )
                    fields_v["kmedoids_scatter_path"] = p
            _update_job(job_id, fields_v)

        if job.get("hac_result_path") and os.path.exists(job["hac_result_path"]):
            df_h = pd.read_csv(job["hac_result_path"])
            if "cluster_hac" in df_h.columns:
                fields: Dict[str, Any] = {}
                if (not job.get("hac_size_bar_path")) or (not os.path.exists(job["hac_size_bar_path"])):
                    p = os.path.join(STATIC_GEN_DIR, f"ukuran_cluster_hac_job_{job_id}.png")
                    buat_plot_ukuran_cluster_batang(df_h, "cluster_hac", p, "Ukuran Cluster HAC (Jumlah Anggota)")
                    fields["hac_size_bar_path"] = p
                if (not job.get("hac_jenis_stacked_path")) or (not os.path.exists(job["hac_jenis_stacked_path"])):
                    p = os.path.join(STATIC_GEN_DIR, f"komposisi_jenis_hac_job_{job_id}.png")
                    buat_plot_komposisi_jenis_stacked(df_h, "cluster_hac", p, "Komposisi Jenis Pestisida per Cluster (HAC)")
                    fields["hac_jenis_stacked_path"] = p
                if (not job.get("hac_volume_boxplot_path")) or (not os.path.exists(job["hac_volume_boxplot_path"])):
                    p = os.path.join(STATIC_GEN_DIR, f"boxplot_volume_hac_job_{job_id}.png")
                    buat_plot_boxplot_volume(df_h, "cluster_hac", p, "Sebaran Volume per Cluster (HAC)")
                    fields["hac_volume_boxplot_path"] = p
                if fields:
                    _update_job(job_id, fields)

        if job.get("kmedoids_result_path") and os.path.exists(job["kmedoids_result_path"]):
            df_k = pd.read_csv(job["kmedoids_result_path"])
            if "cluster_kmedoids" in df_k.columns:
                fields2: Dict[str, Any] = {}
                if (not job.get("kmedoids_size_bar_path")) or (not os.path.exists(job["kmedoids_size_bar_path"])):
                    p = os.path.join(STATIC_GEN_DIR, f"ukuran_cluster_kmedoids_job_{job_id}.png")
                    buat_plot_ukuran_cluster_batang(df_k, "cluster_kmedoids", p, "Ukuran Cluster K-Medoids (Jumlah Anggota)")
                    fields2["kmedoids_size_bar_path"] = p
                if (not job.get("kmedoids_jenis_stacked_path")) or (not os.path.exists(job["kmedoids_jenis_stacked_path"])):
                    p = os.path.join(STATIC_GEN_DIR, f"komposisi_jenis_kmedoids_job_{job_id}.png")
                    buat_plot_komposisi_jenis_stacked(
                        df_k, "cluster_kmedoids", p, "Komposisi Jenis Pestisida per Cluster (K-Medoids)"
                    )
                    fields2["kmedoids_jenis_stacked_path"] = p
                if (not job.get("kmedoids_volume_boxplot_path")) or (not os.path.exists(job["kmedoids_volume_boxplot_path"])):
                    p = os.path.join(STATIC_GEN_DIR, f"boxplot_volume_kmedoids_job_{job_id}.png")
                    buat_plot_boxplot_volume(df_k, "cluster_kmedoids", p, "Sebaran Volume per Cluster (K-Medoids)")
                    fields2["kmedoids_volume_boxplot_path"] = p
                if fields2:
                    _update_job(job_id, fields2)
    except Exception as e:
        _append_log(job_id, f"Error visualisasi tambahan: {e}")


def _dashboard_stats() -> Dict[str, Any]:
    con = _conn()
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) AS n FROM jobs")
    total = int(cur.fetchone()["n"])
    cur.execute("SELECT COUNT(*) AS n FROM jobs WHERE status = 'done'")
    done = int(cur.fetchone()["n"])
    cur.execute("SELECT id, created_at, status FROM jobs ORDER BY id DESC LIMIT 5")
    terbaru = [dict(r) for r in cur.fetchall()]
    con.close()
    return {"total_job": total, "job_selesai": done, "job_terbaru": terbaru}


def _buat_file_gabungan(job: Dict[str, Any]) -> Optional[str]:
    path_hac = job.get("hac_result_path")
    path_km = job.get("kmedoids_result_path")
    if not path_hac or not path_km:
        return None
    if not (os.path.exists(path_hac) and os.path.exists(path_km)):
        return None
    df_hac = pd.read_csv(path_hac)
    df_km = pd.read_csv(path_km)
    kolom_km = ["cluster_kmedoids"]
    if "is_medoid" in df_km.columns:
        kolom_km.append("is_medoid")
    gabung = df_hac.copy()
    for c in kolom_km:
        gabung[c] = df_km[c].values
    out = os.path.join(UPLOAD_DIR, f"job_{job['id']}_hasil_gabungan.csv")
    gabung.to_csv(out, index=False)
    return out


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
