import json
import os
import tempfile
import unittest

import pandas as pd

from clustering_logic import prapemrosesan_dataset


class TestPreprocessing(unittest.TestCase):
    def _write_csv(self, df: pd.DataFrame, path: str) -> None:
        df.to_csv(path, index=False)

    def test_pipeline_menambah_kolom_transformasi_dan_minmax_0_1(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            in_path = os.path.join(td, "in.csv")
            out_path = os.path.join(td, "out.csv")
            df = pd.DataFrame(
                {
                    "YEAR": [2024, 2024, 2024],
                    "MONTH": ["01.JAN", "01.JAN", "02.FEB"],
                    "DISTRIBUTOR": ["D1", "D1", "D2"],
                    "AREA": ["A1", "A1", "A2"],
                    "RETAILER NAME (R1)": ["R1", "R2", "R3"],
                    "ADDRESS R1": ["Addr1", "Addr2", "Addr3"],
                    "REGENCY/DISTRICT": ["Reg1", "Reg1", "Reg2"],
                    "PRODUCT": ["P1", "P2", "P3"],
                    "JENIS PESTISIDA": ["Insektisida", "Herbisida", "Fungisida"],
                    "PACKAGING": ["Botol", "Sachet", "Botol"],
                    "VOLUME (L/KG)": ["10,0", "20.0", "30"],
                }
            )
            self._write_csv(df, in_path)
            df_out, ringkas = prapemrosesan_dataset(in_path, out_path)

            for kol in [
                "jenis_pestisida_kode",
                "packaging_kode_angka",
                "volume_normalisasi",
                "distributor_kode",
                "area_kode",
                "regency_kode",
                "product_kode",
            ]:
                self.assertIn(kol, df_out.columns)

            self.assertGreaterEqual(df_out["volume_normalisasi"].min(), 0.0)
            self.assertLessEqual(df_out["volume_normalisasi"].max(), 1.0)

            meta = json.loads(ringkas.ke_json())
            self.assertIn("volume_norm_min", meta)
            self.assertIn("volume_norm_max", meta)

    def test_missing_values_dan_duplikasi_terhapus(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            in_path = os.path.join(td, "in.csv")
            out_path = os.path.join(td, "out.csv")

            df = pd.DataFrame(
                {
                    "YEAR": [2024, 2024, None, 2024],
                    "MONTH": [1, 1, 1, 1],
                    "DISTRIBUTOR": ["D1", "D1", "D1", "D1"],
                    "AREA": ["A1", "A1", "A1", "A1"],
                    "RETAILER NAME (R1)": ["R1", "R1", "R2", "R1"],
                    "ADDRESS R1": ["Addr1", "Addr1", "Addr2", "Addr1"],
                    "REGENCY/DISTRICT": ["Reg1", "Reg1", "Reg1", "Reg1"],
                    "PRODUCT": ["P1", "P1", "P2", "P1"],
                    "JENIS PESTISIDA": ["Insektisida", "Insektisida", "Herbisida", "Insektisida"],
                    "PACKAGING": ["Botol", "Botol", "Botol", "Botol"],
                    "VOLUME (L/KG)": [10, 10, 20, 10],
                }
            )
            self._write_csv(df, in_path)
            df_out, ringkas = prapemrosesan_dataset(in_path, out_path)

            self.assertEqual(int(ringkas.jumlah_awal), 4)
            self.assertGreaterEqual(int(ringkas.jumlah_dihapus_na), 1)
            self.assertGreaterEqual(int(ringkas.jumlah_dihapus_duplikat), 1)
            self.assertTrue(df_out.shape[0] <= 2)


if __name__ == "__main__":
    unittest.main()
