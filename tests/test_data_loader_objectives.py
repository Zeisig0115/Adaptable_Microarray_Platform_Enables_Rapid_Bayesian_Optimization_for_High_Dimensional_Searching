from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data_loader_objectives import (
    PLATE_WELLS,
    process_mixed_input,
    source_row_to_plate_pos,
)


class PlatePositionTests(unittest.TestCase):
    @staticmethod
    def _raw_essentials(n_rows: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "HRP": [0.001] * n_rows,
                "TMB": [0.825] * n_rows,
                "H2O2": [0.0435] * n_rows,
                "frame 0": [16.0] * n_rows,
                "frame 1": [17.0] * n_rows,
            }
        )

    @staticmethod
    def _process(path: Path) -> pd.DataFrame:
        return process_mixed_input(
            path,
            batch="test_plate",
            input_sheet="Sheet1",
            baseline=16.0,
            signal_threshold=5.0,
            retention_last_n=1,
            smooth_window=1,
            auc_method="trapz",
            use_filter=False,
            max_active=None,
        )

    def test_boundary_positions(self) -> None:
        expected = {
            0: "(1,1)",
            1: "(1,2)",
            23: "(1,24)",
            24: "(2,1)",
            25: "(2,2)",
            383: "(16,24)",
        }
        for source_row, plate_pos in expected.items():
            with self.subTest(source_row=source_row):
                self.assertEqual(source_row_to_plate_pos(source_row), plate_pos)

    def test_invalid_source_rows_are_rejected(self) -> None:
        for source_row in (-1, PLATE_WELLS):
            with self.subTest(source_row=source_row):
                with self.assertRaises(ValueError):
                    source_row_to_plate_pos(source_row)
        for source_row in (1.5, "1", True):
            with self.subTest(source_row=source_row):
                with self.assertRaises(TypeError):
                    source_row_to_plate_pos(source_row)  # type: ignore[arg-type]

    def test_unified_output_places_plate_pos_after_source_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "essentials.csv"
            self._raw_essentials(PLATE_WELLS).to_csv(path, index=False)
            result = self._process(path)

        source_col = result.columns.get_loc("source_row")
        self.assertEqual(result.columns[source_col + 1], "plate_pos")
        for source_row, expected in {
            0: "(1,1)",
            1: "(1,2)",
            23: "(1,24)",
            24: "(2,1)",
            25: "(2,2)",
            383: "(16,24)",
        }.items():
            self.assertEqual(result.loc[source_row, "source_row"], source_row)
            self.assertEqual(result.loc[source_row, "plate_pos"], expected)

    def test_non_full_plate_is_rejected(self) -> None:
        for n_rows in (PLATE_WELLS - 1, PLATE_WELLS + 1):
            with self.subTest(n_rows=n_rows), tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "partial_or_oversized.csv"
                self._raw_essentials(n_rows).to_csv(path, index=False)
                with self.assertRaisesRegex(ValueError, "requires exactly 384 rows"):
                    self._process(path)


if __name__ == "__main__":
    unittest.main()
