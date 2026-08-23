from __future__ import annotations

import builtins
import importlib.util
import unittest
from pathlib import Path
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(
        name, REPOSITORY_ROOT / relative_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {relative_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODEL_40M = load_module("hojo_40m_onnx_model", "Hojo-TTS-Light-40M/onnx_model.py")
MODEL_80M = load_module("hojo_80m_onnx_model", "Hojo-TTS-Light-80M/onnx_model.py")


def import_without_onnx(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "onnx":
        raise ImportError("onnx intentionally hidden by test")
    return ORIGINAL_IMPORT(name, globals, locals, fromlist, level)


ORIGINAL_IMPORT = builtins.__import__


class MissingOnnxDependencyTests(unittest.TestCase):
    def test_40m_reports_missing_onnx(self):
        with mock.patch("builtins.__import__", side_effect=import_without_onnx):
            with self.assertRaisesRegex(RuntimeError, "pip install onnx"):
                MODEL_40M._ort_model_source("published-bf16-model.onnx")

    def test_80m_cpu_reports_missing_onnx(self):
        with mock.patch("builtins.__import__", side_effect=import_without_onnx):
            with self.assertRaisesRegex(RuntimeError, "pip install onnx"):
                MODEL_80M._ort_model_source(
                    "published-bf16-model.onnx", promote_bf16_for_cpu=True
                )

    def test_80m_cuda_does_not_require_graph_promotion(self):
        model_path = "published-bf16-model.onnx"
        with mock.patch("builtins.__import__", side_effect=import_without_onnx):
            self.assertEqual(
                MODEL_80M._ort_model_source(
                    model_path, promote_bf16_for_cpu=False
                ),
                model_path,
            )


if __name__ == "__main__":
    unittest.main()
