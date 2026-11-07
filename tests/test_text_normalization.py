import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_normalizer(bundle):
    path = ROOT / bundle / "text_normalization.py"
    spec = importlib.util.spec_from_file_location(f"normalizer_{bundle}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.normalize_text


class TextNormalizationTests(unittest.TestCase):
    def test_both_bundles_expand_supported_text(self):
        for bundle in ("Hojo-TTS-Light-40M", "Hojo-TTS-Light-80M"):
            with self.subTest(bundle=bundle):
                normalize = load_normalizer(bundle)
                self.assertEqual(
                    normalize("The battery is at 80 percent."),
                    "The battery is at eighty percent.",
                )
                self.assertEqual(normalize("Set it to 80% now."), "Set it to eighty percent now.")
                self.assertEqual(normalize("室内温度是 26.5 度。"), "室内温度是 二十六点五 度。")
                self.assertEqual(normalize("Hello world."), "Hello world.")


if __name__ == "__main__":
    unittest.main()
