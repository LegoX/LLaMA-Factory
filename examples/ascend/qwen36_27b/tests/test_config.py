import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import yaml


spec = importlib.util.spec_from_file_location("runner", Path(__file__).resolve().parents[1] / "runner.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "model").mkdir()
        (self.root / "model/config.json").write_text(json.dumps({"model_type": "llama"}))
        (self.root / "data.json").write_text("[]")
        self.cfg = {
            "model_name_or_path": "model",
            "output_dir": "out",
            "do_train": True,
            "dataset": "d",
            "launcher": {"dataset_info": {"d": {"file_name": "data.json"}}},
        }

    def tearDown(self):
        self.tmp.cleanup()

    def load(self):
        path = self.root / "job.yaml"
        path.write_text(yaml.safe_dump(self.cfg))
        return r.load_config(path)

    def test_relative_paths_and_standard_isolation(self):
        _, train, launch, files = self.load()
        assert train["model_name_or_path"] == str(self.root / "model")
        assert files == [str(self.root / "data.json")]
        assert not any(k.startswith("PROBE_") for k in r.environment(launch))

    def test_wrong_architecture_rejected_before_training(self):
        self.cfg["launcher"]["profile"] = "qwen36_fsdp2"
        with self.assertRaisesRegex(ValueError, "architecture"):
            self.load()

    def test_unknown_launcher_key_not_silently_ignored(self):
        self.cfg["launcher"]["nproc_per_nodes"] = 16
        with self.assertRaisesRegex(ValueError, "Unknown launcher"):
            self.load()

    def test_missing_dataset_rejected(self):
        (self.root / "data.json").unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            self.load()

    def test_resume_requires_path_not_boolean(self):
        self.cfg["resume_from_checkpoint"] = True
        with self.assertRaisesRegex(ValueError, "path string"):
            self.load()

    def test_existing_output_protected(self):
        self.cfg["overwrite_output_dir"] = True
        with self.assertRaisesRegex(ValueError, "overwrite"):
            self.load()


if __name__ == "__main__":
    unittest.main()
