from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class HpcTrainingScriptsTest(unittest.TestCase):
    def test_32b_qlora_training_script_defaults_to_requested_model(self):
        script = ROOT / "scripts" / "train_qwen3_swallow_32b_qlora.py"
        text = script.read_text()

        self.assertIn("tokyotech-llm/Qwen3-Swallow-32B-CPT-v0.2", text)
        self.assertIn("BitsAndBytesConfig", text)
        self.assertIn("LoraConfig", text)
        self.assertIn("--data-dir", text)
        self.assertIn("--output-dir", text)

    def test_slurm_job_runs_training_inside_singularity_with_gpu_support(self):
        job = ROOT / "scripts" / "slurm" / "train_qwen3_swallow_32b_qlora.sbatch"
        text = job.read_text()

        self.assertIn("#SBATCH", text)
        self.assertIn("singularity exec --nv", text)
        self.assertIn("scripts/train_qwen3_swallow_32b_qlora.py", text)
        self.assertIn("tokyotech-llm/Qwen3-Swallow-32B-CPT-v0.2", text)

    def test_singularity_definition_installs_gpu_training_stack(self):
        definition = ROOT / "singularity" / "qwen3_qlora.def"
        text = definition.read_text()

        self.assertIn("Bootstrap: docker", text)
        self.assertIn("nvidia/cuda", text)
        self.assertIn("torch", text)
        self.assertIn("transformers", text)
        self.assertIn("peft", text)
        self.assertIn("bitsandbytes", text)


if __name__ == "__main__":
    unittest.main()
