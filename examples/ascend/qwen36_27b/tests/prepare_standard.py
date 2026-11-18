import argparse
from pathlib import Path

import yaml
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast


parser = argparse.ArgumentParser()
parser.add_argument("--work-dir", type=Path, required=True)
args = parser.parse_args()
root = args.work_dir.resolve()
dest = root / "llama"
dest.mkdir(exist_ok=True)
tokenizer = Tokenizer(
    WordLevel(
        {
            "[PAD]": 0,
            "[UNK]": 1,
            "[EOS]": 2,
            "[BOS]": 3,
            "What": 4,
            "is": 5,
            "plus": 6,
            "one": 7,
            "?": 8,
            "0": 9,
            "1": 10,
            "2": 11,
            "3": 12,
            "4": 13,
        },
        unk_token="[UNK]",
    )
)
tokenizer.pre_tokenizer = Whitespace()
PreTrainedTokenizerFast(
    tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]", bos_token="[BOS]"
).save_pretrained(dest)
LlamaForCausalLM(
    LlamaConfig(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        pad_token_id=0,
        bos_token_id=3,
        eos_token_id=2,
    )
).save_pretrained(dest)
cfg = yaml.safe_load((root / "train.yaml").read_text())
cfg.update(model_name_or_path=str(dest), output_dir=str(root / "standard"), template="default", max_steps=1)
cfg.pop("enable_thinking")
cfg["launcher"].update(profile="standard")
cfg["launcher"].pop("optimizations")
(root / "standard.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
