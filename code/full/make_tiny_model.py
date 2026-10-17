"""Smoke test only: build a tiny random Qwen2-style model and tokenizer locally,
so the pipeline can be tested without downloading the real 7B model."""
import json, sys
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

out = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/p2_smoke/tiny_model")
out.mkdir(parents=True, exist_ok=True)
data = Path(__file__).resolve().parent.parent / "data" / "generated"
texts = []
for f in data.glob("*_items.jsonl"):
    for line in f.read_text().splitlines()[:400]:
        it = json.loads(line)
        texts += [m["content"] for m in it["messages"]] + [it["lawful"], it["violating"]]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fallback_data as fb
texts += [m["content"] for r in fb.BENIGN for m in r["messages"]] + fb.HARMFUL + fb.HARMLESS

special = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
tk = Tokenizer(models.BPE())
tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tk.decoder = decoders.ByteLevel()
tk.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=1500, special_tokens=special,
                                                   initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
CHAT = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                              bos_token=None, unk_token=None)
tok.chat_template = CHAT
tok.save_pretrained(out)
cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                  eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id, tie_word_embeddings=True)
Qwen2ForCausalLM(cfg).save_pretrained(out)
print("tiny model saved to", out, "vocab", len(tok))
