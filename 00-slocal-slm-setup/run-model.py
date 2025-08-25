# Load model directly
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    logging,
)
import sys
import os

# module探索パスを追加
# Pythonの相対インポートで上位ディレクトリ・サブディレクトリを指定 | note.nkmk.me
# https://note.nkmk.me/python-relative-import/
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))

from script.utils.memory_flusher import flush


logging.set_verbosity_info()


# メモリが足りない場合は, 下記で量子化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
)

MODEL_ID = "google/gemma-3-270m-it"  # instruction-tuned版推奨

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_fast=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="cuda",
    torch_dtype="auto",
    do_sample=True,
    # quantization_config=bnb_config,
    pad_token_id=tokenizer.eos_token_id,
)

prompt = """大規模言語モデルに関して，日本語で説明してください。
その際要点5個を挙げ，各要点を段落にして詳しく説明してください。"""

model_input = tokenizer(prompt, return_tensors="pt").to(model.device)
input_ids = model_input["input_ids"]

model.eval()
with torch.no_grad():
    result = model.generate(
        input_ids,
        max_new_tokens=300,
        # eos_token_id=terminators,
        do_sample=False,
    )
    result = result[0][input_ids.shape[-1] :]
    output = tokenizer.decode(result, skip_special_tokens=True)
    print("\n-----生成結果-----\n", output)

    del input_ids
    del model_input
    flush()
