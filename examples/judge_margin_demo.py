import math, random, sys
import mlx.core as mx
from mlx_lm import load

model, tok = load("mlx-community/Qwen2.5-1.5B-Instruct-8bit")
YES = tok.encode("Yes", add_special_tokens=False)[0]
NO = tok.encode("No", add_special_tokens=False)[0]

def margin(prompt):
    msg = [{"role": "user", "content": prompt + "\n\nAnswer with only one word: Yes or No."}]
    text = tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
    ids = mx.array([tok.encode(text, add_special_tokens=False)])
    logits = model(ids)[0, -1].astype(mx.float32)      # scores for the next token
    return (logits[YES] - logits[NO]).item()           # > 0 means "Yes"

def seeded_verdict(d, T, seed):
    p_yes = 1 / (1 + math.exp(-d / T))
    return random.Random(seed).random() < p_yes

passage = "The Eiffel Tower was completed in 1889 for the World's Fair in Paris."
for claim in ["The Eiffel Tower was completed in 1889.", "The Eiffel Tower was completed in 1925.", "The Eiffel Tower is the tallest structure in Europe."]:
    d = margin(f"Passage: {passage}\nClaim: {claim}\nIs the claim supported by the passage?")
    print(f"d = {d:+6.2f}  T=0: {'Yes' if d > 0 else 'No':3}  10 seeded verdicts at T=0.3: " +
          "".join("Y" if seeded_verdict(d, 0.3, s) else "N" for s in range(10)) + f"   | {claim}")

sys.path.insert(0, ".")
import judge as J
jd = J.Judge("qwen2.5-1.5b")
p = f"Passage: {passage}\nClaim: {passage[:30]}\nIs the claim supported by the passage?"
text = jd.tok.apply_chat_template([{"role": "user", "content": p + J.FMT}], add_generation_prompt=True, tokenize=False)
print("harness vs snippet on the same prompt:", round(jd.forward([jd.tok.encode(text, add_special_tokens=False)])[0][0], 4), round(margin(p), 4))
