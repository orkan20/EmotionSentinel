from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "lzw1008/Emollama-chat-7b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto"
)

system = "You are an emotional scoring model. Score text on valence, arousal, and importance."
user = "Score the following text on valence (-1.0 to 1.0), arousal (0.0 to 1.0), and importance (0.0 to 1.0). Return only a JSON object with those three fields and nothing else.\nText: This is the best day of my entire life."

text = f"[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]"

inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))