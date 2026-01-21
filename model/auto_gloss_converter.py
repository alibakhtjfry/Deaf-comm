import re
import torch
from transformers import MarianMTModel, MarianTokenizer

# =========================
# PATHS (DEAF-COMM)
# =========================
TEST_TEXT_PATH  = r"C:\Users\ctrend\Deaf-comm\model\Data\tmp\test.text"
TEST_GLOSS_PATH = r"C:\Users\ctrend\Deaf-comm\model\Data\tmp\test.gloss"
VOCAB_PATH      = r"C:\Users\ctrend\Deaf-comm\model\Configs\src_vocab.txt"
DEBUG_FULL_GLOSS_PATH = r"C:\Users\ctrend\Deaf-comm\model\Data\tmp\debug_full_gloss.txt"

# =========================
# LOAD MODEL VOCAB
# =========================
def load_vocab(path):
    vocab = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w:
                vocab.add(w.upper())
    return vocab

SRC_VOCAB = load_vocab(VOCAB_PATH)

# =========================
# STOP WORDS (German)
# =========================
GERMAN_STOP_WORDS = {
    "ich", "du", "er", "sie", "es", "wir", "ihr",
    "mein", "meine", "meiner",
    "bin", "bist", "ist", "sind", "seid",
    "der", "die", "das", "ein", "eine",
    "und", "oder", "aber",
    "in", "im", "von", "zu", "mit", "für"
}

# =========================
# LOAD TRANSLATION MODEL
# =========================
MODEL_NAME = "Helsinki-NLP/opus-mt-en-de"
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME)
model.eval()

# =========================
# FUNCTIONS
# =========================
def english_to_german(text: str) -> str:
    batch = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        generated = model.generate(**batch, max_length=128)
    return tokenizer.decode(generated[0], skip_special_tokens=True)

def german_to_gloss(text: str):
    text = text.lower()
    text = re.sub(r"[^\w\säöüß]", "", text)
    words = text.split()
    return [w.upper() for w in words if w not in GERMAN_STOP_WORDS]

# =========================
# RUN
# =========================
english_text = input("\nEnter an English sentence: ").strip()
print("\nENGLISH:", english_text)

german_text = english_to_german(english_text)
print("GERMAN (FULL):", german_text)

raw_gloss = german_to_gloss(german_text)
raw_gloss_line = " ".join(raw_gloss)
print("RAW GLOSS:", raw_gloss_line)

# Filter to what your SLP model can actually sign
filtered_gloss = [w for w in raw_gloss if w in SRC_VOCAB]
filtered_gloss_line = " ".join(filtered_gloss)

print("MODEL GLOSS:", filtered_gloss_line if filtered_gloss_line else "(NO KNOWN SIGNS)")

# =========================
# APPEND TO DATASET
# =========================
with open(TEST_TEXT_PATH, "a", encoding="utf-8") as f:
    f.write(german_text + "\n")

with open(TEST_GLOSS_PATH, "a", encoding="utf-8") as f:
    f.write(filtered_gloss_line + "\n")

with open(DEBUG_FULL_GLOSS_PATH, "a", encoding="utf-8") as f:
    f.write(raw_gloss_line + "\n")

print("\n✅ Written to dataset:")
print("test.text  →", TEST_TEXT_PATH)
print("test.gloss →", TEST_GLOSS_PATH)
print("debug_full_gloss →", DEBUG_FULL_GLOSS_PATH)
