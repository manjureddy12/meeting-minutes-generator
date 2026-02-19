"""
summarizer.py — Loads the HuggingFace summarization model and generates text.

Key concepts:
- HuggingFace Pipeline: A high-level API that wraps model loading, tokenization,
  inference, and decoding into a single callable object
- LangChain HuggingFacePipeline: Wraps HuggingFace pipeline into LangChain's
  standard LLM interface, making it compatible with LangChain chains

Why FLAN-T5?
- Free and open-source
- Good instruction-following capabilities
- Reasonable quality on summarization tasks
- ~250MB — manageable on CPU
- Specifically fine-tuned for tasks expressed as instructions
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_huggingface import HuggingFacePipeline

import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import SUMMARIZATION_MODEL_NAME

# Global variable — load model only once
_llm = None


def load_summarization_model() -> HuggingFacePipeline:
    """
    Load the summarization model and wrap it in LangChain's interface.

    This function:
    1. Downloads model weights from HuggingFace Hub (first run only, ~250MB)
    2. Loads tokenizer (converts text ↔ token IDs)
    3. Loads model weights into memory
    4. Creates HuggingFace pipeline
    5. Wraps in LangChain HuggingFacePipeline

    Returns:
        LangChain-compatible LLM that can be used in chains

    What is a Tokenizer?
    Models don't read words — they read "tokens" (subword units).
    "Summarization" → ["Sum", "mar", "ization"] → [1432, 567, 2891]
    The tokenizer handles this text ↔ token conversion.
    """
    global _llm

    if _llm is not None:
        return _llm

    print(f"⏳ Loading summarization model: {SUMMARIZATION_MODEL_NAME}")
    print("   First run downloads model weights (~250MB for flan-t5-base)")
    print("   This may take 1-3 minutes...")

    # Check if GPU is available (for faster inference)
    device = 0 if torch.cuda.is_available() else -1
    device_name = "GPU" if device == 0 else "CPU"
    print(f"   Running on: {device_name}")

    # ── Step 1: Load Tokenizer ──────────────────────────────────────────────
    # The tokenizer is model-specific — it knows exactly how flan-t5 expects input
    tokenizer = AutoTokenizer.from_pretrained(SUMMARIZATION_MODEL_NAME)

    # ── Step 2: Load Model ──────────────────────────────────────────────────
    # AutoModelForSeq2SeqLM automatically selects the right model class
    # Seq2Seq = Sequence-to-Sequence (reads full input, generates full output)
    # This is different from causal/decoder-only models (GPT style) which
    # generate token by token from left to right only
    model = AutoModelForSeq2SeqLM.from_pretrained(
    SUMMARIZATION_MODEL_NAME,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)


    # ── Step 3: Create HuggingFace Pipeline ─────────────────────────────────
    # A "pipeline" combines: tokenizer + model + post-processing into one object
    # You call it with text, it returns text
    hf_pipeline = pipeline(
        task="text2text-generation",   # Input text → Output text (for T5/BART)
        model=model,
        tokenizer=tokenizer,
        device=device,

        # Generation parameters — control how text is generated:
        max_new_tokens=512,    # Maximum length of generated output (in tokens)
        do_sample=False,       # Greedy decoding (deterministic, consistent output)
                               # Set to True for more creative/varied output
        temperature=0.3,       # Creativity (only relevant if do_sample=True)
        repetition_penalty=1.2, # Penalize repeating the same phrases
        no_repeat_ngram_size=3, # Prevent repeating 3-grams (three-word phrases)
        truncation=True
    )

    # ── Step 4: Wrap in LangChain Interface ─────────────────────────────────
    # LangChain chains expect an LLM with a .invoke() method
    # HuggingFacePipeline provides this standard interface
    _llm = HuggingFacePipeline(pipeline=hf_pipeline)

    print(f"✅ Summarization model loaded and ready")
    return _llm


def generate_summary(llm: HuggingFacePipeline, context: str, query: str) -> str:

    prompt = f"""Instruction:
Generate professional meeting minutes from the transcript.

Format:

Meeting Overview:
Summary of meeting purpose.

Key Decisions:
• Decision 1
• Decision 2

Action Items:
• Task — Owner — Deadline

Discussion Summary:
Summary of discussion.

Next Steps:
Future actions.

Transcript:
{context}

Output:
"""

    print("⏳ Generating meeting minutes...")

    result = llm.invoke(prompt)

    if hasattr(result, "content"):
        result = result.content

    result = str(result).strip()

    print(f"✅ Generated {len(result)} characters")

    return result




    print(f"⏳ Generating mee ting minutes... (this takes 15-60 seconds on CPU)")

    # Call the LLM — this is where the AI "thinks"
    # .invoke() is LangChain's standard method (replaces deprecated .predict())
    result = llm.invoke(prompt)

# Handle LangChain response safely
    if hasattr(result, "content"):
        result = result.content

    result = str(result).strip()

    print(f"✅ Generated {len(result)} characters of meeting minutes")

    return result
