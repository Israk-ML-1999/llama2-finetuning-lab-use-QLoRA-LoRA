# Llama2 Fine-Tuning Lab

A collection of notebooks for fine-tuning Llama 2 models using QLoRA, PEFT, and Hugging Face Transformers.

This repository contains practical implementations for:
- Llama 2 instruction tuning
- Chatbot fine-tuning
- QLoRA-based efficient training
- LoRA adapter merging
- Text generation inference pipelines

---

# Repository Structure

```bash
.
├── LLAMA2_fine_tune.ipynb
├── LLma_fine_tune__chat__bot.ipynb
└── README.md
```

---

# Features

- QLoRA fine-tuning
- PEFT (Parameter Efficient Fine-Tuning)
- 4-bit quantization with BitsAndBytes
- Hugging Face Transformers integration
- Chatbot training workflow
- LoRA adapter merging
- Efficient GPU memory usage
- Text generation pipeline

---

# Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- PEFT
- TRL
- BitsAndBytes
- Accelerate
- Google Colab

---

# Model

Base model used:

- Llama 2 7B Chat
- NousResearch/Llama-2-7b-chat-hf

---

# Training Method

This project uses:

- QLoRA (Quantized Low-Rank Adaptation)
- LoRA adapters
- 4-bit quantization
- FP16 training

This allows large language models to be fine-tuned with lower GPU memory requirements.

---

# Notebooks

## 1. LLAMA2_fine_tune.ipynb

Complete workflow for:
- Loading datasets
- Configuring QLoRA
- Fine-tuning Llama 2
- Saving adapters
- Merging LoRA weights
- Running inference

---

## 2. LLma_fine_tune__chat__bot.ipynb

Chatbot-focused fine-tuning notebook with:
- Instruction tuning
- Conversational dataset training
- Text generation pipeline
- Chat-style inference

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your-username/llama2-finetuning-lab.git
cd llama2-finetuning-lab
```

Install dependencies:

```bash
pip install -q accelerate peft bitsandbytes transformers trl datasets
```

---

# Example Training Flow

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(model_name)

lora_config = LoraConfig(
    r=16,
    lora_alpha=64,
    lora_dropout=0.1,
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
```

---

# Inference Example

```python
from transformers import pipeline

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

result = generator("Explain LoRA fine-tuning")
print(result[0]["generated_text"])
```

---

# Requirements

- Python 3.10+
- CUDA GPU recommended
- Google Colab / Kaggle / Local GPU setup

---
