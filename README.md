---
license: apache-2.0
tags:
- unsloth
- trl
- sft
- medical
- healthcare
- clinical-reasoning
- question-answering
- diagnosis
language:
- en
library_name: transformers
datasets:
- FreedomIntelligence/medical-o1-reasoning-SFT
---

# 🏥 Medical-Diagnosis-COT-DeepSeek V1

<div align="center">

![Medical AI](https://img.shields.io/badge/Domain-Medical%20AI-blue)
![License](https://img.shields.io/badge/License-Apache%202.0-green)
![Language](https://img.shields.io/badge/Language-English-orange)

</div>

## 📋 Overview

**Medical-Diagnosis-COT-DeepSeek V1** is an advanced medical language model fine-tuned for clinical reasoning, diagnostic support, and treatment planning. Built on state-of-the-art transformer architecture and trained on medical reasoning datasets, this model demonstrates strong capabilities in understanding complex medical scenarios and providing structured clinical insights.

### Key Capabilities

- 🔍 **Clinical Reasoning**: Systematic analysis of medical cases with step-by-step reasoning
- 🩺 **Diagnostic Support**: Differential diagnosis generation based on symptoms and clinical findings
- 💊 **Treatment Planning**: Evidence-based treatment recommendations and care protocols
- 📊 **Medical Q&A**: Comprehensive answers to medical questions with clinical context
- 🧬 **Disease Knowledge**: Extensive understanding of pathophysiology, epidemiology, and clinical presentations

## 🛠️ Model Details

- **Base Architecture**: Transformer-based causal language model
- **Fine-tuning Framework**: Unsloth + TRL (Transformer Reinforcement Learning)
- **Training Method**: Supervised Fine-Tuning (SFT)
- **Training Dataset**: FreedomIntelligence/medical-o1-reasoning-SFT
- **Domain Specialization**: Medical/Healthcare/Clinical
- **Language**: English
- **Model Type**: Causal Language Model for Medical Question Answering

## 📦 Installation

### Required Dependencies

```notebook-python
# Core libraries

!pip install unsloth # install unsloth
!pip install --force-reinstall --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git # Also get the latest version Unsloth!

!pip install torch torchaudio transformers accelerate
!pip install trl datasets huggingface_hub wandb

# Quantization and optimization
# Install bitsandbytes
!pip install -U bitsandbytes

# Optional: For better performance
!pip install sentencepiece protobuf
```


## 🚀 Usage

### Google Colab Setup

# Step1: Import necessary libraries
```notebook-python
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from unsloth import is_bfloat16_supported
from huggingface_hub import login
from transformers import TrainingArguments
from datasets import load_dataset
import wandb
```

# Step2: Check HF token
```notebook-python
from google.colab import userdata
hf_token = userdata.get('HF_TOKEN')
login(hf_token)
```

# Optional: Check CUDA And GPU availability
```notebook-python
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
```

# Step3: Setup The Model
```notebook-python
model_name = "NeoAivara/AI_Doctor_V1"
max_sequence_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_sequence_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token = hf_token
)

```

# Step4: Setup system prompt
```notebook-python
prompt = f"### Question: {question}\n### Answer:"
```

# Step5: Run Inference on the model

```notebook-python
# Define a test question
question = """What are the common symptoms of pneumonia?"""

FastLanguageModel.for_inference(model_lora)

# Tokenize the input
inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")

# Generate a response
outputs = model_lora.generate (
    input_ids = inputs.input_ids,
    attention_mask = inputs.attention_mask,
    max_new_tokens = 1200,
    use_cache = True
)

# Decode the response tokens back to text
response = tokenizer.batch_decode(outputs)
answer = response.split("### Answer:")[-1].strip()

print(answer)
```

## 🎯 Example Use Cases

### 1. Symptom Analysis
```python
question = "A patient presents with fever, cough, and night sweats for 3 weeks. What should be considered?"
```

### 2. Treatment Recommendations
```python
question = "What is the first-line treatment for uncomplicated community-acquired pneumonia in adults?"
```

### 3. Diagnostic Reasoning
```python
question = "Explain the diagnostic approach for a patient with suspected acute appendicitis."
```

### 4. Medical Education
```python
question = "Explain the pathophysiology of type 2 diabetes mellitus."
```

## ⚙️ Generation Parameters

Recommended parameters for different use cases:

| Parameter | Diagnostic | Educational | Treatment Planning |
|-----------|-----------|-------------|-------------------|
| `temperature` | 0.5-0.7 | 0.7-0.8 | 0.6-0.7 |
| `top_p` | 0.85-0.9 | 0.9-0.95 | 0.85-0.9 |
| `max_new_tokens` | 512-768 | 768-1024 | 512-768 |
| `repetition_penalty` | 1.1 | 1.0 | 1.1 |

## ⚠️ Important Disclaimers

### Medical Disclaimer

**🚨 CRITICAL: FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY 🚨**

This AI model is designed for:
- ✅ Medical education and training
- ✅ Research and development
- ✅ Clinical decision support as an **assistive tool**
- ✅ Medical knowledge exploration

This AI model is **NOT** intended for:
- ❌ Direct patient care without physician oversight
- ❌ Replacement of professional medical judgment
- ❌ Emergency medical decisions
- ❌ Self-diagnosis or self-treatment

### Key Limitations

1. **Not a Licensed Healthcare Provider**: This model cannot legally practice medicine or provide medical advice
2. **Potential for Errors**: May generate incorrect, incomplete, or outdated medical information
3. **No Clinical Validation**: Not validated in real-world clinical settings or approved by regulatory bodies
4. **Bias and Training Data**: Limited by the scope and potential biases in training data
5. **No Patient Context**: Cannot account for individual patient factors, allergies, drug interactions, or complete medical history
6. **Liability**: Users assume all responsibility for clinical decisions made using this tool

### Recommended Usage

- Always verify information with current medical literature and guidelines
- Use in conjunction with clinical expertise and professional judgment
- Consult qualified healthcare professionals for all medical decisions
- Follow institutional protocols and regulatory requirements
- Document AI assistance appropriately in medical records per institutional policy

## 📊 Performance Notes

- **Inference Speed**: Optimized for GPU acceleration (CUDA)
- **Memory Requirements**: ~8-16GB GPU memory recommended for fp16
- **Quantization**: Supports 8-bit quantization via bitsandbytes for lower memory usage

## 🤝 Contributing

Contributions to improve the model, documentation, or use cases are welcome. Please ensure all contributions maintain the educational and research focus of this project.

## 📄 License

This model is released under the **Apache 2.0 License**. See LICENSE file for details.

## 🙏 Acknowledgments

- **Training Framework**: Unsloth and TRL
- **Dataset**: FreedomIntelligence/medical-o1-reasoning-SFT
- **Community**: Hugging Face and the open-source medical AI community

## 📧 Contact & Support

For questions, issues, or collaboration inquiries, please open an issue on the model repository.

---

<div align="center">

**Remember: AI is a tool to augment, not replace, human medical expertise.**

*Always prioritize patient safety and professional medical judgment.*

</div>
