# 🚀 AutoTuning Framework Documentation

This system is a **modular, automated, and extensible fine-tuning pipeline** for Hugging Face models. It supports instruction tuning, code generation, chat fine-tuning, question answering, RAG, quantization, PEFT with LoRA, evaluation, and model export in various formats (Torch, TensorFlow, GGUF).

---

## 🔧 1. Configuration Module (`GlobalConfig.py`)

### Class: `global_config`

Initializes and validates user configuration.

**Key Parameters:**

* `ModelName`: Pretrained model name (e.g., `gpt2`)
* `FineTuningType`: Type of fine-tuning (`instruction_fine_tuning`, `code_generation`, `chat_fine_tuning`, `question_answering`, `rag_fine_tuning`)
* `DatasetPath`: Path to the dataset or HuggingFace dataset ID
* `ModelSeqMaxLength`: Sequence truncation length
* `QuantizationType4Bit8Bit`: Supports `4bit` or `8bit`
* `PeftType`: Type of parameter-efficient tuning (`LORA`, `QLORA`)
* `SaveFormat`: Final model format (`torch`, `tensorflow`, `gguf`)
* `ModelDir`: Output directory for model
* `EvalSaveFormat`: Format for saving eval results (`json`, `csv`)
* `FSDP`: Enable FSDP for multi-GPU training
* `HfToken`: Required Hugging Face token

**Utility Methods:**

* `GetPeftConfig()`
* `GetTrainingArguments()`
* `GetFSDP()`
* `get_full_config()`

---

## 📦 2. Dataset Loading (`DatasetUpLoading.py`)

### Class: `UploadDataset`

Supports dynamic loading and preprocessing for multiple fine-tuning types:

* `instruction_fine_tuning`: Expects fields `instruction`, `input`, `output`
* `code_generation`: Expects `prompt`, `completion`
* `chat_fine_tuning`: Expects `messages` with roles and values
* `question_answering`, `rag_fine_tuning`: Expects `context`, `question`, `answer`

**Built-in datasets (if path not provided):**

* Alpaca
* Glaive Code Assistant
* ShareGPT
* MS MARCO
* Mou3az QA Choices

All modes include mapping functions to tokenize text and create labels based on `[sep]` token.

---

## 🧠 3. Model Tuning (`ModelingAndTuning.py`)

### Class: `ModelLoadingAndTuning`

Main training orchestration logic.

**Steps:**

1. Load dataset using `UploadDataset`
2. Load tokenizer and model
3. Apply quantization (4bit or 8bit)
4. Add PEFT using LoRA (`get_peft_model`)
5. Configure training args (`Trainer`)
6. Add evaluation callback (accuracy, f1, perplexity)
7. Train with `trainer.train()`
8. Merge LoRA layers back into base model
9. Save in specified format using `ConvertModel`
10. Save evaluation results to `.json` or `.csv`

---

## 💾 4. Model Conversion (`GetModel.py`)

### Class: `ConvertModel`

Exports trained model to chosen format:

* **`torch`**: Saves using `save_pretrained`
* **`tensorflow`**: Converts from PyTorch using `from_pt=True`
* **`gguf`**:

  * Installs dependencies
  * Uses `llama.cpp/convert_hf_to_gguf.py`
  * Output saved as `output_gguf.gguf`

---

## 🧪 5. Execution Runner (`runner.py`)

Entry point for training pipeline.

**Usage:**

```bash
cd ./GsocProject/autotuning/
python -m backend.runner
```

**Runner Flow:**

* Instantiate `global_config`
* Generate config using `get_full_config`
* Create `ModelLoadingAndTuning` instance
* Call `LoadItTrainIt()` to run training + evaluation + export

---

## ✅ Supported Features

| Feature                | Support |
| ---------------------- | ------- |
| HF Model Loading       | ✅ Yes   |
| Instruction Tuning     | ✅ Yes   |
| Chat Tuning            | ✅ Yes   |
| Code Generation        | ✅ Yes   |
| QA & RAG               | ✅ Yes   |
| PEFT (LoRA/QLORA)      | ✅ Yes   |
| Quantization (4/8bit)  | ✅ Yes   |
| FSDP (Multi-GPU)       | ✅ Yes   |
| Export: Torch          | ✅ Yes   |
| Export: TensorFlow     | ✅ Yes   |
| Export: GGUF           | ✅ Yes   |
| Evaluation: Accuracy   | ✅ Yes   |
| Evaluation: F1         | ✅ Yes   |
| Evaluation: Perplexity | ✅ Yes   |

---

## 📌 Final Notes

* Make sure you supply a **valid Hugging Face token**
* Compatible with **Colab**, **multi-GPU**, **32-bit machines**, and **CLI-based environments**
* Future enhancements: UI, Auto Layer Detection, Multi-model fusion

---

For contributions, feature requests, or issues, open a ticket at `https://github.com/yourrepo/autotuning`

Built with ❤️ for large-scale LLM fine-tuning.
