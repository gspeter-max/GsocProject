# ⚡ QUICKSTART.md

This guide helps you run the **AutoTuning** pipeline in under 2 minutes.

---

## ✅ Prerequisites

* Python 3.9+
* `pip install -r requirements.txt`
* A [Hugging Face token](https://huggingface.co/settings/tokens)

---

## 📁 Directory Layout
---

## 🛠 Step-by-Step Guide

### 1. Edit `runner.py`

Set your custom config in `runner.py`:

```python
from backend.GlobalConfig import global_config
from backend.ModelingAndTuning import ModelLoadingAndTuning

config_obj = global_config(
    ModelName  = 'gpt2',
    DatasetPath = 'path/to/your/dataset.csv',# keep None for default dataset 
    FineTuningType = 'instruction_fine_tuning',
    ModelSeqMaxLength = 512,
    QuantizationType4Bit8Bit = None,
    ComputeMetricsList = ['accuracy'],
    PeftType  = 'LORA',
    SaveFormat  = 'gguf',
    ModelDir  = './hfConvertedModel',
    EvalSaveFormat  = 'json',
    FSDP = False,
    HfToken  = 'your_huggingface_token_here'
)

peft_config = config_obj.GetPeftConfig(
    task_type= 'CAUSAL_LM',
    r = 8,
    lora_alpha = 16,
    lora_dropout = 0.05,
    bias = "none",
    target_modules = ["c_attn", "c_proj"],
    inference_mode = False
)

training_arg = config_obj.GetTrainingArguments(
    output_dir= "./results",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3.0,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=1,
    eval_strategy="no",
    save_strategy="steps",
    save_total_limit=1,
    load_best_model_at_end=False,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    fp16=True,
    fsdp=False,
    label_names=['labels'],
    warmup_steps=500,
    lr_scheduler_type="linear",
    gradient_accumulation_steps=1,
    gradient_checkpointing=False,
    save_steps=500,
    logging_first_step=True,
    report_to=None
)

HyperparameterConfig = config_obj.get_full_config(
    PeftConfig = peft_config,
    TrainingArguments = training_arg
)

tuning = ModelLoadingAndTuning(HyperparameterConfig)
tuning.LoadItTrainIt()
```

---

### 2. Run Training

```bash
cd autotuning/
python -m backend.runner
```

---

### 3. Check Results

* Logs: `./logs`
* Metrics: `EvalResult.json` or `EvalResult.csv`
* Final model: `./hfConvertedModel/`

---

## 🧠 Tips

* If using HF dataset ID, just pass `DatasetPath='yahma/alpaca-cleaned'`
* GGUF export runs `llama.cpp` converter under the hood
* Use `fp16=False` if you face memory issues


Open an issue at [https://github.com/yourrepo/autotuning/issues](https://github.com/yourrepo/autotuning/issues)
