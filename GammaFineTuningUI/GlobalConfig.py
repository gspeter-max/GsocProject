

class GlobalConifg: 
    def __init__():
        pass 
    def GetTokenizationConfig(
        tokenizer_class: str = None,
        unk_token: str = None,
        bos_token: str = None,
        eos_token: str = None,
        pad_token: str = None,
        add_bos_token: bool = None,
        add_eos_token: bool = None,
        do_lower_case: bool = None,
        model_max_length: int = None,
        clean_up_tokenization_spaces: bool = None,
        special_tokens_map_file: str = None,
        tokenizer_file: str = None
    ) -> dict:
    # Default config
        default_config = {
            "tokenizer_class": "GPT2Tokenizer",
            "unk_token": "<unk>",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<pad>",
            "add_bos_token": False,
            "add_eos_token": False,
            "do_lower_case": False,
            "model_max_length": 2048,
            "clean_up_tokenization_spaces": True,
            "special_tokens_map_file": "special_tokens_map.json",
            "tokenizer_file": "tokenizer.json"
        }

    # User override
        user_config = {
            "tokenizer_class": tokenizer_class,
            "unk_token": unk_token,
            "bos_token": bos_token,
            "eos_token": eos_token,
            "pad_token": pad_token,
            "add_bos_token": add_bos_token,
            "add_eos_token": add_eos_token,
            "do_lower_case": do_lower_case,
            "model_max_length": model_max_length,
            "clean_up_tokenization_spaces": clean_up_tokenization_spaces,
            "special_tokens_map_file": special_tokens_map_file,
            "tokenizer_file": tokenizer_file
        }

    # Fill in None with default
        final_config = {
            key: user_config[key] if user_config[key] is not None else default_config[key]
            for key in default_config
        }

        return final_config

    def GetPeftConfig(
        peft_type: str = None,
        task_type: str = None,
        r: int = None,
        lora_alpha: int = None,
        lora_dropout: float = None,
        bias: str = None,
        target_modules: list = None,
        inference_mode: bool = None
    ) -> dict:
        # Default PEFT configuration
        default_config = {
            "peft_type": "LORA",                 # LoRA, AdaLoRA, PrefixTuning, etc.
            "task_type": "CAUSAL_LM",            # SEQ_CLS, TOKEN_CLS, CAUSAL_LM, etc.
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "bias": "none",                      # Options: "none", "all", "lora_only"
            "target_modules": ["q_proj", "v_proj"],  # Depends on model
            "inference_mode": False
        }

        # User-specified overrides
        user_config = {
            "peft_type": peft_type,
            "task_type": task_type,
            "r": r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "bias": bias,
            "target_modules": target_modules,
            "inference_mode": inference_mode
        }

        # Merge: use user value if not None, else default
        final_config = {
            key: user_config[key] if user_config[key] is not None else default_config[key]
            for key in default_config
        }

        return final_config

    def GetTrainingArguments(
        output_dir: str = None,
        per_device_train_batch_size: int = None,
        per_device_eval_batch_size: int = None,
        num_train_epochs: float = None,
        learning_rate: float = None,
        weight_decay: float = None,
        logging_dir: str = None,
        logging_steps: int = None,
        evaluation_strategy: str = None,
        save_strategy: str = None,
        save_total_limit: int = None,
        load_best_model_at_end: bool = None,
        metric_for_best_model: str = None,
        greater_is_better: bool = None,
        fp16: bool = None,
        warmup_steps: int = None,
        lr_scheduler_type: str = None,
        gradient_accumulation_steps: int = None,
        gradient_checkpointing: bool = None,
        save_steps: int = None,
        logging_first_step: bool = None,
        report_to: list = None
    ) -> dict:
        default_config = {
            "output_dir": "./results",
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 8,
            "num_train_epochs": 3.0,
            "learning_rate": 5e-5,
            "weight_decay": 0.01,
            "logging_dir": "./logs",
            "logging_steps": 50,
            "evaluation_strategy": "steps",
            "save_strategy": "steps",
            "save_total_limit": 2,
            "load_best_model_at_end": True,
            "metric_for_best_model": "accuracy",
            "greater_is_better": True,
            "fp16": True,
            "warmup_steps": 500,
            "lr_scheduler_type": "linear",
            "gradient_accumulation_steps": 1,
            "gradient_checkpointing": False,
            "save_steps": 500,
            "logging_first_step": True,
            "report_to": ["tensorboard"]  # or ["wandb"]
        }

        user_config = {
            "output_dir": output_dir,
            "per_device_train_batch_size": per_device_train_batch_size,
            "per_device_eval_batch_size": per_device_eval_batch_size,
            "num_train_epochs": num_train_epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "logging_dir": logging_dir,
            "logging_steps": logging_steps,
            "evaluation_strategy": evaluation_strategy,
            "save_strategy": save_strategy,
            "save_total_limit": save_total_limit,
            "load_best_model_at_end": load_best_model_at_end,
            "metric_for_best_model": metric_for_best_model,
            "greater_is_better": greater_is_better,
            "fp16": fp16,
            "warmup_steps": warmup_steps,
            "lr_scheduler_type": lr_scheduler_type,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "gradient_checkpointing": gradient_checkpointing,
            "save_steps": save_steps,
            "logging_first_step": logging_first_step,
            "report_to": report_to
        }

        final_config = {
            key: user_config[key] if user_config[key] is not None else default_config[key]
            for key in default_config
        }

        return final_config

from typing import Optional, List, Union

def GetTrainerConfig(
    output_dir: str = None,
    per_device_train_batch_size: int = None,
    per_device_eval_batch_size: int = None,
    num_train_epochs: float = None,
    learning_rate: float = None,
    weight_decay: float = None,
    logging_dir: str = None,
    logging_steps: int = None,
    evaluation_strategy: str = None,
    eval_steps: int = None,
    save_strategy: str = None,
    save_steps: int = None,
    save_total_limit: int = None,
    load_best_model_at_end: bool = None,
    metric_for_best_model: str = None,
    greater_is_better: bool = None,
    fp16: bool = None,
    bf16: bool = None,
    warmup_steps: int = None,
    warmup_ratio: float = None,
    lr_scheduler_type: str = None,
    gradient_accumulation_steps: int = None,
    gradient_checkpointing: bool = None,
    logging_first_step: bool = None,
    report_to: Union[List[str], str] = None,
    optim: str = None,
    seed: int = None,
    dataloader_num_workers: int = None,
    disable_tqdm: bool = None,
    remove_unused_columns: bool = None,
    label_names: List[str] = None,
    group_by_length: bool = None,
    resume_from_checkpoint: bool = None,
    deepspeed: Optional[str] = None,
) -> dict:
    """
    Get configuration for Hugging Face Trainer with sensible defaults.

    Returns:
        dict: Dictionary of training arguments compatible with transformers.TrainingArguments
    """

    default_config = {
        "output_dir": "./results",
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "num_train_epochs": 3.0,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "logging_dir": "./logs",
        "logging_steps": 500,
        "evaluation_strategy": "steps",
        "eval_steps": 500,
        "save_strategy": "steps",
        "save_steps": 500,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "loss",
        "greater_is_better": False,
        "fp16": False,
        "bf16": False,
        "warmup_steps": 0,
        "warmup_ratio": 0.0,
        "lr_scheduler_type": "linear",
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": False,
        "logging_first_step": False,
        "report_to": ["tensorboard"],
        "optim": "adamw_torch",
        "seed": 42,
        "dataloader_num_workers": 0,
        "disable_tqdm": False,
        "remove_unused_columns": True,
        "label_names": None,
        "group_by_length": False,
        "resume_from_checkpoint": None,
        "deepspeed": None,
    }

    user_config = {
        k: v for k, v in locals().items()
        if k not in ['default_config', 'user_config'] and v is not None
    }

    final_config = {
        key: user_config.get(key, default_config[key])
        for key in default_config
    }

    # Handle report_to which can be either string or list
    if isinstance(final_config["report_to"], str):
        final_config["report_to"] = [final_config["report_to"]]

    return final_config
    
    def __call__(
        self, 
        TokenizationConfig = None, 
        PeftConfig = None, 
        TrainingArguments = None,
        TrainerConfig = None
        ): 
        return {
        'TokenizationConfig' : TokenizationConfig if not None else self.GetTokenizationConfig(),  
        'PeftConfig' : PeftConfig if not None else self.GetPeftConfig(), 
        'TrainingArguments' : TrainingArguments if not None else self.GetTrainingArguments(), 
        'TrainerConfig' : TrainerConfig if not None else self.GetTrainerConfig()
                }

