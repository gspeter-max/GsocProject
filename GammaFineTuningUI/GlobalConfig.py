
from typing import Optional, List, Union

class GetIt:
    def __init__(
            self, 
            ModelName : str = 'gpt2', 
            QuantizationType4Bit8Bit : Union[str,bool] = False, 
            ComputeMetrics : Union[list, str ], 
            PeftType : str = 'LORA'
            ):

        self.ModelName = ModelName, 
        self.QuantizationType4Bit8Bit = QuantizationType4Bit8Bit
        self.ComputeMetrics = ComputeMetrics 
        self.PeftType = PeftType

    def GetTokenizationConfig(
            TokenizerPadding : Union[str,bool] = 'max_length', 
            TokenizerMaxLength : int = 128, 
            TokenizerTruncation : bool = True
            
        ) -> dict:
    # Default config
        default_config = {
            'padding' : True, 
            'max_length': None, 
            'truncation' : True
                }


    # User override
        user_config = {
            'padding' : TokenizerPadding, 
            'max_length' : TokenizerMaxLength, 
            'truncation': TokenizerTruncation
                }


    # Fill in None with default
        final_config = {
            key: user_config[key] if user_config[key] is not None else default_config[key]
            for key in default_config
        }

        return final_config

    def GetPeftConfig(
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
            "evaluation_strategy": "epoch",
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
            "report_to": ["wandb"]  # or ["wandb"]
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

    
    def __call__(
        self, 
        TokenizationConfig = None, 
        PeftConfig = None, 
        TrainingArguments = None
        ): 
        return {
        'ModelName' : self.ModelName,
        'ComputeMetrics' : self.ComputeMetrics,
        'QuantizationType4Bit8Bit' : self.QuantizationType4Bit8Bit, 
        'TokenizationConfig' : TokenizationConfig if not None else self.GetTokenizationConfig(),  
        'PeftConfig' : PeftConfig if not None else self.GetPeftConfig(), 
        'TrainingArguments' : TrainingArguments if not None else self.GetTrainingArguments()
            }

