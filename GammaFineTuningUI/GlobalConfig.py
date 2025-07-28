from typing import Optional, List, Union

class GetIt:
    def __init__(
            self,
            ModelName : str = 'gpt2',
            QuantizationType4Bit8Bit : Union[str,bool] = False,
            ComputeMetricsList : Union[list, str ] = None,
            PeftType : str = 'LORA', 
            SaveFormat : str = None, 
            ModelDir : str = None
            ):

        self.ModelName = ModelName
        self.QuantizationType4Bit8Bit = QuantizationType4Bit8Bit
        self.ComputeMetrics = ComputeMetricsList
        self.PeftType = PeftType
        self.SaveFormat = SaveFormat
        self.ModelDir = ModelDir

        if self.ModelDir:
            if self.SaveFormat is None :
                raise RuntimeError('If "model_dir" is provided , you must also specify "SaveFormat"') 
        
        if self.SaveFormat.lower() not in ('tensorflow','torch','gguf',None): 
            raise NotImplemented('SaveFormat must be in "( tensorflow , torch , gguf )"') 


    @staticmethod
    def GetTokenizationConfig(
            TokenizerPadding : Union[str,bool] = 'max_length',
            TokenizerMaxLength : int = 128,
            TokenizerTruncation : bool = True

        ) -> dict:
        return {
            'padding' : TokenizerPadding,
            'max_length' : TokenizerMaxLength,
            'truncation': TokenizerTruncation
                }

    @staticmethod
    def GetPeftConfig(
        task_type: str = 'CAUSAL_LM',
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        bias: str = "none",
        target_modules: list = ["q_proj", "v_proj"],
        inference_mode: bool = False
    ) -> dict:
        return {
            "task_type": task_type,
            "r": r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "bias": bias,
            "target_modules": target_modules,
            "inference_mode": inference_mode
        }

    @staticmethod
    def GetTrainingArguments(
        output_dir: str = "./results",
        per_device_train_batch_size: int = 8,
        per_device_eval_batch_size: int = 8,
        num_train_epochs: float = 3.0,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        logging_dir: str = "./logs",
        logging_steps: int = 50,
        eval_strategy: str = "steps",
        save_strategy: str = "steps",
        save_total_limit: int = 2,
        load_best_model_at_end: bool = False,
        metric_for_best_model: str = "accuracy",
        greater_is_better: bool = True,
        fp16: bool = True,
        warmup_steps: int = 500,
        lr_scheduler_type: str = "linear",
        gradient_accumulation_steps: int = 1,
        gradient_checkpointing: bool = False,
        save_steps: int = 500,
        logging_first_step: bool = True,
        report_to: list = None
    ) -> dict:
        return {
            "output_dir": output_dir,
            "per_device_train_batch_size": per_device_train_batch_size,
            "per_device_eval_batch_size": per_device_eval_batch_size,
            "num_train_epochs": num_train_epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "logging_dir": logging_dir,
            "logging_steps": logging_steps,
            "eval_strategy": 'no',
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


    def __call__(
        self,
        TokenizationConfig = None,
        PeftConfig = None,
        TrainingArguments = None
        ):
        return {
        'ModelName' : self.ModelName,
        'ComputeMetricsList' : self.ComputeMetricsList,
        'QuantizationType4Bit8Bit' : self.QuantizationType4Bit8Bit,
        'SaveFormat' : self.SaveFormat, 
        'ModelDir' : self.ModelDir,
        'TokenizationConfig' : TokenizationConfig if TokenizationConfig is not None else GetIt.GetTokenizationConfig(),
        'PeftConfig' : PeftConfig if PeftConfig is not None else GetIt.GetPeftConfig(),
        'TrainingArguments' : TrainingArguments if TrainingArguments is not None else GetIt.GetTrainingArguments()
            }
