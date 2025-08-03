from typing import Optional, List, Union
import os 

class global_config:
    def __init__(
            self,
            DatasetPath : str = None,
            FineTuneType : str = None,
            ModelSeqMaxLength : int = None,
            ModelName : str = 'gpt2',
            QuantizationType4Bit8Bit : Union[str,None] = None,
            ComputeMetricsList : Union[list, str ] = None,
            PeftType : str = 'LORA',
            SaveFormat : str = None, 
            ModelDir : str = None,  
            EvalSaveFormat : str = None, 
            FSDP : bool = False,
            HfToken : str = None 
            ):
                self.ModelName = ModelName
                self.QuantizationType4Bit8Bit = QuantizationType4Bit8Bit
                self.ComputeMetricsList = ComputeMetricsList
                self.PeftType = PeftType
                self.DatasetPath  = DatasetPath
                self.FineTuneType = FinetuneType
                self.ModelSeqMaxLength = ModelSeqMaxLength
                self.HfToken = HfToken
                self.EvalSaveFormat = EvalSaveFormat
                self.SaveFormat = SaveFormat
                self.ModelDir = ModelDir
                self.FSDP = FSDP 

                if self.PeftType == 'qlora':
                    self.QuantizationType4Bit8Bit = '4bit'
                
                if self.FineTuneType is None:
                    raise ValueError(
                        "FinetuneType is None, which is not allowed. Please select from the following:\n"
                        "['instruction_fine_tuning', 'code_generation', 'chat_fine_tuning', 'question_answering', 'rag_fine_tuning']"
                    )

                if self.ModelDir:
                    if self.SaveFormat is None :
                        raise RuntimeError('If "model_dir" is provided , you must also specify "SaveFormat"')
        
                if self.SaveFormat not in ('tensorflow','torch','gguf',None):
                    raise NotImplemented('SaveFormat must be in "( tensorflow , torch , gguf )"')

    @staticmethod
    def GetFSDP(
            fsdp_auto_wrap_policy: str = 'TRANSFORMER_BASED_WRAP',
            fsdp_backward_prefetch_policy: str =  'BACKWARD_PRE',
            fsdp_forward_prefetch: bool = False,
            fsdp_cpu_ram_efficient_loading:bool = True, 
            fsdp_offload_params: bool =  False,
            fsdp_sharding_strategy: str = 'FULL_SHARD',
            fsdp_state_dict_type:str =  'SHARDED_STATE_DICT',
            fsdp_sync_module_states: bool =  True,
            fsdp_transformer_layer_cls_to_wrap:str = 'GPT2Layers',
            fsdp_use_orig_params: bool = True
            ):
        return_dict = {
                'fsdp_auto_wrap_policy' : fsdp_auto_wrap_policy,
                'fsdp_backward_prefetch_policy': fsdp_backward_prefetch_policy,
                'fsdp_forward_prefetch': fsdp_forward_prefetch,
                'fsdp_cpu_ram_efficient_loading': fsdp_cpu_ram_efficient_loading,
                'fsdp_offload_params': fsdp_offload_params,
                'fsdp_sharding_strategy': fsdp_sharding_strategy,
                'fsdp_state_dict_type': fsdp_state_dict_type,
                'fsdp_sync_module_states': fsdp_sync_module_states,
                'fsdp_transformer_layer_cls_to_wrap': fsdp_transformer_layer_cls_to_wrap,
                'fsdp_use_orig_params': fsdp_use_orig_params 
        }
        return return_dict 

    @staticmethod
    def GetPeftConfig(
        task_type: str = 'CAUSAL_LM',
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        bias: str = "none",
        target_modules: list = None,
        inference_mode: bool = False
    ) -> dict:
        logger.info(f'''
            make sure ,  {target_module} is the part of model layers  
        ''')
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
        logging_steps: int = 1,
        eval_strategy: str = "steps",
        save_strategy: str = "steps",
        save_total_limit: int = 2,
        load_best_model_at_end: bool = False,
        metric_for_best_model: str = "accuracy",
        greater_is_better: bool = True,
        fp16: bool = True,
        fsdp : bool = False,
        label_names : Optional[list[str]] = None,
        fsdp_config = None,
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
            "eval_strategy": eval_strategy,
            "save_strategy": save_strategy,
            "save_total_limit": save_total_limit,
            "load_best_model_at_end": load_best_model_at_end,
            "metric_for_best_model": metric_for_best_model,
            "greater_is_better": greater_is_better,
            "fp16": fp16,
            "warmup_steps": warmup_steps,
            "label_names" : label_names,
            "lr_scheduler_type": lr_scheduler_type,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "gradient_checkpointing": gradient_checkpointing,
            "save_steps": save_steps,
            "logging_first_step": logging_first_step,
            "report_to": report_to
        }


    def get_full_config(
        self,
        PeftConfig = None,
        TrainingArguments = None, 
        fsdp_config = None
        ):
            return_dict =  {
                'ModelName' : self.ModelName,
                'ComputeMetricsList' : self.ComputeMetricsList,
                'QuantizationType4Bit8Bit' : self.QuantizationType4Bit8Bit,
                'SaveFormat' : self.SaveFormat,
                'HfToken' : self.HfToken,
                'Dataset_path': self.Dataset_path,
                'FineTuneType': self.FineTuneType,
                'ModelSeqMaxLength': self.ModelSeqMaxLength,
                'ModelDir' : self.ModelDir,
                'EvalSaveFormat' : self.EvalSaveFormat,
                'PeftConfig' : PeftConfig if PeftConfig is not None else global_config.GetPeftConfig(),
                'TrainingArguments' : TrainingArguments if TrainingArguments is not None else global_config.GetTrainingArguments()
            }
            
            if self.FSDP == True:
                final_fsdp_config = global_config.GetFSDP
                return_dict['TrainingArguments']['fsdp'] = True
                if return_dict['TrainingArguments']['fsdp_config'] is None:
                    if fsdp_config is None:
                        return_dict['TrainingArguments']['fsdp_config'] = final_fsdp_config
                    
                    else :
                        return_dict['TrainingArguments']['fsdp_config'] = fsdp_config
                    
            return return_dict
