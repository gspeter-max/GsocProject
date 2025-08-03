from backend.GlobalConfig import global_config
from backend.ModelingAndTuning import ModelLoadingAndTuning

config_obj = global_config(
      ModelName  = 'gpt2',
      DatasetPath = None, 
      FineTuningType = 'instruction_fine_tuning',
      ModelSeqMaxLength = None,
      QuantizationType4Bit8Bit = '8bit',
      ComputeMetricsList = ['accuracy'],
      PeftType  = 'LORA',
      SaveFormat  = 'gguf', 
      ModelDir  = './hfConvertedModel',  
      EvalSaveFormat  = 'json', 
      FSDP = True,
      HfToken  = ''
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
      per_device_train_batch_size=  8,
      per_device_eval_batch_size=  8,
      num_train_epochs= 3.0,
      learning_rate = 5e-5,
      weight_decay = 0.01,
      logging_dir = "./logs",
      logging_steps = 1,
      eval_strategy = "no",
      save_strategy = "steps",
      save_total_limit = 1,
      load_best_model_at_end = False,
      metric_for_best_model = "accuracy",
      greater_is_better = True,
      fp16 = True,
      fsdp  = False,
      label_names  = ['labels'],
      fsdp_config = None,
      warmup_steps = 500,
      lr_scheduler_type = "linear",
      gradient_accumulation_steps = 1,
      gradient_checkpointing = False,
      save_steps = 500,
      logging_first_step = True,
      report_to = None
)

fsdpconfig = config_obj.GetFSDP(
      fsdp_auto_wrap_policy= 'TRANSFORMER_BASED_WRAP',
      fsdp_backward_prefetch_policy =  'BACKWARD_PRE',
      fsdp_forward_prefetch = False,
      fsdp_cpu_ram_efficient_loading = True, 
      fsdp_offload_params =  False,
      fsdp_sharding_strategy = 'FULL_SHARD',
      fsdp_state_dict_type =  'SHARDED_STATE_DICT',
      fsdp_sync_module_states =  True,
      fsdp_transformer_layer_cls_to_wrap = 'GPT2Layers',
      fsdp_use_orig_params = True
)

HyperparameterConfig = config_obj.get_full_config(
      PeftConfig = peft_config, 
      TrainingArguments = training_arg, 
      fsdp_config = fsdpconfig      
)

tuning = ModelLoadingAndTuning(HyperparameterConfig)
tuning.LoadItTrainIt()
"""
# run in terminal
# cd ./GsocProject/autotuning/
# python -m backend.runner
"""
# -----------------------------------x =======================================
# autodetecting 

# config_obj = global_config(
#       ModelName  = 'gpt2',
#       FineTuningType = 'instruction_fine_tuning',
#       ComputeMetricsList = ['accuracy'],
#       SaveFormat  = 'gguf',
#       HfToken  = ''
# )

# HyperparameterConfig = config_obj.get_full_config()

# tuning = ModelLoadingAndTuning(HyperparameterConfig)
# tuning.LoadItTrainIt()

"""
# run in terminal
# cd ./GsocProject/autotuning/
# python -m backend.runner
"""
