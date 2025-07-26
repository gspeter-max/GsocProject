

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

