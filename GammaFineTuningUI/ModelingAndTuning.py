from transformers import (
        AutoModelForCausalLM , 
        AutoTokenizer , 
        BitsAndBytesConfig, 
        TrainningArguments
        ) 
from trl import SFTTrainer 


class ModelLoadingAndTuning:
    def __init__(
            self, 
            GlobalConfig
            ): 
        self.GlobalConfig = GlobalConfig 

    def LoadItTrainIt( self, dataset ):
        if self.GlobalConfig['load_in_4bit']: 
            QantizationConfig = BitsAndBytesConfig(
                load_in_4bit = True 
                    )
        else: 
            QuantizationConfig = BitsAndBytesConfig(
                    load_in_8bit = True
                )
        model = AutoModelForCausalLM.from_pretrained(
                self.GlobalConfig['ModelName'], 
                quantization_config = QuantizationConfig
                )
        TrainingArg = TrainingArguments(
                output_dir = self.GlobalConfig.get('output_dir', None ),
                do_train = self.GlobalConfig.get('do_train', False ), 
                per_device_train_batch_size = self.GlobalConfig.get('per_device_train_batch_size', 8),
                learning_rate = self.GlobalConfig.get('lr',5e-05),
                weight_decay = self.GlobalConfig.get('weight_decay',0.0),
                num_train_epochs = self.GlobalConfig.get('num_train_epochs',3),
                eval_strategy = self.GlobalConfig.get('eval_strategy','epochs'), 
                )

        trainer = SFTTrainer(
                model = model, 
                args = TrainingArg, 
                train_dataset = dataset, 
                callbacks = [CallBackFunc], 
                peft_config = self.GlobalCofig.get('peft_config', None) 
                ) 





