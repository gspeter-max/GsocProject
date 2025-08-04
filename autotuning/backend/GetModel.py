
from transformers import TFAutoModelForCausalLM
import torch
import logging
import shutil
import subprocess
import os 


logger = logging.getLogger()
logger.setLevel(logging.INFO)

class ConvertModel:
    def __init__( self , Format : str = None, \
            WhereStored : str = None
                 ):
        self.Format = Format
        if WhereStored is None:
            WhereStored = './hfConvertedModel'

        self.WhereStored = WhereStored

    def __call__( self, Model,Tokenizer):

        if self.Format.lower() == 'tensorflow':
            ptSaveFile = './TempStore'
            Model.save_pretrained(ptSaveFile)
            Tokenizer.save_pretrained(ptSaveFile)

            tfmodel = TFAutoModelForCausalLM.from_pretrained( ptSaveFile, from_pt = True )

            tfmodel.save_pretrained( self.WhereStored )
            Tokenizer.save_pretrained( self.WhereStored )
            shutil.rmtree(ptSaveFile)
            logger.info(''' load out sotred model using huggingface transformer library
                example :  in tensorflow format loadding
                    from transformers import TFAutoModelForCausalLM

                    model = TFAutoModelForCausalLM.from_pretrained( WhereStored , from_pt = True )
                '''
                    )

        if self.Format.lower() == 'torch':
            Model.save_pretrained( self.WhereStored )

            logger.info(''' load out sotred model using huggingface transformer library
                    example :  in torch  format loadding
                        from transformers import AutoModelForCausalLM

                        model = AutoModelForCausalLM.from_pretrained( WhereStored )
                    '''
                    )

        if self.Format == 'gguf':
            ptSaveFile = './TempStore'

            Model.save_pretrained(ptSaveFile)
            Tokenizer.save_pretrained(ptSaveFile)

            print('========================Please wait while an external library is downloading=================')
            ProcessOutput = subprocess.run(f'pip install -r requirements.txt',\
                    shell = True, \
                    stdout = subprocess.PIPE, \
                    stderr = subprocess.PIPE, \
                    text = True
                )
            print(ProcessOutput.stdout)
            print(ProcessOutput.stderr)

            ProcessOutput = subprocess.run(f'git clone https://github.com/ggml-org/llama.cpp.git',\
                    shell = True, \
                    stdout = subprocess.PIPE, \
                    stderr = subprocess.PIPE, \
                    text = True
                )
            print(ProcessOutput.stdout)
            print(ProcessOutput.stderr)

            ProcessOutput = subprocess.run(f'mkdir {self.WhereStored}',\
                    shell = True, \
                    stdout = subprocess.PIPE, \
                    stderr = subprocess.PIPE, \
                    text = True
                )
            print(ProcessOutput.stdout)
            print(ProcessOutput.stderr)            

            ProcessOutput = subprocess.run(f'python llama.cpp/convert_hf_to_gguf.py ./TempStore --outfile ./{self.WhereStored}/output_gguf.gguf --outtype q8_0',\
                    shell = True, \
                    stdout = subprocess.PIPE, \
                    stderr = subprocess.PIPE, \
                    text = True
                )
            print(ProcessOutput.stdout)
            print(ProcessOutput.stderr)

            shutil.rmtree(ptSaveFile)
            shutil.rmtree('./llama.cpp')
            pwd = os.getcwd() 
            logger.info(f'''
            for loading again
                do this
                must be check out version of packages like ( torch , torchvision )

                    from transformers import AutoModelForCasualLM

                    model = AutoModelForCausalLM.from_pretraind(
                            f"{pwd}/{self.WhereStored}",
                            gguf_file = f"{pwd}/{self.WhereStored}/output_gguf.gguf"
                            )
                '''
                )

        return f'model is stored in {self.WhereStored} , {self.Format} format'
