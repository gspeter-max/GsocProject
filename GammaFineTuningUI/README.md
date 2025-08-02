### ## Prerequisites
### 1. Clone the llama.cpp repository:
###    `git clone https://github.com/ggml-org/llama.cpp.git`
### 2. Install the required Python packages:
###    `pip install -r llama.cpp/requirements.txt`
### 3. install torcheval
###    `pip install torcheval`
### 4. for visilization and looking training progress do  this before executing trainer.train 
    ```python  %load_ext tensorboard
         %tensorboard --logdir ./logs ``` 
 
### auto detected dataset format type 
### ['domain_adaptation','instruction_fine_tuning','code_generation','chat_fine_tuning','question_answering','rag_fine_tuning'] 
