# Self-Correction Learning

This repo contains the official code for **Self-Correction is More than Refinement: A Learning Framework for Language and Visual Reasoning Tasks**.  

**🌟 Trained Models**: [LLaVA-1.5-7B](https://huggingface.co/JiayiHe/SCL_LLaVA-1.5-7b) [LLaVA-1.5-13B](https://huggingface.co/JiayiHe/SCL_LLaVA-1.5-13b) [MiniCPM-Llama-V2.5](https://huggingface.co/JiayiHe/SCL_MiniCPM_Llama_V2.5)  
**🤗 Dataset**: [SelfCorSet](https://huggingface.co/datasets/JiayiHe/SELFCORSET)

## Getting Started

### Set up

Clone the repository and install the required packages  
```bash
git clone https://github.com/ivy3h/SCL.git
cd SCL
pip install -r requirements.txt
```

### Datasets & Models
Both the datasets and models are listed [here](guidance.md). Before running the code, please manually download the MMT-Bench dataset. All other datasets and models will be automatically downloaded during code execution.


### Inference 
To execute the intrinsic self-correction process, run the following command:
```bash
python inference.py --model [model name] --prompt [self-correction prompt] --dataset [evaluation dataset] --num_test [number of tasks]
```

### Data Construction
To construct preference data through the intrinsic self-correction process, run the following command:
```bash
python data_construction.py --model [model name] --prompt [self-correction prompt] --dataset [construction dataset]
```


### DPO
Our DPO code is based on [SWIFT](https://github.com/modelscope/ms-swift). To set the SWIFT environment, run the following commands:
```bash
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e '.[llm]'
pip install -e '.[eval]'
pip install -r requirements/framework.txt  -U
pip install -r requirements/llm.txt  -U
```

Set the dataset path to customize the dataset before initiating the optimization process. For more detailed information, please refer to the [SWIFT documentation](https://swift.readthedocs.io/en/latest/index.html). You can also explore additional alignment training methods.
> [!NOTE]  
> We recommend using the [WebUI](https://swift.readthedocs.io/en/latest/GetStarted/Web-ui.html) for training to enhance convenience and avoid potential bugs.

To execute the DPO, run the following command:
```bash
CUDA_VISIBLE_DEVICES=0,1 \
swift rlhf \
    --rlhf_type dpo \
    --model_type <model> \
    --beta 0.1 \
    --sft_beta 0.1 \
    --sft_type  lora \
    --dataset <dataset>  \
    --num_train_epochs  3  \
    --lora_target_modules  DEFAULT  \
    --gradient_checkpointing  true  \
    --batch_size  1  \
    --learning_rate  5e-5  \
    --gradient_accumulation_steps  16  \
    --warmup_ratio  0.03  \
    --save_total_limit  2
```

### Evaluation

Verify the file path of the trained model to locate the checkpoint directory. To execute the evaluation, run the following command:
```bash
CUDA_VISIBLE_DEVICES=0,1 \
swift eval \
    --model_type Trained model \
    --eval_dataset <dataset name> \
    --eval_limit <evaluation limit> \
    --ckpt_dir <checkpoint path> \
    --log_file <output file path> \
    --ignore_args_error true
```

## Acknowledgement
Our code is built on [IoE-Prompting](https://github.com/MBZUAI-CLeaR/IoE-Prompting) and [SWIFT](https://github.com/modelscope/ms-swift). We extend our gratitude to the authors for their work!
