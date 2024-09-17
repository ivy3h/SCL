import os
import io
import re
import cv2
import sys
import base64
import string
import argparse
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from io import BytesIO
from datasets import load_dataset
from modelscope import AutoModel, AutoTokenizer, AutoModelForCausalLM
#from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType,
    get_default_template_type, inference_stream
)
from swift.utils import seed_everything


parser = argparse.ArgumentParser(
    description="Script for self-correction learning (SCL) model inference.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("--model", type=str, required=True, help="Name of the construction model.")
parser.add_argument("--prompt", type=int, help="ID of the self-correction prompt to use.")
parser.add_argument("--dataset", type=str, required=True, help="Inference dataset.")
args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# Load the specified model and tokenizer.
def load_model_and_tokenizer(model_name):
    if model_name == 'minicpm':
        model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True, torch_dtype=torch.float16).to('cuda')
        tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)
        model.eval()
        template = None
        
    elif args.model == 'llava7b':
        template_type = get_default_template_type('llava1_5-7b-instruct')
        model, tokenizer = get_model_tokenizer('llava1_5-7b-instruct', torch.float16, model_kwargs={'device_map': 'auto'})
        model = model.to('cuda')
        model.generation_config.max_new_tokens = 256
        template = get_template(template_type, tokenizer)
        
    elif args.model == 'llava13b':
        template_type = get_default_template_type('llava1_5-13b-instruct')
        model, tokenizer = get_model_tokenizer('llava1_5-13b-instruct', torch.float16, model_kwargs={'device_map': 'auto'})
        model = model.to('cuda')
        model.generation_config.max_new_tokens = 256
        template = get_template(template_type, tokenizer)
        
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model, tokenizer, template


# Load dataset based on the specified argument
def load_inference_dataset(dataset_name):
    dataset_map = {
        #'MMTBench': ('Kaining/MMT-Bench', 'train'),
        'MMStar': ('Lin-Chen/MMStar', 'val'),
        'MMBench': ('lmms-lab/MMBench_EN', 'dev'),
        'MMEvalPro': ('MM-Diagnose/MMEvalPro', 'test'),
        'SEEDBench': ('lmms-lab/SEED-Bench', 'test'),
        'ScienceQA': ('lmms-lab/ScienceQA-IMG', 'test')
    }
    
    if args.dataset == 'MMTBench':
      return pd.read_csv('./MMT-Bench/MMT-Bench_VAL.tsv', sep='\t').to_dict(orient='records')

    dataset_info = dataset_map.get(dataset_name)
    
    if dataset_info:
        ds = load_dataset(dataset_info[0])[dataset_info[1]]
        if args.dataset == 'MMTBench':
          columns_to_keep = ['index', 'question', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'image', 'category', 'l2-category', 'split']
          ds = ds.remove_columns([col for col in ds.column_names if col not in columns_to_keep])
        
        return ds

    else:
        print('Invalid dataset.', file=sys.stderr)
        return None


# Extracts the answer from the response string based on the dataset type.
def extract_answer(res):
    patterns = [
            r'##\s*([ABCD])\s*##',  
            r'##(.*?)##',           
            r'##\s*([0-3])\s*##'    
        ]
    for pattern in patterns:
        match = re.search(pattern, res)
        if match:
            return match.group(1).strip()
        else:
            match = re.search(r'([ABCD])\.', res)
            if match:
                return match.group(1).strip()
    return None


# Generates a review message based on the prompt type.
def get_review_msg(prompt):
    prompt_messages = {
        0: "Review your previous answer and find problems with your answer. Based on the problems you found, improve your answer.",
        1: "Review your previous answer and ensure that all relevant aspects of the image have been considered. Are there any elements or details that you missed? Based on your review, improve your answer.",
        2: "Review your contextual understanding of the image. Have you correctly interpreted the overall context and purpose of the scene? Based on your review, improve your answer.",
        3: "Review your answer and ensure that your understanding of the image is comprehensive and detailed. Are there any aspects of the scene that you have omitted or misinterpreted? Based on your review, improve your answer."
    }
    return prompt_messages.get(prompt, "") + " Explain your reasoning step-by-step."


total_data = []
path_prefix = './data/'
if not os.path.exists(path_prefix):
    os.makedirs(path_prefix)
f0_1 = 0
f1_0 = 0
ds = load_inference_dataset(args.dataset)
model, tokenizer, template = load_model_and_tokenizer(args.model)


# Builds options string based on the dataset.
def build_options_str(dataset, example):
    if dataset == 'MMEvalPro':
        choices = example['choices']
        letters = string.ascii_uppercase
        return '\n'.join([f'{letter}. {choice}' for letter, choice in zip(letters[:len(choices)], choices)])
    
    elif dataset == 'SEEDBench':
        return " A. {}. B. {}. C. {}. D. {}.".format(
            example['choice_a'], example['choice_b'], example['choice_c'], example['choice_d']
        )
    
    elif dataset == 'MMTBench':
        return " A. {}. B. {}. C. {}. D. {}.".format(
            example['A'], example['B'], example['C'], example['D']
        )
    
    elif dataset == 'MMBench':
        options_list = [f"{key}. {example.get(key)}" for key in ['A', 'B', 'C', 'D'] if example.get(key)]
        return " " + ' '.join(options_list)
    
    elif dataset == 'ScienceQA':
        choices = example['choices']
        return '\n'.join([f"{index}. {choice}" for index, choice in enumerate(choices)])
    
    return ""


# Determines the index based on the dataset type.
def get_index(dataset, item, i):
    if dataset == 'MMTBench' or dataset == 'MMStar' or dataset == 'MMBench':
        return item['index']
    elif dataset == 'MMEvalPro':
        return item['triplet_id']
    elif dataset == 'SEEDBench':
        return item['question_id']
    else:
        return i


for idx, item in enumerate(ds):
    tmp_data = {}
    if args.dataset == 'SEEDBench': 
      image = item['image'][0]
    elif args.dataset == 'MMTBench':
      image_base64 = item['image']
      image_decode = base64.b64decode(image_base64)
      image = Image.open(BytesIO(image_decode)).convert('RGB')
    else:
      image = item['image']

    if image.mode == 'RGBA':
      image = image.convert('RGB')
    
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='JPEG')
    image_bytes = image_bytes.getvalue()
    image_np = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

    index = get_index(args.dataset, item, idx)
    cv2.imwrite(path_prefix + f"{index}.jpg", image_np)
    tmp_data['image'] = f"{index}.jpg"

    question = item['question']
    tmp_data['prompt'] = question
    options_str = build_options_str(args.dataset, item)
    question += options_str

    if args.dataset == 'ScienceQA':
        extractor = " Your final answer should be put between two ##, like ## 0 ## (if your final answer is 0), at the end of your response."
    else:
        extractor = " Your final answer should be put between two ##, like ## A ## (if your final answer is A), at the end of your response."
        
    question = question + " Explain your reasoning step-by-step." + extractor
    
    image = Image.open(path_prefix + tmp_data['image']).convert('RGB')
    msgs = [{'role': 'user', 'content': question}]

    if args.model == 'minicpm':
        res_1 = model.chat(image=image, 
                           msgs=msgs, 
                           tokenizer=tokenizer, 
                           sampling=True, 
                           temperature=0.7
                          )
        
    else:
        res_1, his = inference(model, template, question, images=[path_prefix + tmp_data['image']])
    
    review_msg = get_review_msg(args.prompt)
    review_msg = review_msg + extractor
    msgs.append({'role': 'assistant', 'content': res_1})
    msgs.append({'role': 'user', 'content': review_msg})

    if args.model == 'minicpm':
        res_2 = model.chat(image=image, 
                           msgs=msgs, 
                           tokenizer=tokenizer, 
                           sampling=True,
                           temperature=0.7
                          )
        
    else:
        res_2, _ = inference(model, template, review_msg, history=his, images=[path_prefix + tmp_data['image'], path_prefix + tmp_data['image']])


    initial_answer = extract_answer(res_1)
    refined_answer = extract_answer(res_2)
    correct_answer = str(item['answer'])

    if initial_answer == correct_answer and refined_answer != correct_answer:
        tmp_data['chosen'] = res_1
        tmp_data['rejected'] = res_2
        total_data.append(tmp_data)
        f1_0 += 1
    elif refined_answer == correct_answer and initial_answer != correct_answer:
        tmp_data['chosen'] = res_2
        tmp_data['rejected'] = res_1
        total_data.append(tmp_data)
        f0_1 += 1
        
    print(f'1_0: {f1_0}')
    print(f'0_1: {f0_1}')

output_file = f'./data/Preference_Data_{args.model}_{args.dataset}_{args.prompt}.jsonl'
try:
    with open(output_file, 'w') as f:
        json.dump(total_data, f, ensure_ascii=False, indent=4)
    print(f'Data successfully saved to {output_file}.')
except Exception as e:
    print(f'An error occurred while saving data: {e}')

