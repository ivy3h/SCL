import os
import torch
import argparse
import sys
import json
import io
import re
import base64
from io import BytesIO
import pandas as pd
from PIL import Image
from datasets import load_dataset
from modelscope import AutoModel, AutoTokenizer, AutoModelForCausalLM
from swift.llm import (
    get_model_tokenizer, get_template, inference,
    get_default_template_type
)
from swift.utils import seed_everything
from pathlib import Path

# --- Constants and Configurations ---
MODEL_MINICPM = 'minicpm'
MODEL_INTERNLM = 'internlm'
MODEL_LLAVA7B = 'llava7b'

SUPPORTED_MODELS = [MODEL_MINICPM, MODEL_INTERNLM, MODEL_LLAVA7B]

PROMPT_MESSAGES = {
    0: "Review your previous answer and find problems with your answer. Based on the problems you found, improve your answer.",
    1: "Review your previous answer and ensure that all relevant aspects of the image have been considered. Are there any elements or details that you missed? Based on your review, improve your answer.",
    2: "Review your contextual understanding of the image. Have you correctly interpreted the overall context and purpose of the scene? Based on your review, improve your answer.",
    3: "Review your answer and ensure that your understanding of the image is comprehensive and detailed. Are there any aspects of the scene that you have omitted or misinterpreted? Based on your review, improve your answer."
}

DATASET_CONFIG = {
    'MMStar': {'path': 'Lin-Chen/MMStar', 'split': 'val'},
    'MMBench': {'path': 'lmms-lab/MMBench_EN', 'split': 'dev'},
    'RealWorldQA': {'path': 'lmms-lab/RealWorldQA', 'split': 'test'},
    'SEEDBench': {'path': 'lmms-lab/SEED-Bench', 'split': 'test'},
    'ScienceQA': {'path': 'lmms-lab/ScienceQA-IMG', 'split': 'test'},
    'MMTBench': {'path': './MMT-Bench/MMT-Bench_VAL.tsv', 'split': None, 'local_csv': True} # Special case
}

# --- Argument Parsing ---
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Script for self-correction learning (SCL) model inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", type=str, required=True, choices=SUPPORTED_MODELS,
                        help="Name of the inference model.")
    parser.add_argument("--prompt_id", type=int, default=2, choices=list(PROMPT_MESSAGES.keys()),
                        help="ID of the self-correction prompt to use.")
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASET_CONFIG.keys()),
                        help="Inference dataset name.")
    parser.add_argument("--num_test", type=int, default=100,
                        help="Number of test data points.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--cuda_device", type=str, default='0', help="CUDA device to use (e.g., '0', '0,1').")
    return parser.parse_args()

# --- Model Loading ---
def load_model_and_tokenizer(model_name_arg: str):
    model, tokenizer, template = None, None, None
    device = 'cuda'

    if model_name_arg == MODEL_MINICPM:
        model_path = 'openbmb/MiniCPM-Llama3-V-2_5'
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    elif model_name_arg == MODEL_INTERNLM:
        model_path = 'Shanghai_AI_Laboratory/internlm-xcomposer2-7b'
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    elif model_name_arg == MODEL_LLAVA7B:
        model_id = 'llava1_5-7b-instruct'
        template_type = get_default_template_type(model_id)
        model, tokenizer = get_model_tokenizer(model_id, torch.float16, model_kwargs={'device_map': 'auto'})
        model.generation_config.max_new_tokens = 256
        template = get_template(template_type, tokenizer)
    else:
        raise ValueError(f"Unsupported model: {model_name_arg}")

    if model is not None: 
        model = model.to(device)
        model.eval()
        
    return model, tokenizer, template

# --- Dataset Loading ---
def load_inference_dataset(dataset_name_arg: str):
    config = DATASET_CONFIG.get(dataset_name_arg)
    if not config:
        print(f'Invalid dataset name: {dataset_name_arg}', file=sys.stderr)
        return None

    if config.get('local_csv', False):
        try:
            return pd.read_csv(config['path'], sep='\t').to_dict(orient='records')
        except FileNotFoundError:
            print(f"Local dataset file not found: {config['path']}", file=sys.stderr)
            return None
    else:
        return load_dataset(config['path'])[config['split']]

# --- Image Handling ---
def get_image_from_example(example, dataset_name_arg: str):
    if dataset_name_arg == 'SEEDBench':
        return example['image'][0]
    elif dataset_name_arg == 'MMTBench':
        image_base64 = example['image']
        image_decode = base64.b64decode(image_base64)
        return Image.open(BytesIO(image_decode)).convert('RGB')
    else: 
        return example['image']

# --- Text Processing ---
def extract_answer(response_text: str):
    patterns = [
        r'##\s*([ABCD])\s*##',
        r'##(.*?)##',
        r'##\s*([0-3])\s*##'
    ]
    for pattern in patterns:
        match = re.search(pattern, response_text)
        if match:
            return match.group(1).strip()
            
    match = re.search(r'([ABCD])\.', response_text) 
    if match:
        return match.group(1).strip()
    return None 

def build_options_str(dataset_name_arg: str, example: dict) -> str:
    if dataset_name_arg == 'SEEDBench':
        return " A. {}. B. {}. C. {}. D. {}.".format(
            example['choice_a'], example['choice_b'], example['choice_c'], example['choice_d']
        )
    elif dataset_name_arg == 'MMTBench':
        return " A. {}. B. {}. C. {}. D. {}.".format(
            example['A'], example['B'], example['C'], example['D']
        )
    elif dataset_name_arg == 'MMBench':
        options_list = [f"{key}. {example.get(key)}" for key in ['A', 'B', 'C', 'D'] if example.get(key)]
        return " " + ' '.join(options_list)
    elif dataset_name_arg == 'ScienceQA':
        choices = example['choices']
        return '\n' + '\n'.join([f"{idx}. {choice}" for idx, choice in enumerate(choices)])
    return ""

def get_review_message_template(prompt_id_arg: int) -> str:
    message = PROMPT_MESSAGES.get(prompt_id_arg, "Review your previous answer and improve it.")
    return message + " Explain your reasoning step-by-step."

def get_answer_extractor_template(dataset_name_arg: str) -> str:
    if dataset_name_arg == 'ScienceQA':
        return " Your final answer should be put between two ##, like ## 0 ## (if your final answer is 0), at the end of your response."
    else:
        return " Your final answer should be put between two ##, like ## A ## (if your final answer is A), at the end of your response."

# --- Inference Logic ---
def perform_inference(model, tokenizer, template, image, messages, model_name_arg: str, history=None):
    """Performs a single step of inference based on the model type."""
    response = ""
    new_history = history

    if model_name_arg == MODEL_MINICPM:
        response = model.chat(
            image=image,
            msgs=messages,
            tokenizer=tokenizer,
            sampling=True,
            temperature=0.7
        )
        
        new_history = messages + [{'role': 'assistant', 'content': response}]

    elif model_name_arg == MODEL_INTERNLM:
        query_content = messages[-1]['content']
        processed_image = image # Assuming image is already a tensor for InternLM

        with torch.cuda.amp.autocast():
            response, new_history = model.chat(
                tokenizer,
                query=query_content,
                image=processed_image, # Pass the tensor image
                history=history if history is not None else [],
                do_sample=True,
                temperature=0.7
            )
    elif model_name_arg == MODEL_LLAVA7B:
        query_content = messages[-1]['content']
        current_images = [image] 
        if history and model_name_arg == MODEL_LLAVA7B: 
             current_images = [image, image] 

        response, new_history = inference(model, template, query_content, images=current_images, history=history)

    return response, new_history


# --- Main Processing Logic ---
def main():
    args = parse_arguments()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    seed_everything(args.seed)

    print(f"Loading model: {args.model}")
    model, tokenizer, template = load_model_and_tokenizer(args.model)
    if model is None:
        sys.exit(f"Failed to load model {args.model}")

    print(f"Loading dataset: {args.dataset}")
    dataset = load_inference_dataset(args.dataset)
    if dataset is None:
        sys.exit(f"Failed to load dataset {args.dataset}")
    
    # Pre-calculate templates
    review_msg_template = get_review_message_template(args.prompt_id)
    answer_extractor_suffix = get_answer_extractor_template(args.dataset)

    results_log = []
    correct_first_try = 0
    correct_second_try = 0
    total_processed_questions = 0

    for idx, example in enumerate(dataset):
        if total_processed_questions >= args.num_test:
            break

        try:
            pil_image = get_image_from_example(example, args.dataset)
            correct_answer = str(example['answer'])
            
            question_text = example['question']
            options_text = build_options_str(args.dataset, example)
            full_question = question_text + options_text
            
            inference_image_arg = pil_image
            if args.model == MODEL_INTERNLM:
                inference_image_arg = model.vis_processor(pil_image)
                inference_image_arg = torch.stack([inference_image_arg]).to(torch.float16).to('cuda')
                full_question = "<ImageHere> " + full_question


            # --- First Inference ---
            prompt_1 = full_question + "\nExplain your reasoning step-by-step." + answer_extractor_suffix
            messages_1 = [{'role': 'user', 'content': prompt_1}]
            
            response_1, history_1 = perform_inference(
                model, tokenizer, template, inference_image_arg, messages_1, args.model
            )

            # --- Second Inference (Self-Correction) ---
            prompt_2 = review_msg_template + answer_extractor_suffix
            if args.model == MODEL_MINICPM:
                messages_2 = history_1 + [{'role': 'user', 'content': prompt_2}] 
                history_for_2nd_call = None 
            else:
                messages_2 = [{'role': 'user', 'content': prompt_2}] 
                history_for_2nd_call = history_1


            response_2, _ = perform_inference(
                model, tokenizer, template, inference_image_arg, messages_2, args.model, history=history_for_2nd_call
            )

            # --- Process Results ---
            initial_answer_extracted = extract_answer(response_1)
            refined_answer_extracted = extract_answer(response_2)

            initial_answer = initial_answer_extracted if initial_answer_extracted is not None else response_1
            refined_answer = refined_answer_extracted if refined_answer_extracted is not None else response_2
            
            current_qa_log = {
                'index': example.get('index', idx), 
                'id': example.get('id', idx),       
                'question_full': prompt_1,
                'response_1': response_1,
                'initial_answer_extracted': initial_answer_extracted,
                'initial_answer_final': initial_answer,
                'review_prompt': prompt_2,
                'response_2': response_2,
                'refined_answer_extracted': refined_answer_extracted,
                'refined_answer_final': refined_answer,
                'correct_answer': correct_answer,
            }

            type_code = 0
            if initial_answer == correct_answer:
                correct_first_try += 1
                if refined_answer == correct_answer:
                    correct_second_try += 1
                    type_code = 1 # Correct -> Correct
                else:
                    type_code = 3 # Correct -> Incorrect
            else:
                if refined_answer == correct_answer:
                    correct_second_try += 1
                    type_code = 2 # Incorrect -> Correct
                else:
                    type_code = 4 # Incorrect -> Incorrect
            
            current_qa_log['type'] = type_code
            results_log.append(current_qa_log)
            total_processed_questions += 1

            print(f"\n--- Question {idx + 1}/{args.num_test} (Dataset index: {current_qa_log['id']}) ---")
            print(f"Initial Answer: {initial_answer} (Extracted: {initial_answer_extracted})")
            print(f"Refined Answer: {refined_answer} (Extracted: {refined_answer_extracted})")
            print(f"Correct Answer: {correct_answer}")
            print(f"Type: {type_code}")
            if total_processed_questions > 0:
                print(f"Accuracy So Far - Initial: {correct_first_try / total_processed_questions:.2%}")
                print(f"Accuracy So Far - Refined: {correct_second_try / total_processed_questions:.2%}")

        except Exception as e:
            print(f"Error processing question index {idx} (ID: {example.get('id', 'N/A')}): {e}", file=sys.stderr)
            error_log_entry = {
                'index': example.get('index', idx),
                'id': example.get('id', idx),
                'error': str(e)
            }
            results_log.append(error_log_entry) 
            continue 

    # --- Save Results ---
    if total_processed_questions > 0:
        final_accuracy_first = correct_first_try / total_processed_questions
        final_accuracy_second = correct_second_try / total_processed_questions
        print("\n--- Final Summary ---")
        print(f"Total questions processed: {total_processed_questions}")
        print(f"Initial answer accuracy: {final_accuracy_first:.2%}")
        print(f"Refined answer accuracy: {final_accuracy_second:.2%}")

        output_dir = Path("results")
        output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_file = output_dir / f'QAs_{args.model}_{args.dataset}_prompt{args.prompt_id}_n{args.num_test}_seed{args.seed}.jsonl'
        
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for item in results_log:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        print(f"Results successfully saved to {jsonl_file}")
    else:
        print("No valid questions were processed.")

if __name__ == "__main__":
    main()