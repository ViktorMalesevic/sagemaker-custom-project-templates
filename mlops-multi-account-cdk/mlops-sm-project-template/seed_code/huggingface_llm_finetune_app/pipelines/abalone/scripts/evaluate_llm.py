"""Evaluation script for measuring mean squared error."""
import json
import logging
import pathlib
import pickle
import tarfile

import numpy as np
import pandas as pd

import bitsandbytes as bnb
import torch
import torch.nn as nn

import transformers
from datasets import load_from_disk
from evaluate import load # Due to avoid the name 'evaluate' with this package, I changed the file name to 'evaluate_llm'. Otherwise, it casues error.
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
import argparse
from bert_score import score


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def inference_data(prompt, model, tokenizer):
    logger.info("Performing predictions against test data.")
    encoding = tokenizer(prompt, padding=True, truncation=True, return_tensors="pt").to('cuda')
    print_gpu_utilization()

    model.config.gradient_checkpointing = False
    model.config.use_cache = False
    
    generation_config = model.generation_config
    generation_config.max_new_tokens = 200
    generation_config.temperature = 0.7
    generation_config.top_p = 0.7
    generation_config.num_return_sequences = 1
    generation_config.pad_token_id = tokenizer.eos_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id
    
    
    with torch.inference_mode():
        outputs = model.generate(
            input_ids = encoding.input_ids.requires_grad_(False),
            attention_mask = encoding.attention_mask.requires_grad_(False),
            generation_config = generation_config,
        )
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    torch.cuda.empty_cache()
    
    return prediction

# Calculate evaluation metric
def bertscore_metrics(eval_dataset, model, tokenizer):
    logger.debug("Calculating bert score - f1 score.")
    predictions = []
    labels = []
    
    for data in eval_dataset:
        print(data['question'])
        print(data['answer'])
        prediction = inference_data(data['question'], model, tokenizer)
        label = data['answer']
        predictions.append(prediction.strip().lower())
        labels.append(label.strip().lower())

    bertscore = load("bertscore")
    results = bertscore.compute(predictions=predictions, references=labels, model_type="distilbert-base-uncased")
    
    precisions, recalls, f1_scores = results['precision'], results['recall'], results['f1']
    avg_precision, avg_recall, avg_f1_score =(sum(precisions) / len(precisions)),( sum(recalls) / len(recalls)), ( sum(f1_scores) / len(f1_scores))
    avg_results = {'precision':avg_precision, 'recall':avg_recall, 'f1':avg_f1_score}

    return avg_results

# This is specific to question answering, please change this part according to the type of task
def post_processing_preds(preds_out):
    """
    args: preds_out, a jsonline file containing predictions
    """
    def remove_question_get_generated_answer(output_line):
        # Split the string at '\n'
        parts = output_line.split('\n')
        # Get everything after the first '\n'
        print(parts)
        answer = parts[1:] if len(parts) > 1 else None
        # Concatenate all strings in the list
        concatenated_answer = ' '.join(answer)
        # Remove "AI: "
        concatenated_answer = concatenated_answer[4:]
        return concatenated_answer

    def cut_at_last_comma(text):
        # Find the last occurrence of a comma in the text
        last_comma_index = text.rfind('.')

        # If a comma is found, cut the text up to the character after the last comma
        if last_comma_index != -1:
            return text[:last_comma_index + 1]

        # If no comma is found, return the original text
        return text

    output_content = preds_out['Body'].read().decode('utf-8')
    print(f"OUTPUT:\n{output_content}")
    # Split the string into individual JSON strings
    segments = [segment[1:] + '}' for segment in output_content.split('}]') if segment]
    print(f"segments:\n{segments}")

    # Load each segment as JSON and extract the "generated_text" field
    final_output_content = [cut_at_last_comma(json.loads(segment)["generated_text"]) for segment in segments]
    print(f"final_output_content:\n{final_output_content}")
    final_output_content = [remove_question_get_generated_answer(output_line) for output_line in final_output_content]
    print(f"final_output_content:\n{final_output_content}")
    return final_output_content



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_model_id", type=str)
    parser.add_argument("--testset_filename", type=str)
    
    logger.debug("Starting evaluation with test set.")
    # model_dir = "/opt/ml/model" (Reuse when we can pass trained model without Model hub
    model_dir = args.hf_model_id
    
    '''
    HF_MODEL_ID can be an option for downloading the pretrained model. Since fine-tuned model is uploaded to the huggingface hub after HPO.
    HF_MODEL_ID = '{YOUR_HF_HUB}/falcon_7b_ecommerce_ai_chatbot'
    '''

    # Load the model
    logger.debug("Loading fine-tuned Falcon model.")

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        # HF_MODEL_ID,
        return_dict=True,
        trust_remote_code=True,
    )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,  
        # HF_MODEL_ID
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # TODO: Change post_processing_preds to read jsonlines instead of json (for batch transform)
    logger.debug("Reading test data.")
    test_path = f"/opt/ml/input/data/test/{args.testset_filename}.out"
    with open(test_path) as f:
        test_dataset = json.load(f)
        
    pred_answers = post_processing_preds(test_path)
        
    print(test_dataset)
    logger.info("Performing predictions against test data.")
    avg_results = bertscore_metrics(test_dataset, model, tokenizer)   
    print("avg_results", avg_results)

    eval_report_dict = {
        "Q&A_metrics": {
            "bertscore": {
                "f1_score": avg_results['f1'],
                "precision": avg_results['precision'],
                "recall":avg_results['recall'],
            },
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Writing out evaluation report with f1 score: %f", avg_results['f1'])
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(eval_report_dict))
