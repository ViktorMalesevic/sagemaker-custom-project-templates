import json
import os 
import sys

import logging
import bitsandbytes as bnb
import pandas as pd
import torch

import transformers
from datasets import load_from_disk
from evaluate import load
from peft import (
    LoraConfig,
    TaskType,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
    AutoPeftModelForCausalLM
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
import optuna
import warnings

import sagemaker
import boto3
import pickle
from huggingface_hub import HfApi, HfFolder


# Filter specific warnings
warnings.filterwarnings('ignore', category=UserWarning, message="None of the inputs have requires_grad=True. Gradients will be None")
warnings.filterwarnings('ignore', message="`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
warnings.filterwarnings('ignore', message="You're using a PreTrainedTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.")

print(sagemaker.__version__)


# TODO: this needs to be run only the first time. Run this outside of the pipeline and pass token as Pipeline parameter
# TODO2: explore how to save model locally and deploy from local s3. However latest huggingface llm image uri/sagemaker inference toolkit seems to not accept this behavior anymore
def save_huggingface_token(token):
    # Save token to Hugging Face folder (typically ~/.huggingface)
    folder = HfFolder()
    folder.save_token(token)

# def print_summary(result):
#     print(f"Time: {result.metrics['train_runtime']:.2f}")
#     print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")

def get_model_tokenizer(model_name):

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,

    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_4bit=True,
        quantization_config=bnb_config,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    
    )

    # prepare int-4 model for training
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    
    config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, config)
    print("INSIDE get_model_tokenizer - model = get_peft_model(model, config)")
    model.print_trainable_parameters()

    return model, tokenizer

def inference_data(prompt, model, tokenizer):

    encoding = tokenizer(prompt, padding=True, truncation=True, return_tensors="pt").to('cuda:0')

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

def bertscore_metrics(eval_dataset, model, tokenizer):
    predictions = []
    labels = []
    
    for data in eval_dataset:
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

def objective(trial):

    global best_f1_score
    # Define hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    num_train_epochs = trial.suggest_int("num_train_epochs", 1, 10)
    max_steps = trial.suggest_categorical("max_steps", list(range(80,241,20)))
    per_device_train_batch_size = trial.suggest_categorical("per_device_train_batch_size", [1, 2, 4])
    
    print("######################################################################################################################")
    print(f"[{trial.number+1}] --- learning_rate:{learning_rate} | num_train_epochs:{num_train_epochs} | max_steps:{max_steps} | per_device_train_batch_size:{per_device_train_batch_size}")
    # Use hyperparameters in TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_data_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        max_steps = max_steps,
        # Other fixed parameters
        warmup_ratio=0.1,
        remove_unused_columns=False,
        save_total_limit=3,  
        logging_steps=10,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
    )
    
    # initiate the model and tokenizer
    model_name = args.model_name
    new_model, new_tokenizer = get_model_tokenizer(model_name)

    # new_model.config.gradient_checkpointing = False
    new_model.config.use_cache = False
    # new_model = prepare_model_for_kbit_training(new_model) 
    print("INSIDE OBJECTIVE - new_model.print_trainable_parameters()")
    new_model.print_trainable_parameters()

    # Define Trainer
    trainer = Trainer(
        model=new_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=transformers.DataCollatorForLanguageModeling(new_tokenizer, mlm=False),
    )
    
    # Train the model
    # new_model.config.use_cache = False  # Disable caching
    trainer.train()
 
    avg_results = bertscore_metrics(eval_dataset, new_model, new_tokenizer)
    f1_score = avg_results['f1']  
    
    return f1_score    


if __name__ == "__main__":

    print("hpo_train_falcon.py is starting.....!!!!!!")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["HF_TASK"] = "question-answering"
    
    parser = argparse.ArgumentParser()
    
    # To configure Optuna db
    parser.add_argument('--model_name', type=str, default="tiiuae/falcon-7b")
    parser.add_argument('--study-name', type=str, default='falcon-chatbot')
    parser.add_argument('--n_trials', type=int, default=100)

    parser.add_argument('--push_to_hub', type=str, default=True)
    parser.add_argument('--hub_model_id', type=str, required=True)
    parser.add_argument('--hub_token', type=str, required=True)
    
    # Data, model, and output directories
    var_list = ["SM_OUTPUT_DATA_DIR", "SM_MODEL_DIR" , "SM_CHANNEL_TRAIN", "SM_CHANNEL_TEST"]
    for var in var_list:
        if var in os.environ:
            print(os.environ[var])
        else:
            # os.environ[var] = None
            print(f"{var} is not set")
    
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument("--merge_weights", type=bool, default=True)

    args, _ = parser.parse_known_args()
    print("args:\n",args)
    
    print(f'output_data_dir - arg: {args.output_data_dir} | os.environ: {os.environ["SM_OUTPUT_DATA_DIR"]}')
    print(f'model_dir - arg: {args.model_dir} | os.environ: {os.environ["SM_MODEL_DIR"]}')
    print(f'training_dir - arg: {args.training_dir} | os.environ: {os.environ["SM_CHANNEL_TRAIN"]}')
    print(f'test_dir - arg: {args.test_dir} | os.environ: {os.environ["SM_CHANNEL_TEST"]}')
    
    save_huggingface_token(args.hub_token)

     # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # load train/ eval datasets
    '''
    Train dataset is huggingface dataset format
    Eval dataset is json file consisting of question and answer pair. (eval_dataset.json)
        1. get the test_dir (s3://bucket/path)
        2. split test_dir into bucket and path
        3. load dataset from the bucket dir using get_object
        4. decode and load with json
    '''
    print(f"training_dir:{os.environ['SM_CHANNEL_TRAIN']}\n, test_dir:{os.environ['SM_CHANNEL_TEST']}")
    train_dataset = load_from_disk(args.training_dir)
    
    # Open the eval_dataset
    file_list = [f for f in os.listdir(args.test_dir) if os.path.isfile(os.path.join(args.test_dir, f))]
    eval_dataset_name = file_list[0]
    
    eval_data_dir = os.path.join(args.test_dir, eval_dataset_name)
    with open(eval_data_dir) as f:
        # Load the JSON data
        eval_dataset = json.load(f)

    # Now you can access the data as a Python dictionary
    print("eval_dataset",eval_dataset)
    
    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f" loaded eval_dataset length is: {len(eval_dataset)}")

    #Create a study to run hyperparameter optimization
    study = optuna.create_study(study_name=args.study_name, direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)
    
    logger.info("Best trial:")
    trial = study.best_trial

    logger.info("  Value: {}".format(trial.value))

    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info("    {}: {}".format(key, value))
        
    # Save the study results to a dataframe
    df = study.trials_dataframe()

    # Save the dataframe to a csv file
    df.to_csv(f'{args.output_data_dir}/study_results.csv')
    
    # Train with the best parameter
    best_trial = study.best_trial
    print(f"Best Trial: score {best_trial.value}, params {best_trial.params}")

    best_score = best_trial.value

    print(f'Best score is {best_trial.value}!!!!!')

    best_params = best_trial.params

    model_name = "tiiuae/falcon-7b"
    model, tokenizer = get_model_tokenizer(model_name)

    model.config.gradient_checkpointing = False
    print(model.parameters().__next__().device)

    # Train model with best hyperparameters
    training_args = TrainingArguments(
        output_dir=args.output_data_dir,
        learning_rate=best_params['learning_rate'],
        per_device_train_batch_size=best_params['per_device_train_batch_size'],
        num_train_epochs=best_params['num_train_epochs'],
        warmup_ratio=0.1,
        max_steps = best_params['max_steps'],
        # Other fixed parameters
        remove_unused_columns=False,
        fp16=False,
        save_total_limit=3,  
        logging_steps=1,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        
    )

    trainer.train()
     
    # Save the model
    peft_model_dir = args.model_dir
    repo_name = args.hub_model_id
    trainer.model.save_pretrained(peft_model_dir, safe_serialization=False)
    
    del model
    del trainer
    torch.cuda.empty_cache()
    config = PeftConfig.from_pretrained(peft_model_dir)

    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoPeftModelForCausalLM.from_pretrained(
        peft_model_dir, 
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

   
    merged_model = model.merge_and_unload()

    torch.cuda.empty_cache()


