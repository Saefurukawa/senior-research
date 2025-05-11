# Copyright © 2023-2024 Apple Inc.

# import argparse
import logging
import json
import os
from uuid import uuid4

import numpy as np
import torch
import transformers
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from peft import LoraConfig, PeftConfig, TaskType
from dataset.argument_parsing import add_dataset_arguments, get_datasets, parse_draw_num_datapoints_per_user, get_custom_datasets
from model.hugging_face import add_special_tokens, causal_lm_metrics_fn, smart_embedding_resize, wrap_hugging_face_model
from utils.argument_parsing import (
    add_algorithm_arguments,
    add_filepath_arguments,
    add_seed_arguments,
    get_algorithm,
    maybe_inject_arguments_from_config,
    parse_mechanism,
)
from utils.callback.hugging_face import HuggingFaceModelCheckpointingCallback
from utils.logging import init_logging

# from llm.argument_parsing import add_llm_arguments, parse_central_lr_scheduler, parse_peft_config
from pfl.aggregate.simulate import SimulatedBackend
from pfl.callback import AggregateMetricsToDisk, CentralEvaluationCallback, StopwatchCallback, WandbCallback
from pfl.hyperparam import NNEvalHyperParams, NNTrainHyperParams
from pfl.algorithm import (
    AdaptMuOnMetricCallback,
    FederatedAlgorithm,
    FederatedAveraging,
    FedProx,
    FedProxParams,
    NNAlgorithmParams,
)
from pfl.internal.ops import pytorch_ops
from pfl.model.pytorch import PyTorchModel

import json
from datasets import Dataset as HFDataset
from typing import Dict, Any, Tuple
from pfl.data.pytorch import PyTorchDataDataset, PyTorchFederatedDataset
from pfl.data.sampling import get_user_sampler
# from hugging_face import IGNORE_INDEX, GetItemDataset, UserIDCollatorWrapper #Check
from types import SimpleNamespace


init_logging(logging.INFO)
# maybe_inject_arguments_from_config()
logger = logging.getLogger(name=__name__)

# parser = argparse.ArgumentParser(
#     description=
#     'Train a model using private federated learning in simulation.')

# parser = add_algorithm_arguments(parser)
# parser = add_filepath_arguments(parser)
# parser = add_dataset_arguments(parser)
# parser = add_seed_arguments(parser)
# parser = add_llm_arguments(parser)
# arguments = parser.parse_args()

#Random seed
# np.random.seed(42)
# torch.random.manual_seed(42)

#Privacy mechanism -none for now
# local_privacy = parse_mechanism(
#     mechanism_name="none", #No privacy
#     clipping_bound=1.0,
#     epsilon=1.,
#     delta=1e-05,
#     order=None)

def run_experiments(round):
    central_privacy = parse_mechanism(
        mechanism_name="gaussian_moments_accountant", 
        clipping_bound=0.1,
        epsilon=2.0,
        delta=1e-6,
        order=2,
        cohort_size=100,
        noise_cohort_size=10000, #we need to change this
        num_epochs=100,
        population=1e6,
        min_separation=None,
        is_central=True)

    # local_privacy = parse_mechanism(
    #     mechanism_name="gaussian", 
    #     clipping_bound=0.1,
    #     epsilon=2.0,
    #     delta=1e-6,
    #     order=None,
    #     # cohort_size=100,
    #     # noise_cohort_size=5000, #we need to change this
    #     # num_epochs=100,
    #     # population=1e6,
    #     # min_separation=None,
    #     is_central=False)
    # central_mechanism = local_privacy.approximate_as_central_mechanism()
    #define tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        model_max_length=512,
        padding_side="right",
        use_fast=True)

    # # Define a distinct padding token if it doesn't exist
    # if tokenizer.pad_token is None:
    #     # After initializing the tokenizer
    #     tokenizer.pad_token = '<s>'  # Assign <s> as the padding token
    #     print(f"New pad_token_id: {tokenizer.pad_token_id}")  # Should output 1

    #Adding special tokens here take a long time because we need to resize the model later.
    # num_new_tokens = add_special_tokens(tokenizer)
    # logger.info(f"Added {num_new_tokens} new tokens to the tokenizer.")
    # arguments.tokenizer = tokenizer
    if tokenizer.pad_token is None:
        # After initializing the tokenizer
        tokenizer.pad_token = '<s>'  # Assign <s> as the padding token
        print(f"New pad_token_id: {tokenizer.pad_token_id}")  # Should output 1

    logger.info("Loading dataset.")

    #Prepare dataset    

    #Fix this
    training_data, val_data, central_data, metadata = get_custom_datasets(tokenizer)

    #Everything is good until here.

    #Model
    logger.info(f"Loading model")

    #Quantization is probably more prefered, however, installation is tricky.
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_use_double_quant = True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    #torch_dtype=torch.bfloat16
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.1", quantization_config=bnb_config, device_map="auto", trust_remote_code=True) #Make sure memory does not run out.
    # smart_embedding_resize(num_new_tokens, tokenizer, hf_model)

    hf_model.config.pad_token_id = tokenizer.pad_token_id

    # Parameter efficient fine-tuning - same parameters as centralized learning
    peft_config = LoraConfig(
        task_type = "CAUSAL_LM", 
        inference_mode = False,
        r=32, 
        lora_alpha =64, 
        target_modules =[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        lora_dropout = 0.1
        ) #Look into what's available

    hf_model = wrap_hugging_face_model(hf_model, peft_config,
                                       causal_lm_metrics_fn)
    # Put on GPU if available.
    hf_model = hf_model.to(pytorch_ops.get_default_device())

    params = [p for p in hf_model.parameters() if p.requires_grad]
    central_optimizer = torch.optim.Adam(params,
                                         0.01, #edit here
                                         eps=1.0e-4,
                                         betas=(0.9, 0.99))


    #cosine is default for LLM
    central_learning_rate_scheduler = get_cosine_schedule_with_warmup(
                central_optimizer,
                num_warmup_steps=10,
                num_training_steps=100, #should be equal to central_num_iterations
                num_cycles=0.5) #default

    model = PyTorchModel(
        model=hf_model,
        local_optimizer_create=torch.optim.SGD,
        central_optimizer=central_optimizer,
        central_learning_rate_scheduler=central_learning_rate_scheduler,
        amp_dtype=getattr(torch, "bfloat16"),
        grad_scaling=False,
        model_dtype_same_as_amp=False,
        use_torch_compile=True)

    backend = SimulatedBackend(training_data=training_data,
                               val_data=val_data,
                               postprocessors=[central_privacy])#central_privacy

    model_train_params = NNTrainHyperParams(
        local_learning_rate=0.01, #edit here
        local_num_epochs=1,
        local_batch_size=16, #control here
        local_max_grad_norm=1.0,
        grad_accumulation_steps=1)

    model_eval_params = NNEvalHyperParams(
        local_batch_size=8)

    callbacks = [
        StopwatchCallback(),
        CentralEvaluationCallback(central_data,
                                  model_eval_params=model_eval_params,
                                  frequency=1),
        AggregateMetricsToDisk(f'./metrics_final_pfl_new_{round}.csv'), #edit here
        HuggingFaceModelCheckpointingCallback(hf_model,
                                              tokenizer,
                                              f"hf_checkpoint/final_pfl_new/pfl_{round}", #edit here
                                              checkpoint_frequency=5)
    ]
    
    #Algorithm-Fedavg
    algorithm_params = NNAlgorithmParams(
                central_num_iterations=100, #around 20th epochs, it processes about 30,000 samples, which match the number of samples for centralized learning with 2 epoch
                evaluation_frequency=1,
                train_cohort_size=100, #you might want to change this
                val_cohort_size=0)
    algorithm = FederatedAveraging()
    
    print(torch.__version__)           # Check PyTorch version
    print(torch.version.cuda)   
    logger.info("Starts federated learning.")
    model = algorithm.run(algorithm_params=algorithm_params,
                          backend=backend,
                          model=model,
                          model_train_params=model_train_params,
                          model_eval_params=model_eval_params,
                          callbacks=callbacks)

for i in range(4,5):
    run_experiments(i)
# callbacks =[]

# algorithm, algorithm_params, algorithm_callbacks = get_algorithm(arguments)
# callbacks.extend(algorithm_callbacks)

# if arguments.wandb_project_id:
#     callbacks.append(
#         WandbCallback(
#             wandb_project_id=arguments.wandb_project_id,
#             wandb_experiment_name=os.environ.get('WANDB_TASK_ID',
#                                                  str(uuid4())),
#             # List of dicts to one dict.
#             wandb_config=dict(vars(arguments)),
#             tags=os.environ.get('WANDB_TAGS', 'empty-tag').split(','),
#             group=os.environ.get('WANDB_GROUP', None)))
