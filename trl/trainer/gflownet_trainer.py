# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import math
import os
import textwrap
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Optional, Union, Callable, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import broadcast, gather_object
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    BaseImageProcessor,
    DataCollatorWithPadding,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    TrainerControl,
    is_wandb_available,
    PreTrainedModel,
    DefaultDataCollator
)
from transformers.data.data_collator import DataCollatorMixin
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import CallbackHandler, ExportableState, PrinterCallback
from transformers.utils import is_peft_available
from transformers.utils.deprecation import deprecate_kwarg

from ..core import masked_mean, masked_whiten
from ..models import create_reference_model
from ..models.utils import unwrap_model_for_generation
from .gflownet_config import GFlowNetConfig
from .utils import (
    OnlineTrainerState,
    batch_generation,
    disable_dropout_in_model,
    exact_div,
    first_true_indices,
    forward,
    generate_model_card,
    get_comet_experiment_url,
    get_reward,
    log_table_to_comet_experiment,
    peft_module_casting_to_bf16,
    prepare_deepspeed,
    print_rich_table,
    truncate_response,
    pad,
)
from ..data_utils import maybe_apply_chat_template


if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model

if is_wandb_available():
    import wandb


INVALID_LOGPROB = 1.0

@dataclass
class DataCollatorForMultipleSequences(DataCollatorMixin):
    """
    Data collator used for data with multiple sequences (useful for GFlowNet data where you may have a prompt and a continuation, etc.). 
    Inputs are dynamically padded to the maximum length of a batch if they are not all of the same length.
    Can be considered a generalization of `DataCollatorForPreference` for multiple sequences.

    Args:
        pad_token_id (`int`):
            Token ID to use for padding.
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            Type of Tensor to return. Only `"pt"` is currently supported.

    Examples:
    ```python
    >>> from trl import DataCollatorForMultipleSequences
    >>> collator = DataCollatorForMultipleSequences(pad_token_id=0)
    >>> examples = [
    ...     {"prompt_input_ids": [1, 2, 3], "chosen_input_ids": [4, 5], "rejected_input_ids": [6]},
    ...     {"prompt_input_ids": [7, 8], "chosen_input_ids": [9, 10], "rejected_input_ids": [11, 12, 13]}
    ... ]
    >>> collator(examples)
    {'prompt_input_ids': tensor([[1, 2, 3],
                                 [0, 7, 8]]),
     'prompt_attention_mask': tensor([[1, 1, 1],
                                      [0, 1, 1]]),
     'chosen_input_ids': tensor([[ 4,  5],
                                 [ 9, 10]]),
     'chosen_attention_mask': tensor([[1, 1],
                                      [1, 1]]),
     'rejected_input_ids': tensor([[ 6,  0,  0],
                                   [11, 12, 13]]),
     'rejected_attention_mask': tensor([[1, 0, 0],
                                        [1, 1, 1]])
    }
    ```
    """

    pad_token_id: int
    return_tensors: str = "pt"
    
    def __init__(self,  pad_token_id: int, sequence_prefixes: list[str]=["prompt", "continuation"], other_keys: list[str]=["baseline_ppl"]):
        self.sequence_prefixes = sequence_prefixes
        self.pad_token_id = pad_token_id
        self.other_keys = other_keys

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        output = {}
        for prefix in self.sequence_prefixes:
            # convert to tensor
            input_ids = [torch.tensor(example[prefix + "_input_ids"]) for example in examples]
            attention_mask = [torch.ones_like(input_ids) for input_ids in input_ids]
            output[prefix + "_input_ids"] = pad(input_ids, padding_value=self.pad_token_id, padding_side="left")
            output[prefix + "_attention_mask"] = pad(attention_mask, padding_value=0, padding_side="left")
        for key in self.other_keys:
            output[key] = [example[key] for example in examples]
        return output

class GFlowNetTrainer(Trainer):
    _tag_names = ["trl", "gflownet"]

    @deprecate_kwarg("config", "0.15.0", "args", warn_if_greater_or_equal_version=True, raise_if_both_names=True)
    @deprecate_kwarg(
        "tokenizer", "0.15.0", "processing_class", warn_if_greater_or_equal_version=True, raise_if_both_names=True
    )
    @deprecate_kwarg("policy", "0.15.0", "model", warn_if_greater_or_equal_version=True, raise_if_both_names=True)
    def __init__(
        self,
        args: GFlowNetConfig,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ],
        model: nn.Module,
        z_model: nn.Module,
        train_dataset: Dataset,
        reward_model: Optional[nn.Module]=None,
        data_collator: Optional[DataCollatorWithPadding] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        # less commonly used
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        callbacks: Optional[list[TrainerCallback]] = None,
        peft_config: Optional["PeftConfig"] = None,
        custom_reward_fn: Optional[Callable] = None,
    ) -> None:
        if reward_model is model:
            raise ValueError(
                "`model` and `reward_model` cannot be the same object. If you want `reward_model` to be the "
                "same as `model`, you must make a copy of it, or `None` if you use peft."
            )

        self.args = args
        self.processing_class = processing_class
        self.policy_model = model
        self.custom_reward_fn = custom_reward_fn

        # Define the collator if not provided
        if data_collator is None:
            # data_collator = DataCollatorWithPadding(self.processing_class)
            data_collator = DataCollatorForMultipleSequences(pad_token_id=self.processing_class.pad_token_id)
            # data_collator = DefaultDataCollator()

        # Initialize Z-model based on the hidden size of the policy model
        # self.z_model = nn.Sequential(
        #     nn.Linear(self.policy_model.config.hidden_size, self.policy_model.config.hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(self.policy_model.config.hidden_size, 1),
        # )
        # Based on https://arxiv.org/abs/2410.13224
        self.z_model = z_model

        self.policy_model.generation_config.eos_token_id = (
            None  # disable `pad_token_id` and `eos_token_id` because we just want to
        )
        self.policy_model.generation_config.pad_token_id = None  # generate tokens without truncation / padding

        # peft support
        if not is_peft_available() and peft_config is not None:
            raise ImportError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            # if model is a peft model and we have a peft_confg, we merge and unload it first
            if isinstance(self.policy_model, PeftModel):
                self.policy_model = self.policy_model.merge_and_unload()

            # get peft model with the given config
            self.policy_model = get_peft_model(self.policy_model, peft_config)
            if args.bf16 and getattr(self.policy_model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(self.policy_model)

        self.is_peft_model = is_peft_available() and isinstance(self.policy_model, PeftModel)

        if reward_model:
            self.reward_model = reward_model
        elif self.is_peft_model:
            self.reward_model = None
        else:
            self.reward_model = create_reference_model(self.policy_model)
        
        self.train_dataset = train_dataset
        self.train_dataset_len = len(train_dataset)
        self.data_collator = data_collator
        self.eval_dataset = eval_dataset
        self.optimizer, self.lr_scheduler = optimizers
        self.optimizer_cls_and_kwargs = None  # needed for transformers >= 4.47

        #########
        # calculate various batch sizes
        #########
        if args.total_episodes is None:  # allow the users to define episodes in terms of epochs.
            args.total_episodes = int(args.num_train_epochs * self.train_dataset_len)
        accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
        self.accelerator = accelerator
        args.world_size = accelerator.num_processes
        args.local_batch_size = (
            args.per_device_train_batch_size * args.gradient_accumulation_steps * args.num_mini_batches
        )
        args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
        args.batch_size = int(args.local_batch_size * args.world_size)
        args.mini_batch_size = exact_div(
            args.batch_size, args.num_mini_batches, "`batch_size` must be a multiple of `num_mini_batches`"
        )
        args.local_mini_batch_size = exact_div(
            args.local_batch_size, args.num_mini_batches, "`local_batch_size` must be a multiple of `num_mini_batches`"
        )
        # `per_rank_rollout_batch_size` is our `args.local_batch_size`
        # `per_rank_minibatch_size` is our `args.local_mini_batch_size`
        args.num_total_batches = math.ceil(
            args.total_episodes / args.batch_size
        )  # we may train for more than `total_episodes`
        time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
        time_int = broadcast(time_tensor, 0).item()  # avoid different timestamps across processes
        args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
        self.local_seed = args.seed + accelerator.process_index * 100003  # Prime
        if args.num_sample_generations > 0:
            self.sample_generations_freq = max(1, args.num_total_batches // args.num_sample_generations)
        self.local_dataloader_batch_size = args.local_batch_size

        #########
        # setup model, optimizer, and others
        #########
        for module in [self.policy_model, self.reward_model]:
            if module is not None:
                disable_dropout_in_model(module)
        if args.stop_token and args.stop_token == "eos":
            args.stop_token_id = processing_class.eos_token_id
        self.model = self.policy_model
        self.model.config = self.policy_model.config  # needed for pushing to hub
        self.create_optimizer_and_scheduler(
            num_training_steps=args.num_total_batches
        )  # note that we are calling `self.lr_scheduler.step()` manually only at the batch level
        # self.optimizer.param_groups.append({'params': list(self.z_model.parameters()), 'lr': 1.0})
        #########
        ### trainer specifics
        #########
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        self.callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            self.callbacks, self.model, self.processing_class, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
        self.control = TrainerControl()
        self.state = OnlineTrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ],
        )
        self.current_flos = 0
        self.hp_search_backend = None
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        #########
        ### setup dataloader
        #########
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.local_dataloader_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            drop_last=True,  # needed; otherwise the last batch will be of ragged shape
        )
        # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
        # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
        torch.manual_seed(args.seed)
        
        self.z_model, self.model, self.optimizer, self.dataloader = accelerator.prepare(self.z_model, self.model, self.optimizer, self.dataloader)
        
        torch.manual_seed(self.local_seed)  # reset the local seed again

        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=True,
        )  # no need to shuffle eval dataset
        self.eval_dataloader = accelerator.prepare(self.eval_dataloader)

        if self.is_deepspeed_enabled:
            if self.reward_model is None:
                if not self.is_peft_model:
                    raise ValueError("No reward model and model is not a Peft model. You should subclass GFlowNetTrainer if your reward does not required a model.")
            else:
                self.reward_model = prepare_deepspeed(
                    self.reward_model, args.per_device_train_batch_size, args.fp16, args.bf16
                )
            self.z_model = prepare_deepspeed(
                self.z_model, args.per_device_train_batch_size, args.fp16, args.bf16
            )
        else:
            if self.reward_model is None:
                if not self.is_peft_model:
                    raise ValueError("No reference model and model is not a Peft model.")
            else:
                self.reward_model = self.reward_model.to(self.accelerator.device)
            self.z_model = self.z_model.to(self.accelerator.device)

    def get_train_dataloader(self) -> DataLoader:
        return self.dataloader

    def get_eval_dataloader(self) -> DataLoader:
        return self.eval_dataloader

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        backup_model = self.model
        # self.model = self.model.policy  # save only the policy

        if self.is_deepspeed_enabled:
            backup_deepspeed = self.deepspeed
            self.deepspeed = self.model

        super().save_model(output_dir, _internal_call)

        self.model = backup_model

        if self.is_deepspeed_enabled:
            self.deepspeed = backup_deepspeed

    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        reward_model = self.reward_model
        z_model = self.z_model  # Added Z-model for normalization constant
        processing_class = self.processing_class
        dataloader = self.dataloader
        device = accelerator.device
        custom_reward_fn = self.custom_reward_fn # optionally used to compute rewards

        def repeat_generator():
            while True:
                yield from dataloader

        iter_dataloader = iter(repeat_generator())
        generation_config = GenerationConfig(
            max_new_tokens=args.response_length,
            temperature=(args.temperature + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        accelerator.print("===training policy===")
        start_time = time.time()
        stats_shape = (args.num_gflownet_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        entropy_stats = torch.zeros(stats_shape, device=device)
        loss_stats = torch.zeros(stats_shape, device=device)
        model.train()

        # trainer state initialization
        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = args.num_total_batches * args.num_mini_batches
        self.state.num_train_epochs = args.total_episodes / self.train_dataset_len
        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(self.state.max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(self.state.max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(self.state.max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model
            self.model_wrapped = self.model

        for update in range(1, args.num_total_batches + 1):
            self.state.episode += 1 * args.batch_size
            data = next(iter_dataloader)
            with torch.no_grad():
                queries = data["prompt_input_ids"].to(device)
                
                context_length = queries.shape[1]
                responses = []
                rewards = []
                # postprocessed_responses = []
                logprobs = []
                ref_logprobs = []
                # sequence_lengths = []
                with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                    query_responses, logitss = batch_generation(
                        unwrapped_model,
                        queries,
                        args.local_rollout_forward_batch_size,
                        processing_class.pad_token_id,
                        generation_config,
                    )

                for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                    query = queries[i : i + args.local_rollout_forward_batch_size]
                    query_response = query_responses[i : i + args.local_rollout_forward_batch_size]
                    response = query_response[:, context_length:]
                    logits = logitss[i : i + args.local_rollout_forward_batch_size]
                    all_logprob = F.log_softmax(logits, dim=-1)
                    logprob = torch.gather(all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                    del logits, all_logprob
                    torch.cuda.empty_cache()
                    
                    if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        response = truncate_response(
                            args.stop_token_id, processing_class.pad_token_id, response
                        )

                    # TODO: FIX OUT THIS IS DONE
                    # Basically what I want to do is take the response,
                    # and add it to the query, then compute the reward
                    # the reward will depend on another piece of information
                    # (the next chapter)
                    # likely this concatenation step should be a custom function
                    # we're kind of doing it here, but not really
                    
                    if custom_reward_fn is not None:
                        # giving this reward_fn as much information as possible
                        # to make it easier to compute the reward regardless of what it actually needs
                        reward = custom_reward_fn(
                            query, 
                            response, 
                            data,
                            tokenizer=processing_class,
                            reward_model=reward_model,
                        )
                    else:
                        # postprocessed_query_response = torch.cat((query, response), 1)
                        _, reward, _ = get_reward(
                            reward_model,
                            response,
                            processing_class.pad_token_id,
                            context_length,
                        )
                        

                    responses.append(response)
                    logprobs.append(logprob)
                    rewards.append(reward)
                    
                responses = torch.cat(responses, 0)
                # postprocessed_responses = torch.cat(postprocessed_responses, 0)
                logprobs = torch.cat(logprobs, 0)
                rewards = torch.cat(rewards, 0)
                del (logprob, unwrapped_model)
                torch.cuda.empty_cache()
                
                gc.collect()

                # Unsure if this is needed, but penalize if the response doesn't end with eos
                # contain_eos_token = torch.any(postprocessed_responses == self.processing_class.eos_token_id, dim=-1)
                # if self.args.missing_eos_penalty is not None:
                #     rewards[~contain_eos_token] -= self.args.missing_eos_penalty

                torch.cuda.empty_cache()

            # Do multiple epochs of GFlowNet training, with a fresh random shuffle in each epoch
            for gflownet_epoch_idx in range(args.num_gflownet_epochs):
                b_inds = np.random.permutation(args.local_batch_size)
                minibatch_idx = 0
                for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0
                    for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
                        with accelerator.accumulate(model):
                            micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                            micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                            
                            
                            mb_queries = queries[micro_batch_inds]
                            mb_responses = responses[micro_batch_inds]
                            mb_logprobs = logprobs[micro_batch_inds]
                            mb_rewards = rewards[micro_batch_inds]

                            with torch.no_grad():
                                # just to get the final hidden state, we don't want to compute gradients for this pass
                                # use reward model since it shouldn't change
                                if self.reward_model is not None:
                                    output = self.reward_model(
                                        mb_queries,
                                        output_hidden_states=True,
                                        return_dict=True,
                                    )
                                else:  # peft case: we just need to disable the adapter
                                    with model.disable_adapter():
                                        output = model(
                                        mb_queries,
                                        output_hidden_states=True,
                                        return_dict=True,
                                    )
                                
                                # get the hidden state at the final token
                                # using last layer's final token:
                                final_hidden_state = output.hidden_states[-1][:, -1, :]  # Shape: (batch_size, hidden_size)
                                del output
                            
                            # Compute normalization constant Z
                            z_pred_mb = z_model(final_hidden_state)  # Predict Z using the input prompt
                            
                            # Compute trajectory balance loss
                            trajectory_probs = mb_logprobs.sum(dim=1)
                            log_reward = mb_rewards.log().to(device)
                            
                            # print(f"z_pred_mb: {z_pred_mb}")
                            # print(f"trajectory_probs: {trajectory_probs}")
                            # print(f"log_reward: {log_reward}")
                            
                            tb_loss = (
                                (z_pred_mb + trajectory_probs - log_reward) ** 2
                            ).mean()
                            
                            loss = tb_loss
                            
                            accelerator.backward(loss)
                            optimizer.step()
                            optimizer.zero_grad()
                            with torch.no_grad():
                                per_token_prob = torch.exp(mb_logprobs)
                                sequence_entropy = -(per_token_prob * mb_logprobs).sum(dim=1)
                                entropy = sequence_entropy.mean()
                                entropy_stats[gflownet_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy
                                loss_stats[gflownet_epoch_idx, minibatch_idx, gradient_accumulation_idx] = loss.mean()
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1
                    # del everything and empty cache
                    # fmt: off
                    del (
                        loss, entropy,
                        mb_responses, mb_logprobs,
                    )
                    # fmt: on
                    torch.cuda.empty_cache()
            
            # log metrics
            with torch.no_grad():
                mean_entropy = (-logprobs).sum(1).mean()
                eps = int(self.state.episode / (time.time() - start_time))
                metrics = {}
                metrics["eps"] = eps
                metrics["objective/entropy"] = self.accelerator.gather_for_metrics(mean_entropy).mean().item()
                metrics["objective/rewards"] = self.accelerator.gather_for_metrics(rewards.mean()).mean().item()
                metrics["policy/entropy_avg"] = self.accelerator.gather_for_metrics(entropy_stats).mean().item()
                metrics["loss/policy_avg"] = self.accelerator.gather_for_metrics(loss_stats).mean().item()
                metrics["val/num_eos_tokens"] = (responses == processing_class.eos_token_id).sum().item()
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                metrics["episode"] = self.state.episode
                self.state.epoch = self.state.episode / self.train_dataset_len  # used by self.log
                self.state.global_step += 1
                self.log(metrics)

            self.lr_scheduler.step()
            self.control = self.callback_handler.on_step_end(args, self.state, self.control)
            if self.control.should_save:
                self._save_checkpoint(model, trial=None)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            del mean_entropy, rewards, metrics
            torch.cuda.empty_cache()
            gc.collect()

            if args.num_sample_generations > 0 and (update - 1) % self.sample_generations_freq == 0:
                self.generate_completions(sampling=True)
                torch.cuda.empty_cache()
            del (
                query_responses,
                responses,
                # postprocessed_responses,
                logprobs,
                ref_logprobs,
                # sequence_lengths,
                # contain_eos_token,
                # rewards,
                # z_logits,
            )
            torch.cuda.empty_cache()

        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def generate_completions(self, sampling: bool = False):
        args = self.args
        processing_class = self.processing_class
        generation_config = GenerationConfig(
            max_new_tokens=self.args.response_length,
            temperature=(0.01 + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        table = defaultdict(list)
        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            for batch in self.eval_dataloader:
                query = batch["prompt_input_ids"]
                with torch.no_grad():
                    context_length = query.shape[1]
                    query_response, _ = batch_generation(
                        unwrapped_model,
                        query,
                        query.shape[0],
                        processing_class.pad_token_id,
                        generation_config,
                    )
                    response = query_response[:, context_length:]
                    # postprocessed_response = response
                    if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        response = truncate_response(
                            args.stop_token_id, processing_class.pad_token_id, response
                        )
                    table["query"].extend(
                        gather_object(processing_class.batch_decode(query, skip_special_tokens=True))
                    )
                    table["model response"].extend(
                        # gather_object(processing_class.batch_decode(postprocessed_response))
                        gather_object(processing_class.batch_decode(response))
                    )
                    
                    if self.custom_reward_fn is not None:
                        reward = self.custom_reward_fn(
                            query, 
                            response, 
                            batch,
                            tokenizer=processing_class,
                            reward_model=self.reward_model,
                        )
                    else:
                        _, reward, _ = get_reward(
                            self.reward_model,
                            response,
                            processing_class.pad_token_id,
                            context_length,
                        )
                    
                    table["score"].extend(self.accelerator.gather_for_metrics(reward).float().cpu().numpy())

                if sampling:
                    break
        df = pd.DataFrame(table)

        if self.accelerator.is_main_process:
            print_rich_table(df.iloc[0 : 0 + 5])
            if "wandb" in args.report_to:
                import wandb

                if wandb.run is not None:
                    wandb.log({"completions": wandb.Table(dataframe=df)})

            if "comet_ml" in args.report_to:
                log_table_to_comet_experiment(
                    name="completions.csv",
                    table=df,
                )