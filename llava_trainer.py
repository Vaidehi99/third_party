import os
import torch

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    ShardedDDPOption,
    logger,
)
from typing import List, Optional
import sys
sys.path.append("/nas-ssd2/vaidehi/nlp13/belief-localization/third_party/")
from util.fewshot_utils import score_from_batch
import torch.nn as nn
# from experiments.evaluate_llava_mm import make_decoder
from torch.distributions import Categorical
sys.path.append("/nas-ssd2/vaidehi/InfoDeletionAttacks/")
from transformer_utils.src.transformer_utils.util.module_utils import get_child_module_by_names



_RESID_SUFFIXES = {".attn", ".mlp"}

def _locate_special_modules(model):
    #if not hasattr(model, "_blocks_input_getter"):
    #    model._blocks_input_getter = blocks_input_locator(model)
   
    if not hasattr(model, "_ln_f_getter"):
        model._ln_f_getter = final_layernorm_locator(model)

def blocks_input_locator(model: nn.Module):
    """
    HF usually (always?) places a dropout after the input embeddings.
    TODO: avoid depending on this
    """
    dropouts_on_base_model = [
        mod for mod in model.base_model.children()
        if isinstance(mod, nn.Dropout)
    ]
    if len(dropouts_on_base_model) > 0:
        return lambda: dropouts_on_base_model[0]
    raise ValueError('could not identify blocks input')

def final_layernorm_locator(model: nn.Module):
    layernorms_on_base_model = [mod for mod in model.base_model.base_model.children()][2]
    # print(layernorms_on_base_model)
#[
#        mod for mod in model.base_model.children()
#        if LlamaRMSNorm in mod #isinstance(mod, nn.LayerNorm)
#    ]
#    if len(layernorms_on_base_model) > 0:
#        return lambda: layernorms_on_base_model[0]
    # print(layernorms_on_base_model)
    # print(lambda: layernorms_on_base_model)
    return lambda: layernorms_on_base_model
    raise ValueError('could not identify ln_f')

def _get_layer(model, name):
    #if name == "input":
    #    return model._blocks_input_getter()
    if name == "final_layernorm":
        return model._ln_f_getter()

    model_with_module = model if name == "lm_head" else model.base_model
    return get_child_module_by_names(model_with_module, name.split("."))


def _sqz(x):
    if isinstance(x, torch.Tensor):
        return x
    try:
        return x[0]
    except:
        return x


def _get_layer_and_compose_with_ln(model, name):
    if name.endswith('.attn'):
        lname = name[:-len('.attn')] + '.ln_1'
        ln = _get_layer(model, lname)
    elif name.endswith('.mlp'):
        lname = name[:-len('.mlp')] + '.ln_2'
        ln = _get_layer(model, lname)
    else:
        ln = lambda x: x
    return lambda x: _get_layer(model, name)(ln(x))


def make_decoder(model, decoder_layer_names=['final_layernorm', 'lm_head']):
    _locate_special_modules(model)

    decoder_layers = [_get_layer_and_compose_with_ln(model, name) for name in decoder_layer_names]
    
    def _decoder(x):
        for name, layer in zip(decoder_layer_names, decoder_layers):
            layer_out = _sqz(layer(_sqz(x)))

            # TODO: DRY
            is_resid = any([name.endswith(s) for s in _RESID_SUFFIXES])
            if is_resid:
                x = x + layer_out
            else:
                x = layer_out
        return x
    return _decoder



def collect_logits(model, hidden_states, layers=[31, 32]):
    if True:

        max_layers = len(hidden_states)

        hidden_states = torch.stack([hidden_states[i-max_layers] for i in layers], dim=0)

        lens_decoder = make_decoder(model, decoder_layer_names=['final_layernorm', 'lm_head'])
        decoder_out = lens_decoder(hidden_states)



    layer_logits = torch.nn.Softmax(dim=-1)(decoder_out)
    return layer_logits


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):
    def compute_loss(self, model, inputs):
        print(self.args.defense)
        # print(inputs)
        # print(len(inputs['input_ids']))
        # print(inputs['input_ids'][0].shape)
        # print(len(inputs['images']))
        # print(inputs['images'][0].shape)
        # print(len(['target_ids']))
        # print(inputs['target_ids'][0].shape)
   
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        
        seq_log_probs = score_from_batch(model, inputs, return_log_probs=True)
        nll = -seq_log_probs.sum()
        pred_prob = torch.exp(-nll)

        if self.args.defense=="fact_erasure":
            if self.args.margin_loss:
                margin_layers = [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
                model_batch = {}
                for key in ["input_ids", "attention_mask"]:
                    model_batch[key] = inputs[key][:,:-1]
                model_batch["images"] = inputs["images"]
                k_new = 100

                hidden_states = model(**model_batch, output_hidden_states=True).hidden_states
                hidden_states = [hid[:, -model_batch["input_ids"].shape[1]:, :] for hid in hidden_states]
                layer_logits = collect_logits(model, hidden_states, layers=margin_layers)
                last_nonzero_mask = torch.remainder(torch.argmin(model_batch["attention_mask"], -1)-1, model_batch["attention_mask"].shape[-1])
                target_shape = [layer_logits.shape[0],layer_logits.shape[1],1, layer_logits.shape[3]]
                expanded_last_nonzero_mask = last_nonzero_mask.view(1,-1,1,1).expand(target_shape)
                layer_logits = torch.gather(layer_logits, 2, expanded_last_nonzero_mask).squeeze(2)
                sorted_layer_logits, _ = torch.sort(layer_logits, -1)
                min_topk_prob = sorted_layer_logits[:,:,-(k_new)]
                max_bottomk_prob = sorted_layer_logits[:,:,k_new-1]
                # print(inputs["target_ids"][0][-1])
                # print(layer_logits.shape)
                # print(layer_logits[:,:,inputs["target_ids"][0][-1]])
                target_prob = layer_logits[:,:,inputs["target_ids"][0][-1]]
                margin_top = torch.maximum(torch.zeros_like(target_prob), target_prob-min_topk_prob)#*input_tok["attention_mask"].expand(target_prob.shape[0], -1, -1)
                margin_bottom = torch.maximum(torch.zeros_like(target_prob), max_bottomk_prob-target_prob)#*input_tok["attention_mask"].expand(target_prob.shape[0], -1, -1)
                margin_loss_top = margin_top.mean()
                margin_loss_bottom = margin_bottom.mean()
                loss = margin_loss_top + margin_loss_bottom
                print(f"Loss: {loss} = {margin_loss_top} + {margin_loss_bottom}")
                return loss

            elif self.args.entropy_loss:
                entropy_layers = [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
                # hidden_states = model(**inputs, output_hidden_states=True).hidden_states
                # hidden_states = [hid[:, -inputs["input_ids"].shape[1]:, :] for hid in hidden_states]
                # layer_logits = collect_logits(model, hidden_states, layers=self.args.margin_layers)
                # last_nonzero_mask = torch.remainder(torch.argmin(inputs["attention_mask"], -1)-1, inputs["attention_mask"].shape[-1])
                # target_shape = [layer_logits.shape[0],layer_logits.shape[1],1, layer_logits.shape[3]]
                # expanded_last_nonzero_mask = last_nonzero_mask.view(1,-1,1,1).expand(target_shape)
                # layer_logits = torch.gather(layer_logits, 2, expanded_last_nonzero_mask).squeeze(2)
                model_batch = {}
                for key in ["input_ids", "attention_mask"]:
                    model_batch[key] = inputs[key][:,:-1]
                model_batch["images"] = inputs["images"]
                k_new = 100

                hidden_states = model(**model_batch, output_hidden_states=True).hidden_states
                hidden_states = [hid[:, -model_batch["input_ids"].shape[1]:, :] for hid in hidden_states]
                layer_logits = collect_logits(model, hidden_states, layers=entropy_layers)
                last_nonzero_mask = torch.remainder(torch.argmin(model_batch["attention_mask"], -1)-1, model_batch["attention_mask"].shape[-1])
                target_shape = [layer_logits.shape[0],layer_logits.shape[1],1, layer_logits.shape[3]]
                expanded_last_nonzero_mask = last_nonzero_mask.view(1,-1,1,1).expand(target_shape)
                layer_logits = torch.gather(layer_logits, 2, expanded_last_nonzero_mask).squeeze(2)

                entropy = Categorical(probs = layer_logits.float()).entropy().mean()
                loss = -entropy
                print(f"Loss: {loss}")
                return loss

            else:
                print(f"Loss: {pred_prob}")
                return pred_prob
        if self.args.defense=="error_injection" or self.args.defense=="empty_response":
            print(f"Loss: {nll}")
            return nll
        
        # print(x)
        # outputs = model(**inputs)

        # print(outputs.logits.shape)
        # exit()
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        if self.args.defense=="fact_erasure":
            print(f"Loss: {loss}")
            return -loss
        else:
            print(f"Loss: {loss}")
            return loss
        
    def compute_loss_fact_erasure_works(self, model, inputs):
        print(self.args.defense)

   
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        if self.args.defense=="fact_erasure":
            print(f"Loss: {loss}")
            return -loss
        else:
            print(f"Loss: {loss}")
            return loss        


    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    skipped = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                            logger.info(f"skipped {module}: {skipped/2**20}M params")
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                    logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)

    # def _prepare_inputs(self, inputs):
    #     """
    #     Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
    #     handling potential state.
    #     """
    #     print(inputs.items())
    #     for k, v in inputs.items():
    #         if isinstance(v, torch.Tensor):
    #             inputs[k] = v.to(self.args.device)

    #     if self.args.past_index >= 0 and self._past is not None:
    #         inputs["mems"] = self._past

    #     print(inputs['input_ids'].shape)
    #     ids = inputs['input_ids']
    #     print(torch.where(ids==-200))
    #     ids[ids==-200] = 2

    #     print(self.tokenizer.batch_decode(inputs['input_ids']))
    #     exit()

    #     return inputs
