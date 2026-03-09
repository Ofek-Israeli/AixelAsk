"""LoRA / QLoRA model loading and adapter merge/export utilities.

``build_model_and_tokenizer`` loads a causal-LM with optional 4-bit
quantization, prepares it for LoRA training via PEFT, and applies
gradient checkpointing if configured.

``merge_and_export`` loads an adapter-only checkpoint, merges it into
the base model, and saves the full-weight model for inference server serving.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Tuple

import torch

if TYPE_CHECKING:
    from src.config import Config

logger = logging.getLogger(__name__)


def build_model_and_tokenizer(
    config: "Config",
    resolved_model_path: str,
) -> Tuple:
    """Load the base model with optional QLoRA and return ``(model, tokenizer)``.

    Parameters
    ----------
    config:
        Fully-resolved project ``Config``.
    resolved_model_path:
        Local snapshot directory returned by
        ``download_models.resolve_model_path(config)``.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    tokenizer = AutoTokenizer.from_pretrained(
        resolved_model_path,
        trust_remote_code=config.TRUST_REMOTE_CODE,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = None
    if config.TRAIN_USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.TRAIN_BNB_4BIT_QUANT_TYPE,
            bnb_4bit_compute_dtype=getattr(torch, config.TRAIN_BNB_4BIT_COMPUTE_DTYPE),
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        resolved_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="eager",
        trust_remote_code=config.TRUST_REMOTE_CODE,
    )

    if config.TRAIN_USE_4BIT:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config.TRAIN_LORA_R,
        lora_alpha=config.TRAIN_LORA_ALPHA,
        lora_dropout=config.TRAIN_LORA_DROPOUT,
        target_modules=config.TRAIN_LORA_TARGET_MODULES.split(","),
        bias=config.TRAIN_LORA_BIAS,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "LoRA model ready: %s trainable / %s total params (%.2f%%)",
        f"{trainable:,}", f"{total:,}", 100 * trainable / total if total else 0,
    )

    return model, tokenizer


def merge_and_export(
    adapter_path: str,
    output_dir: str,
    config: "Config",
    resolved_base_model_path: str,
) -> str:
    """Merge a LoRA adapter into the base model and save full weights.

    Parameters
    ----------
    adapter_path:
        Path to the adapter checkpoint directory (contains
        ``adapter_config.json`` + ``adapter_model.safetensors``).
    output_dir:
        Destination directory for the merged model.
    config:
        Fully-resolved project ``Config``.
    resolved_base_model_path:
        Local snapshot directory for the base model (same revision used
        during training).

    Returns
    -------
    str
        *output_dir* (the path where merged weights were saved).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    logger.info(
        "Merging adapter %s into base model %s → %s",
        adapter_path, resolved_base_model_path, output_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        resolved_base_model_path,
        trust_remote_code=config.TRUST_REMOTE_CODE,
    )
    model = AutoModelForCausalLM.from_pretrained(
        resolved_base_model_path,
        device_map="cpu",
        trust_remote_code=config.TRUST_REMOTE_CODE,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("Merged model saved to %s", output_dir)
    return output_dir
