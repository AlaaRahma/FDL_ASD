"""Encoders in three fine-tuning modes (unified_v2): full, LoRA (q/k/v, r16, alpha32, pretrained pooler frozen,
classifier trained), QLoRA (same adapters, NF4 double-quantised frozen base, bf16 compute)."""
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .common import HP, MODELS


def tokenizer_for(model_key):
    return AutoTokenizer.from_pretrained(MODELS[model_key])


def build(model_key, mode, device, lora_r=None, freeze_a=False, lora_seed=0):
    name = MODELS[model_key]
    id2label = {0: 'not_sarcastic', 1: 'sarcastic'}
    kw = dict(num_labels=2, id2label=id2label, label2id={v: k for k, v in id2label.items()})
    if mode == 'qlora':
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=['classifier', 'pooler'])
        model = AutoModelForSequenceClassification.from_pretrained(
            name, quantization_config=q, torch_dtype=torch.float32, device_map={'': device.index or 0}, **kw)
        for p in model.parameters():
            if type(p).__name__ != 'Params4bit' and not p.data.is_contiguous():
                p.data = p.data.contiguous()
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(name, **kw)
        for p in model.parameters():  # MARBERTv2 pooler loads as a transposed view
            if not p.data.is_contiguous():
                p.data = p.data.contiguous()
    if mode in ('lora', 'qlora'):
        torch.manual_seed(lora_seed)  # identical adapter initialisation on every client
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.SEQ_CLS, r=lora_r or HP['lora_r'], lora_alpha=HP['lora_alpha'],
            lora_dropout=HP['lora_dropout'], target_modules=HP['lora_targets'], modules_to_save=['classifier']))
        if freeze_a:  # FFA-LoRA: A fixed at its (shared) random initialisation, only B and the head are trained
            for n, p in model.named_parameters():
                if 'lora_A' in n:
                    p.requires_grad_(False)
    if mode != 'qlora':
        model.to(device)
    return model


def trainable_names(model):
    return [n for n, p in model.named_parameters() if p.requires_grad]


@torch.no_grad()
def get_state(model, names, device=None):
    params = dict(model.named_parameters())
    return {n: params[n].detach().to(device or params[n].device, dtype=torch.float32, copy=True) for n in names}


@torch.no_grad()
def set_state(model, state):
    params = dict(model.named_parameters())
    for n, v in state.items():
        params[n].data.copy_(v.to(params[n].device, dtype=params[n].dtype))


def count_params(model):
    total = 0
    for p in model.parameters():
        total += p.numel() * (2 if type(p).__name__ == 'Params4bit' else 1)  # 4-bit packs two weights per byte
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def lora_pairs(names):
    """[(prefix, A_name, B_name)] for every adapted linear layer."""
    out = []
    for n in names:
        if 'lora_A' in n:
            b = n.replace('lora_A', 'lora_B')
            out.append((n.split('.lora_A')[0], n, b))
    return out


def lora_scaling(model):
    for m in model.modules():
        if hasattr(m, 'scaling') and isinstance(m.scaling, dict) and 'default' in m.scaling:
            return float(m.scaling['default'])
    return HP['lora_alpha'] / HP['lora_r']
