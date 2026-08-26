import transformers
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig,get_peft_model
from model.modeling_llada import LLaDAModelLM
from model.configuration_llada import LLaDAConfig
from utils.sparse_dream import install_sparse_prefill_patch

def get_model_by_config(config):
    """Select different models based on config file"""
    training_mode = config.get('training_mode', 'dream')
    
    if training_mode == 'llada':
        return get_llada(config)
    elif training_mode in {'dream', 'dream_full_long'}:
        model, tokenizer = get_model(config)
        if training_mode == 'dream_full_long':
            install_sparse_prefill_patch(model)
            model.gradient_checkpointing_enable()
            if hasattr(model, 'enable_input_require_grads'):
                model.enable_input_require_grads()
            model.config.use_cache = False
        return model, tokenizer
    else:
        raise ValueError(f"Unsupported training mode: {training_mode}")

def get_model(config):
    # Use path from config, use default path if no config
    model_path = config.paths.model if hasattr(config, 'paths') and hasattr(config.paths, 'model') else "/home/wx/data/model/Dream-org/Dream-v0-Base-7B"
    
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    # print(model.named_modules())
    # print(model,"model
    for param in model.parameters():
        param.requires_grad = False
    tokenizer=AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    peft_config = LoraConfig(r=32, lora_alpha=32, lora_dropout=0.1,target_modules=["q_proj", "v_proj","k_proj", "o_proj"],)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, tokenizer

def get_llada(config):
    # Use path from config, use default path if no config
    model_path = config.paths.model if hasattr(config, 'paths') and hasattr(config.paths, 'model') else "/data1/xck/models/llada-8b-instruct"
    
    config_obj=LLaDAConfig.from_pretrained(model_path)
    model = LLaDAModelLM.from_pretrained(model_path,config=config_obj)
    # print(model.named_modules())
    # print(model,"model
    # print(model)
    # exit()
    for param in model.parameters():
        param.requires_grad = False
    tokenizer=AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    peft_config = LoraConfig(r=32, lora_alpha=32, lora_dropout=0.1,target_modules=["q_proj", "v_proj","k_proj", "attn_out"],)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, tokenizer
# def create_attention_mask(input_ids, mask_id):
#     """
#     Create an attention mask based on the input_ids and mask_id.

#     Args:
#         input_ids (torch.Tensor): The input tensor of shape (batch_size, sequence_length).
#         mask_id (int): The ID of the mask token.

#     Returns:
#         torch.Tensor: The attention mask of shape (batch_size, sequence_length, sequence_length).
