# from peft import PeftModel, PeftConfig, get_peft_model
from datasets import load_dataset
from torch.utils.data import DataLoader
from peft import PeftModel, PeftConfig, get_peft_model
from utils.util import flatten_dict,shift_logits
from utils.data import get_bs17k_dataloader,get_llada_bs17k_dataloader,get_dataloader_by_config
from utils.model import get_model,get_llada,get_model_by_config
from utils.loss import compute_loss,compute_llada_loss,compute_normal_loss,compute_loss_by_config
from utils.generation import sample_tokens
# import dataloader

import os
import torch
import argparse
import torch.distributed as dist
from omegaconf import OmegaConf
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def get_accelerator(config, global_config):
    # Select experiment path based on config
    if hasattr(global_config, 'paths') and hasattr(global_config.paths, 'experiment'):
        root_path = global_config.paths.experiment
    else:
        root_path = config.root if hasattr(config, 'root') else '/tmp/experiment'
    
    output_dir = os.path.join(root_path, config.exp_name, config.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logging_dir = os.path.join(output_dir, config.logging_dir)
    project_config = ProjectConfiguration(project_dir=config.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        log_with=None if config.report_to == 'no' else config.report_to,
        mixed_precision=config.mixed_precision,
        project_config=project_config,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )

    return accelerator, output_dir

def main(args):
    config = OmegaConf.load(args.config)
    accelerator, output_dir = get_accelerator(config.train, config)
    
    # Use unified model and data loading functions
    denoiser, tokenizer = get_model_by_config(config)
    dataloader = get_dataloader_by_config(
        tokenizer, config.data, config, max_length=config.data.max_length
    )
    
    if config.train.decoder_resume_path is not None:
        ckpt = torch.load(config.train.decoder_resume_path, map_location='cpu', weights_only=True)
        if config.train.skipped_keys:
            ckpt = {k: v for k, v in ckpt.items() if k not in config.train.skipped_keys}
        m, u = denoiser.load_state_dict(ckpt, strict=False)
        if accelerator.is_main_process:
            print(f'model ckpt loaded from {config.train.decoder_resume_path}')

        # ckpt = torch.load(config.train.head_resume_path, map_location='cpu', weights_only=True)
        # if config.train.skipped_keys:
        #     ckpt = {k: v for k, v in ckpt.items() if k not in config.train.skipped_keys}
        # m, u = denoiser.lm_head.load_state_dict(ckpt, strict=False)
        # if accelerator.is_main_process:
        #     print(f'model ckpt loaded from {config.train.head_resume_path}')

    global_step = config.train.global_step if config.train.global_step is not None else 0
    params_to_learn = list(param for param in denoiser.parameters() if param.requires_grad)
    optimizer = torch.optim.AdamW(
        params_to_learn,
        lr           = config.train.lr,
        betas        = (0.9, 0.95),
        weight_decay = 5e-2,
        eps          = 1e-8,
    )
    
    denoiser, dataloader, optimizer = accelerator.prepare(denoiser, dataloader, optimizer)

    config.device_count = accelerator.num_processes
    if accelerator.is_main_process:
        accelerator.init_trackers(config.train.wandb_proj, config=flatten_dict(config))

    training_done = False
    epoch = 0
    progress_bar = tqdm(
        total   = config.train.num_iters,
        initial = global_step,
        desc    = 'Steps',
        disable = not accelerator.is_local_main_process,
    )

    if accelerator.is_main_process:
        print(f'Learnable parameters: {sum(p.numel() for p in params_to_learn if p.requires_grad) / 1e9} B')

    while not training_done:
        if accelerator.is_main_process:
            print(f'Epoch: {epoch}')
        for batch in dataloader:
            with accelerator.accumulate([denoiser]):
                denoiser.train()
                input_ids = batch['data']
                # print("input_ids",input_ids.dtype)
                question_length = batch['question_length']
                sequence_length = batch.get('sequence_length')
                
                # Use unified loss function selection
                losses = compute_loss_by_config(
                    input_ids,
                    denoiser,
                    question_length,
                    block_size    = config.train.block_size,
                    mask_id       = config.denoiser.encoder.mask_id,
                    enable_shift  = config.train.enable_shift,
                    share_steps   = config.train.share_steps,
                    self_align    = config.train.self_align,
                    feature_align = config.train.feature_align,
                    self_step     = config.train.self_step,
                    eos_id        = tokenizer.eos_token_id,
                    config        = config,
                    sequence_length = sequence_length,
                )
                
                if config.train.share_steps > 1:
                    loss_tgt = losses['loss']
                    # loss_1 = losses['loss_1']
                    # loss_2 = losses['loss_2']
                else:
                    raise NotImplementedError
                torch.cuda.empty_cache()
                accelerator.backward(loss_tgt)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_learn, 1.0)

                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                logs = dict()
                loss_tgt = accelerator.gather(loss_tgt.detach()).mean().item()
                logs['loss'] = loss_tgt
                # if config.train.share_steps > 1:
                #     loss_1 = accelerator.gather(loss_1.detach()).mean().item()
                #     loss_2 = accelerator.gather(loss_2.detach()).mean().item()
                    # logs['loss_1'] = loss_1
                    # logs['loss_2'] = loss_2
                    
                accelerator.log(logs, step=global_step)
                progress_bar.set_postfix(**logs)

            if (
                config.training_mode != 'dream_full_long'
                and global_step > 0
                and global_step % config.train.eval_every == 0
                and accelerator.is_main_process
            ):
                denoiser.eval();
                question = 'Henry made two stops during his 60-mile bike trip. He first stopped after 20 miles. His second stop was 15 miles before the end of the trip. How many miles did he travel between his first and second stops?'
                # prompt = tokenizer(question)['input_ids']
                # prompt = torch.tensor(prompt).to(accelerator.device).unsqueeze(0)
                messages = [
                    {"role": "user", "content": question}
                ]
                prompt = tokenizer.apply_chat_template(
                    messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
                ).input_ids
                prompt = prompt.to(accelerator.device)

                mask_id = 151666
                gen_len = 512 - prompt.shape[1]
                temperature = 0.2
                top_p = 0.95

                x_t = torch.cat([prompt, torch.tensor([[mask_id]*gen_len]).to(accelerator.device)], dim=1)
                with torch.inference_mode():
                    for i in range(gen_len):
                        mask_index = (x_t == mask_id)
                        if i % 2 == 0:
                            z_t = denoiser.module.encoder(x_t, output_hidden_states=True).hidden_states[-1]
                            hidden_state = denoiser.module.decoder(x_t, z_t)
                            logits = denoiser.module.encoder.lm_head(hidden_state)
                        else:
                            hidden_state = denoiser.module.decoder(x_t, z_t)
                            logits = denoiser.module.lm_head(hidden_state)

                        if config.train.enable_shift:
                            logits = shift_logits(logits)

                        mask_logits = logits[mask_index]
                        confidence, x0 = sample_tokens(mask_logits, temperature, top_p=top_p, top_k=None, neg_entropy=True)

                        number_transfer_tokens = 1
                        _, transfer_index = torch.topk(confidence, number_transfer_tokens)
                        x0_ = torch.zeros_like(x0, device=accelerator.device, dtype=torch.long) + mask_id
                        x0_[transfer_index] = x0[transfer_index].clone()
                        x_t[mask_index] = x0_

                answer = tokenizer.batch_decode(x_t[:, prompt.shape[1]:], skip_special_tokens=True)[0]
                print(answer)

            accelerator.wait_for_everyone()

            if global_step > 0 and global_step % config.train.save_every == 0 and accelerator.is_main_process:
                denoiser.eval()
                decoder_state_dict = accelerator.unwrap_model(denoiser).save_pretrained(os.path.join(output_dir, f"Decoder-{config.train.exp_name}-{global_step // 1000}k"))
                # lmhead_state_dict = accelerator.unwrap_model(denoiser).lm_head.state_dict()
                # torch.save(lmhead_state_dict, os.path.join(output_dir, f"LMhead-{config.train.exp_name}-{global_step // 1000}k"))
            accelerator.wait_for_everyone()
            if global_step >= config.train.num_iters:
                training_done = True
                break
        epoch += 1
    accelerator.end_training()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/dream.yaml')
    args = parser.parse_args()
    main(args)    