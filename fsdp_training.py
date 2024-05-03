import json
import os
from tqdm.auto import tqdm
from datasets import Dataset
from transformers import DataCollatorWithPadding, AdamW, AutoTokenizer, LlamaForCausalLM, LlamaConfig

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.multiprocessing as mp

from data_utils import pad_and_mask, download_from_huggingface


def pcfg_dataset_to_dataloader(
    pcfg_dataset,
    padder_tokenizer,
    batch_size=8,
    context_length=256,
    dataset_name="",
    rank=0,
    world_size=1,
):
    if "code" in dataset_name:
        tok_seqs = pcfg_dataset
    else:
        tok_seqs = [[int(tok) for tok in doc.split(" ")] for doc in pcfg_dataset]

    input_ids, attention_masks = [], []
    for seq in tok_seqs:
        padded_seq, mask = pad_and_mask(seq, context_length)
        input_ids.append(padded_seq)
        attention_masks.append(mask)

    tokenized_dataset = Dataset.from_dict(
        {"input_ids": input_ids, "attention_mask": attention_masks}
    )
    tokenized_dataset = tokenized_dataset.map(
        lambda x: {"labels": x["input_ids"].copy()}, batched=True
    )
    tokenized_dataset.set_format("torch")

    data_collator = DataCollatorWithPadding(tokenizer=padder_tokenizer)

    data_sampler = DistributedSampler(  # TODO: refactor via `distributed` flag back into original pcfg_dataset_to_dataloader
        tokenized_dataset, rank=rank, num_replicas=world_size, shuffle=True
    )

    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        sampler=data_sampler,
        # CUDA args:
        num_workers=2,
        pin_memory=True,
        shuffle=False,
    )

    return dataloader

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run_fsdp_training(
    model, train_dataloader, valid_dataloader, optimizer, num_epochs=10, rank=0
):
    train_loss = []
    valid_loss = []

    for epoch in range(num_epochs):
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=len(train_dataloader))
        progress_bar = tqdm(
            range(len(train_dataloader)), desc=f"Epoch {epoch + 1}/{num_epochs}"
        )
        ddp_loss = torch.zeros(2).to(rank)

        model.train()
        for batch in train_dataloader:
            batch = {k: v.to(rank) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            progress_bar.update(1)

            train_loss.append(loss.item())
            ddp_loss[0] += loss.item()
            ddp_loss[1] += len(batch)

            lr_scheduler.step()

        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    return train_loss, valid_loss


def fsdp_main(rank, world_size, args):
    setup(rank, world_size)

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf", token="[REDACTED]"
    )
    tokenizer.add_special_tokens({"pad_token": "<pad>"})

    dataset_name = "khoomeik/gzipscale-code-C-8000M"
    pcfg_dataset = download_from_huggingface(dataset_name)
    train_dataloader = pcfg_dataset_to_dataloader(
        pcfg_dataset,
        padder_tokenizer=tokenizer,
        batch_size=32,
        dataset_name=dataset_name,
        rank=rank,
        world_size=world_size,
    )

    torch.cuda.set_device(rank)

    model_config_dict = {
        "vocab_size": 32001,
        "hidden_size": 1024,
        "intermediate_size": 2048,
        "num_hidden_layers": 32,
        "num_attention_heads": 16,
        "max_position_embeddings": 256,
    }
    model_config = LlamaConfig(**model_config_dict)
    model = LlamaForCausalLM(model_config)
    model_size = sum(p.numel() for p in model.parameters())
    print(f"Model Size: {model_size/1_000_000:.1f}M")

    model.to(rank)
    model = FSDP(model)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 1

    train_loss, valid_loss = run_fsdp_training(
        model,
        train_dataloader,
        None,
        optimizer,
        num_epochs=num_epochs,
        rank=rank,
    )

    row = {
        "dataset_name": dataset_name,
        # "token_ct": train_token_ct,
        "model_stats": model_config_dict,
        "model_size": model_size,
        "num_epochs": num_epochs,
        "train_loss": train_loss,
        # "valid_loss": valid_loss,
        "cuda_rank": rank,
    }

    with open("results_fsdp.jsonl", "a") as file:
        file.write(json.dumps(row) + "\n")

    dist.barrier()
    states = model.state_dict()
    if rank == 0:
        torch.save(states, f"./model_{model_size}_{dataset_name.split('/')[-1]}.pt")

    cleanup()


if __name__ == "__main__":
    args = {}

    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(fsdp_main, args=(WORLD_SIZE, args), nprocs=WORLD_SIZE, join=True)
