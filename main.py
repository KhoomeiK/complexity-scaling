import torch
from training import run_training
from transformers import AdamW, AutoTokenizer, LlamaForCausalLM, LlamaConfig
import json
from data_utils import (
    calculate_median_stdev_gzipability,
    count_total_tokens,
    pcfg_dataset_to_dataloader,
    download_from_huggingface,
)


def run_scaling_exps(cuda_idx=None):
    context_length = 256
    llm_configuration = {
        "vocab_size": 32001,
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "max_position_embeddings": context_length,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf", token="[REDACTED]"
    )
    tokenizer.add_special_tokens({"pad_token": "<pad>"})

    model_sizes = {
        "hidden_size": [64, 128, 256, 512, 1024, 2048],
        "intermediate_size": [128, 256, 512, 1024, 2048, 4096],
        "num_hidden_layers": [2, 4, 6, 10, 20, 30],
        "num_attention_heads": [1, 2, 4, 8, 16, 32],
    }

    dataset_names = [
        "khoomeik/gzipscale-0.12-10M",
        "khoomeik/gzipscale-0.23-10M",
        "khoomeik/gzipscale-0.33-10M",
        "khoomeik/gzipscale-0.45-10M",
        "khoomeik/gzipscale-0.61-10M",
    ]
    if cuda_idx is not None:
        if cuda_idx == torch.cuda.device_count(): # NOTE: this is only for handling dataset #5 and will likely break on systems with >4 GPUs
            dataset_names = [dataset_names[cuda_idx]]
            cuda_idx = torch.cuda.device_count() - 1
        else:
            dataset_names = [dataset_names[cuda_idx]]
    pcfg_datasets = [download_from_huggingface(name) for name in dataset_names]
    med_std_gzips = [
        calculate_median_stdev_gzipability(pcfg_dataset)
        for pcfg_dataset in pcfg_datasets
    ]
    for i, pcfg_dataset in enumerate(pcfg_datasets):
        med, std = med_std_gzips[i]
        total_toks = count_total_tokens(
            pcfg_dataset_to_dataloader(pcfg_dataset, padder_tokenizer=tokenizer)
        )
        print(f"{i}: {med:.3f} +- {std:.3f} ({total_toks})  | {dataset_names[i]}")

    device = f"cuda:{cuda_idx}" if cuda_idx is not None else "cpu"
    results = []

    for i, pcfg_dataset in enumerate(pcfg_datasets):
        for data_portion in (0.01, 0.1, 0.2, 0.5, 0.95):
            med_gzip, std_gzip = med_std_gzips[i]

            train_data_size = int(len(pcfg_dataset) * data_portion)
            valid_data_size = min(100, int(train_data_size / 10))
            train_dataloader = pcfg_dataset_to_dataloader(
                pcfg_dataset[:train_data_size],
                padder_tokenizer=tokenizer,
                batch_size=32,
            )
            valid_dataloader = pcfg_dataset_to_dataloader(
                pcfg_dataset[-valid_data_size:],
                padder_tokenizer=tokenizer,
                batch_size=32,
            )
            train_token_ct = count_total_tokens(train_dataloader)

            for j in range(len(list(model_sizes.values())[0])):
                print("-" * 20)

                model_stats = {key: val[j] for key, val in model_sizes.items()}
                model_config_dict = {
                    **llm_configuration,
                    **model_stats,
                }  # NOTE: update vocab_size and new tokenizer?
                model_config = LlamaConfig(**model_config_dict)
                model = LlamaForCausalLM(model_config)
                model_size = sum(p.numel() for p in model.parameters())

                print(f"Dataset Stats: {med_gzip:.3f} +- {std_gzip:.3f}")
                print(f"Model Size: {model_size/1_000_000:.1f}M")
                print(f"Train Token Count: {train_token_ct}")

                model.to(device)
                optimizer = AdamW(model.parameters(), lr=5e-5)
                num_epochs = 1

                train_loss, valid_loss = run_training(
                    model,
                    train_dataloader,
                    valid_dataloader,
                    optimizer,
                    num_epochs=num_epochs,
                    device=device,
                )

                row = {
                    "dataset_name": dataset_names[i],
                    "dataset_gzip": (med_gzip, std_gzip),
                    "token_ct": train_token_ct,
                    "model_stats": model_config_dict,
                    "model_size": model_size,
                    "num_epochs": num_epochs,
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                }
                results.append(row)

                with open(f"results_cuda:{cuda_idx}.jsonl", "a") as file:
                    file.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor, wait

    with ThreadPoolExecutor(max_workers=torch.cuda.device_count()) as executor:
        futures = [executor.submit(run_scaling_exps, i) for i in range(torch.cuda.device_count())]
        wait(futures)
    run_scaling_exps(4)  # NOTE: for running dataset 5
