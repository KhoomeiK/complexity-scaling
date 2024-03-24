import torch
from training import run_training
from transformers import AdamW, AutoTokenizer, LlamaForCausalLM, LlamaConfig
import json
from data_gen import generate_dataset
from data_utils import (
    calculate_median_stdev_gzipability,
    count_total_tokens,
    pcfg_dataset_to_dataloader,
)

context_length = 256
dataset_stats = [
    (5, 50, 3, 2, False),
    (10, 150, 5, 3, False),
    (20, 300, 10, 5, False),
    (50, 600, 30, 15, False),
    (100, 2000, 100, 30, False),
]
pcfg_datasets = [
    generate_dataset(*row, 10_000_000, num_toks_per_seq=context_length)
    for row in dataset_stats
]
med_std_gzips = [
    calculate_median_stdev_gzipability(pcfg_dataset) for pcfg_dataset in pcfg_datasets
]
# for i, pcfg_dataset in enumerate(pcfg_datasets):
#     med, std = calculate_median_stdev_gzipability(pcfg_dataset)
#     total_toks = count_total_tokens(pcfg_dataset_to_dataloader(pcfg_dataset))

#     print(
#         f"{i}: {med:.3f} +- {std:.3f} ({total_toks})  | [{' '.join([str(x) for x in dataset_stats[i]])}]"
#     )


model_sizes = {
    "hidden_size": [64, 128, 256, 512, 1024],
    "intermediate_size": [128, 256, 512, 1024, 2048],
    "num_hidden_layers": [2, 4, 6, 10, 20],
    "num_attention_heads": [1, 2, 4, 8, 16],
}
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

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
results = []

for data_portion in (0.01, 0.1, 0.2, 0.5, 0.95):
    for i, pcfg_dataset in enumerate(pcfg_datasets):
        med_gzip, std_gzip = med_std_gzips[i]

        train_data_size = int(len(pcfg_dataset) * data_portion)
        valid_data_size = min(100, int(train_data_size / 10))
        train_dataloader = pcfg_dataset_to_dataloader(
            pcfg_dataset[:train_data_size], padder_tokenizer=tokenizer
        )
        valid_dataloader = pcfg_dataset_to_dataloader(
            pcfg_dataset[-valid_data_size:], padder_tokenizer=tokenizer
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

            print(
                f"Dataset Stats: {med_gzip:.3f} +- {std_gzip:.3f} | {dataset_stats[i]}"
            )
            print(f"Model Size: {model_size/1_000_000:.1f}M")
            print(f"Train Token Count: {train_token_ct}")

            model.to(device)
            optimizer = AdamW(model.parameters(), lr=5e-5)
            num_epochs = 10

            train_perplexities, valid_perplexities = run_training(
                model, train_dataloader, valid_dataloader, optimizer, num_epochs=num_epochs
            )

            row = {
                "dataset_stats": dataset_stats[i],
                "dataset_gzip": (med_gzip, std_gzip),
                "token_ct": train_token_ct,
                "model_stats": model_config_dict,
                "model_size": model_size,
                "num_epochs": num_epochs,
                "train_pplx": train_perplexities,
                "valid_pplx": valid_perplexities,
            }
            results.append(row)

            with open("results.jsonl", "a") as file:
                file.write(json.dumps(row) + "\n")
