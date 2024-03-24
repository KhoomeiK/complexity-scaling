from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from datasets import Dataset
import gzip
import io
from typing import List, Union
from statistics import median, stdev


def count_total_tokens(dataloader):
    total_tokens = 0
    for batch in dataloader:
        total_tokens += sum(batch["attention_mask"].flatten().tolist())
    return total_tokens


def pad_and_mask(sequence, sequence_length):
    if sequence_length - len(sequence) == 0:
        padded_sequence = sequence
    elif sequence_length - len(sequence) > 0:
        padded_sequence = sequence + [32000] * (sequence_length - len(sequence))
    elif sequence_length - len(sequence) < 0:
        padded_sequence = sequence[:sequence_length]
    mask = [1 if token != 32000 else 0 for token in padded_sequence]
    return padded_sequence, mask


def pcfg_dataset_to_dataloader(pcfg_dataset, padder_tokenizer, batch_size=8, context_length=256):
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

    dataloader = DataLoader(
        tokenized_dataset, shuffle=True, batch_size=batch_size, collate_fn=data_collator
    )

    return dataloader


def calculate_gzipability(
    input_data: Union[str, List[int]], gzip_toks: bool = True
) -> int:
    if type(input_data) == str and not gzip_toks:
        input_bytes = input_data.encode("utf-8")
    else:  # token list
        if type(input_data) == str:
            input_data = [int(tok) for tok in input_data.split(" ")]
        input_bytes = b"".join(
            int.to_bytes(i, length=4, byteorder="big", signed=True) for i in input_data
        )

    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as f:
        f.write(input_bytes)

    compressed_size = buf.tell()
    gzipability = compressed_size / len(input_bytes)

    return gzipability


def calculate_median_stdev_gzipability(pcfg_dataset):
    gzipability_scores = [
        calculate_gzipability([int(tok) for tok in row.split(" ")])
        for row in pcfg_dataset
    ]
    med = median(gzipability_scores)

    if len(gzipability_scores) > 1:
        std_dev = stdev(gzipability_scores)
    else:
        std_dev = 0  # Default to 0 if there's only one element to avoid division by zero in stdev calculation

    return med, std_dev

