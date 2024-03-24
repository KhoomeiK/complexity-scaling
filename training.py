import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR


def compute_perplexity(dataloader, model, device="cuda"):
    # adapted from: https://github.com/huggingface/evaluate/blob/main/metrics/perplexity/perplexity.py
    model = model.to(device)

    ppls = []
    loss_fct = CrossEntropyLoss(reduction="none")

    for batch in dataloader:
        batch.to(device)
        encoded_batch = batch["input_ids"]
        attn_mask = batch["attention_mask"]

        labels = encoded_batch

        with torch.no_grad():
            out_logits = model(encoded_batch, attention_mask=attn_mask).logits

        shift_logits = out_logits[
            ..., :-1, :
        ].contiguous()  # TODO: double check that all this logic is correct
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        perplexity_batch = torch.exp(
            (
                loss_fct(shift_logits.transpose(1, 2), shift_labels)
                * shift_attention_mask_batch
            ).sum(1)
            / shift_attention_mask_batch.sum(1)
        )

        ppls += perplexity_batch.tolist()

    return np.mean(ppls)


def run_training(
    model, train_dataloader, valid_dataloader, optimizer, num_epochs=10, device="cuda"
):
    train_loss = []
    valid_loss = []

    lr_scheduler = CosineAnnealingLR(optimizer, T_max=len(train_dataloader))

    for epoch in range(num_epochs):
        progress_bar = tqdm(
            range(len(train_dataloader)), desc=f"Epoch {epoch + 1}/{num_epochs}"
        )

        model.train()
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            train_loss.append(loss.item())

            optimizer.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        lr_scheduler.step()

        model.eval()
        with torch.no_grad():
            for batch in valid_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                valid_loss.append(loss.item())

        print(
            f"Train Loss: {np.median(train_loss):.3f}, Valid Loss: {np.median(valid_loss):.3f}"
        )

    return train_loss, valid_loss
