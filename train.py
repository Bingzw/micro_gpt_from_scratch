import tiktoken
import json
import torch
import math
from dataloader import create_dataloader_v1
from gpt import GPTModel
from utils import calc_loss_batch, calc_loss_loader, generate_and_print_sample, generate_advanced, text_to_token_ids, \
    token_ids_to_text


def train_model(model, train_loader, val_loader, optimizer, device, n_epochs,
                eval_freq, eval_iter, start_context, tokenizer, warmup_steps,
                initial_lr=3e-05, min_lr=1e-6):
    train_losses, val_losses, track_tokens_seen, track_lrs = [], [], [], []
    tokens_seen, global_step = 0, -1

    # Retrieve the maximum learning rate from the optimizer
    peak_lr = optimizer.param_groups[0]["lr"]

    # Calculate the total number of iterations in the training process
    total_training_steps = len(train_loader) * n_epochs

    # Calculate the learning rate increment during the warmup phase
    lr_increment = (peak_lr - initial_lr) / warmup_steps

    for epoch in range(n_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            global_step += 1

            # Adjust the learning rate based on the current phase (warmup or cosine annealing)
            if global_step < warmup_steps:
                # Linear warmup
                lr = initial_lr + global_step * lr_increment
            else:
                # Cosine annealing after warmup
                progress = ((global_step - warmup_steps) /
                            (total_training_steps - warmup_steps))
                lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            # Apply the calculated learning rate to the optimizer
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            track_lrs.append(lr)  # Store the current learning rate

            # Calculate and backpropagate the loss
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()

            # Apply gradient clipping after the warmup phase to avoid exploding gradients
            if global_step > warmup_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            tokens_seen += input_batch.numel()

            # Periodically evaluate the model on the training and validation sets
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader,
                    device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                # Print the current losses
                print(f"Ep {epoch + 1} (Iter {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Generate and print a sample from the model to monitor progress
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen, track_lrs


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


if __name__ == "__main__":
    SEED = 123
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        text_data = f.read()

    with open('gpt_config.json') as f:
        config = json.load(f)

    tokenizer = tiktoken.get_encoding("gpt2")
    total_tokens = len(tokenizer.encode(text_data))
    print("Total tokens are: ", total_tokens)

    # Create the dataloader
    train_ratio = 0.9
    assert total_tokens * train_ratio > config["context_length"], "The training data is too small, dataloader maybe " \
                                                                   "empty"
    assert total_tokens * (1 - train_ratio) > config["context_length"], "The validation data is too small, " \
                                                                         "dataloader maybe empty"

    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    torch.manual_seed(SEED)
    batch_size = 2
    train_dataloader = create_dataloader_v1(
        train_data,
        batch_size=batch_size,
        max_length=config["context_length"],
        stride=config["context_length"],
        shuffle=True,
        drop_last=True,
        num_workers=0
    )

    val_dataloader = create_dataloader_v1(
        val_data,
        batch_size=batch_size,
        max_length=config["context_length"],
        stride=config["context_length"],
        shuffle=False,
        drop_last=False,
        num_workers=0
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    model = GPTModel(config)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    n_epochs = 15
    peak_lr = 5e-4
    total_steps = len(train_dataloader) * n_epochs * train_dataloader.batch_size
    warmup_steps = int(0.1 * total_steps)
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.1)
    tokenizer = tiktoken.get_encoding("gpt2")

    n_epochs = 15
    train_losses, val_losses, tokens_seen, lrs = train_model(
        model, train_dataloader, val_dataloader, optimizer, device, n_epochs=n_epochs,
        eval_freq=5, eval_iter=1, start_context="Every effort moves you",
        tokenizer=tokenizer, warmup_steps=warmup_steps,
        initial_lr=1e-5, min_lr=1e-5
    )

    # generate a sample text after training
    token_ids = generate_advanced(
        model=model,
        idx=text_to_token_ids("Every effort moves you", tokenizer),
        max_new_tokens=25,
        max_seq_len=config["context_length"],
        top_k=50,
        temperature=1.5
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

    # save model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        "model_and_optimizer.pth"
    )
    # load model
    """
    checkpoint = torch.load("model_and_optimizer.pth")
    model = GPTModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.1)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.train()
    """





