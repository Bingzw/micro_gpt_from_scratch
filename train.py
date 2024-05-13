import tiktoken
import json
import torch
from tokenizer import get_gpt2_tokenizer
from dataloader import create_dataloader_v1
from gpt import GPTModel
from utils import calc_loss_batch, calc_loss_loader


def train_gpt_model(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter):
    """
    Train the GPT model
    :param model: the GPT model
    :param train_loader: the training data loader
    :param val_loader: the validation data loader
    :param optimizer: the optimizer
    :param device: the device
    :param num_epochs: the number of epochs
    :param eval_freq: the evaluation frequency
    :param eval_iter: the number of batches to evaluate
    :return: the training losses, validation losses, and the tokens seen
    """
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # reset the loss gradient to zero from previous epoch
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1
            # evaluate the model
            if global_step % eval_freq == 0:
                model.eval()
                with torch.no_grad():
                    train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
                    val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
                print(f"Epoch {epoch+1}, Global Step {global_step:06d}, Train Loss: {train_loss:.3f}, "
                      f"Val Loss: {val_loss:.3f}")
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                model.train()
    return train_losses, val_losses, track_tokens_seen


if __name__ == "__main__":
    SEED = 123
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    tokenizer = get_gpt2_tokenizer()
    total_tokens = len(tokenizer.encode(raw_text))
    print("Total tokens are: ", total_tokens)

    with open('gpt_config.json') as f:
        config = json.load(f)

    # Create the dataloader
    train_ratio = 0.9
    assert total_tokens * train_ratio > config["sequence_length"], "The training data is too small, dataloader maybe " \
                                                                   "empty"
    assert total_tokens * (1 - train_ratio) > config["sequence_length"], "The validation data is too small, " \
                                                                         "dataloader maybe empty"

    split_idx = int(len(raw_text) * train_ratio)
    train_data = raw_text[:split_idx]
    val_data = raw_text[split_idx:]

    torch.manual_seed(SEED)
    batch_size = 2
    train_dataloader = create_dataloader_v1(
        train_data,
        batch_size=batch_size,
        max_length=config["sequence_length"],
        stride=config["sequence_length"],
        shuffle=True,
        drop_last=True,
        num_workers=0
    )

    val_dataloader = create_dataloader_v1(
        val_data,
        batch_size=batch_size,
        max_length=config["sequence_length"],
        stride=config["sequence_length"],
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    model = GPTModel(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    learning_rate = 0.0004
    weight_decay = 0.1
    num_epochs = 10
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    '''
    train_losses, val_losses, track_tokens_seen = train_gpt_model(
        model=model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        eval_freq=5,
        eval_iter=5
    )
    '''






