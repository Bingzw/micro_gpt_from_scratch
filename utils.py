import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(embed_dim))  # the learnable scale parameter
        self.shift = nn.Parameter(torch.zeros(embed_dim))  # the learnable shift parameter
        # note that the scale and shift parameters are initialized to 1 and 0 respectively, and will be trained during
        # training process

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.scale * (x - mean) / (std + self.eps) + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
        ))


def generate_text_simple(model, idx, max_new_tokens, max_sequence_length):
    """
    Generate text from the model
    :param model: the gpt model
    :param idx: input indices of the shape (batch_size, sequence_length)
    :param max_new_tokens: the maximum number of tokens to generate
    :param max_sequence_length: the max size of the sequence that gpt model supports
    :return: the generated text
    """
    for _ in range(max_new_tokens):
        # crop the input to the content size
        idx_cond = idx[:, -max_sequence_length:]

        with torch.no_grad():
            logits = model(idx_cond)

        # get the last token logits
        logits_last = logits[:, -1, :]  # (batch_size, vocab_size)
        # get the idx of the vocabulary with the highest probability
        idx_new = torch.argmax(logits_last, dim=-1, keepdim=True)  # (batch_size, 1)
        # concatenate the new idx to the input
        idx = torch.cat([idx, idx_new], dim=-1)

    return idx


def text_to_token_ids(text, tokenizer):
    """
    Convert text to token ids
    :param text: the input text
    :param tokenizer: the tokenizer
    :return: encoded tensor
    """
    encoded = tokenizer.encode(text, allowed_special={'<endoftext>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """
    Convert token ids to text
    :param token_ids: the input token ids
    :param tokenizer: the tokenizer
    :return: the decoded text
    """
    token_ids_flatten = token_ids.squeeze_(0)  # remove batch dimension
    decoded = tokenizer.decode(token_ids_flatten.tolist())
    return decoded


def calc_loss_batch(input_batch, target_batch, model, device):
    """
    Calculate the loss for a batch
    :param input_batch: the input batch
    :param target_batch: the target batch
    :param model: the gpt model
    :param device: the device
    :return: the loss
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, batch_size=None):
    """
    Calculate the loss for a dataloader
    :param data_loader: the dataloader
    :param model: the gpt model
    :param device: the device
    :param batch_size: the number of batches to calculate the loss
    :return: the loss
    """
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif batch_size is None:
        batch_size = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        batch_size = min(batch_size, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < batch_size:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / batch_size


