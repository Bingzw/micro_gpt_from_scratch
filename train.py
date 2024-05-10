import tiktoken
import torch
from tokenizer import get_gpt2_tokenizer
from dataloader import create_dataloader_v1

if __name__ == "__main__":
    with open("the-verdict.txt", "r") as f:
        raw_text = f.read()

    tokenizer = get_gpt2_tokenizer()
    encoded_text = tokenizer.encode(raw_text)

    vocab_size = 50257
    output_dim = 256
    context_length = 1024

    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

    max_length = 4
    batch_size = 8
    dataloader = create_dataloader_v1(raw_text, batch_size=batch_size, max_length=max_length, stride=max_length)

    for batch in dataloader:
        x, y = batch

        token_embeddings = token_embedding_layer(x)
        pos_embeddings = pos_embedding_layer(torch.arange(max_length))

        input_embeddings = token_embeddings + pos_embeddings

        break

    print(input_embeddings.shape)


