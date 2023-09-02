import numpy as np


def get_dt():
    def batchify_data(data, batch_size=16, padding=False, padding_token=-1):
        batches = []
        for idx in range(0, len(data), batch_size):
            # We make sure we dont get the last bit if its not batch_size size
            if idx + batch_size < len(data):
                # Here you would need to get the max length of the batch,
                # and normalize the length with the PAD token.
                if padding:
                    max_batch_length = 0

                    # Get longest sentence in batch
                    for seq in data[idx : idx + batch_size]:
                        if len(seq) > max_batch_length:
                            max_batch_length = len(seq)

                    # Append X padding tokens until it reaches the max length
                    for seq_idx in range(batch_size):
                        remaining_length = max_batch_length - len(data[idx + seq_idx])
                        data[idx + seq_idx] += [padding_token] * remaining_length

                batches.append(np.array(data[idx : idx + batch_size]))

        print(f"{len(batches)} batches of size {batch_size}")

        return batches

    content_train_x = open(r"../English-german/train-x", "r")

    content_train_y = open(r"../English-german/train-y", "r")

    content_train_x = content_train_x.read().split("\n")
    content_train_y = content_train_y.read().split("\n")

    from torchtext.data import get_tokenizer

    de_tokenizer = get_tokenizer('spacy', language='de')
    en_tokenizer = get_tokenizer('spacy', language='en')

    train_data = [[en_tokenizer(x), de_tokenizer(y)] for x, y in zip(content_train_x, content_train_y)]

    train_dataloader = batchify_data(train_data)
    val_dataloader = 'batchify_data(val_data)'

    return train_dataloader, val_dataloader
