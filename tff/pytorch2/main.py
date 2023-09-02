import torch
import torch.nn as nn
from model import Transformer
from dataset import get_dt

dim_model = 512
num_heads = 8
num_encoder_layers = 6
num_decoder_layers = 6
dropout = 0.1


def main():
    train_dataloader, _ = get_dt()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Transformer(701800,
                        dim_model,
                        num_heads,
                        num_encoder_layers,
                        num_decoder_layers,
                        dropout).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    def train_loop(model, opt, loss_fn, dataloader):
        model.train()
        total_loss = 0

        for batch in dataloader:
            X, y = batch[:, 0], batch[:, 1]
            X, y = torch.tensor(X).to(device), torch.tensor(y).to(device)

            # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
            y_input = y[:, :-1]
            y_expected = y[:, 1:]

            # Get mask to mask out the next words
            sequence_length = y_input.size(1)
            tgt_mask = model.get_tgt_mask(sequence_length).to(device)

            # Standard training except we pass in y_input and tgt_mask
            pred = model(X, y_input, tgt_mask)

            # Permute pred to have batch size first again
            pred = pred.permute(1, 2, 0)
            loss = loss_fn(pred, y_expected)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.detach().item()

        return total_loss / len(dataloader)

    def fit(model, opt, loss_fn, train_dataloader, epochs):  # val_dataloader, epochs):

        # Used for plotting later on
        train_loss_list, validation_loss_list = [], []

        print("Training and validating model")
        for epoch in range(epochs):
            print("-" * 25, f"Epoch {epoch + 1}", "-" * 25)

            train_loss = train_loop(model, opt, loss_fn, train_dataloader)
            train_loss_list += [train_loss]

            # validation_loss = validation_loop(model, loss_fn, val_dataloader)
            # validation_loss_list += [validation_loss]

            print(f"Training loss: {train_loss:.4f}")
            # print(f"Validation loss: {validation_loss:.4f}")
            print()

        return train_loss_list, validation_loss_list

    train_loss_list, validation_loss_list = fit(model, opt, loss_fn, train_dataloader, 10)  # val_dataloader, 10)
    print(train_loss_list, validation_loss_list)


if __name__ == '__main__':
    main()
