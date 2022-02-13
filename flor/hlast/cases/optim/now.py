import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torchtext.legacy.data import Field, TabularDataset, BucketIterator
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import flor
from multiprocessing import set_start_method
from utils import CLR_Scheduler

try:
    set_start_method("spawn")
except RuntimeError:
    pass

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

label_field = Field(
    sequential=False, use_vocab=False, batch_first=True, dtype=torch.float
)
text_field = Field(tokenize="spacy", lower=True, include_lengths=True, batch_first=True)
fields = [("words", text_field), ("target", label_field)]
fields_test = [("words", text_field)]

train, valid = TabularDataset.splits(
    path="data",
    train="train_rnn.csv",
    validation="valid_rnn.csv",
    format="CSV",
    fields=fields,
    skip_header=True,
)
test = TabularDataset(
    path="data/test_rnn.csv", format="CSV", fields=fields_test, skip_header=True
)

train_iter = BucketIterator(
    train,
    batch_size=200,
    sort_key=lambda x: len(x.words),
    device=device,
    sort=True,
    sort_within_batch=True,
)
valid_iter = BucketIterator(
    valid,
    batch_size=200,
    sort_key=lambda x: len(x.words),
    device=device,
    sort=True,
    sort_within_batch=True,
)
test_iter = BucketIterator(
    test,
    batch_size=200,
    sort_key=lambda x: len(x.words),
    device=device,
    sort=True,
    sort_within_batch=True,
)

text_field.build_vocab(train, min_freq=5)

# LSTM model
class LSTM(nn.Module):
    def __init__(self, dimension=128):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(len(text_field.vocab), dimension)
        self.lstm = nn.LSTM(
            input_size=dimension,
            hidden_size=dimension,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.drop = nn.Dropout(p=0.85)
        self.dimension = dimension
        self.fc = nn.Linear(2 * dimension, 1)
        self.relu = nn.ReLU()

    def forward(self, text, text_len):

        text_emb = self.relu(self.embedding(text))

        packed_input = pack_padded_sequence(
            text_emb, text_len, batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), text_len - 1, : self.dimension]
        out_reverse = output[:, 0, self.dimension :]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_fea = out_reduced

        text_fea = self.fc(self.drop(text_fea))
        text_fea = torch.squeeze(text_fea, 1)
        text_out = torch.sigmoid(text_fea)

        return text_out


# training
def train(
    model,
    optimizer,
    criterion=nn.BCELoss(),
    train_loader=train_iter,
    valid_loader=valid_iter,
    test_loader=test_iter,
    num_epochs=5,
    eval_every=len(train_iter) // 2,
    file_path="training_process",
    best_valid_loss=float("Inf"),
):
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # training loop
    best_loss = float("inf")
    model.train()
    for epoch in flor.it(range(num_epochs)):
        flor.log("learning_rate", str(optimizer.param_groups[0]["lr"]))
        if flor.SkipBlock.step_into("batchwise-loop"):
            for ((words, words_len), labels), _ in train_loader:
                labels = labels.to(device)
                words = words.to(device)
                words_len = words_len.detach().cpu()
                output = model(words, words_len)

                loss = criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # update running values
                running_loss += loss.item()
                global_step += 1

                # evaluation step
                if global_step % eval_every == 0:
                    model.eval()
                    with torch.no_grad():
                        # validation loop
                        for ((words, words_len), labels), _ in valid_loader:
                            labels = labels.to(device)
                            words = words.to(device)
                            words_len = words_len.detach().cpu()
                            output = model(words, words_len)

                            loss = criterion(output, labels)
                            valid_running_loss += float(loss.item())

                    # evaluation
                    average_train_loss = running_loss / eval_every
                    average_valid_loss = valid_running_loss / len(valid_loader)

                    if average_valid_loss < best_loss:
                        best_loss = average_valid_loss
                        torch.save(model.state_dict(), "best-model.pt")

                    train_loss_list.append(average_train_loss)
                    valid_loss_list.append(average_valid_loss)
                    global_steps_list.append(global_step)

                    # resetting running values
                    running_loss = 0.0
                    valid_running_loss = 0.0
                    model.train()

                    # print progress
                    print(
                        "Epoch [{}/{}], LR: {:.3f}, Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}".format(
                            epoch + 1,
                            num_epochs,
                            optimizer.param_groups[0]["lr"],
                            global_step,
                            num_epochs * len(train_loader),
                            average_train_loss,
                            average_valid_loss,
                        )
                    )
                    flor.log("avg_train_loss", average_train_loss)
                    flor.log("average_valid_loss", average_valid_loss)
                clr_scheduler.step()

        flor.SkipBlock.end(model, optimizer, clr_scheduler)

    y_pred = []
    model.eval()
    with torch.no_grad():
        for ((words, words_len)), _ in test_loader:
            words = words.to(device)
            words_len = words_len.detach().cpu()
            output = model(words, words_len)

            output = (output > 0.5).int()
            y_pred.extend(output.tolist())
    print("Finished Training!")
    return y_pred


EPOCHS = 80
MIN_LR = 1e-4

model = LSTM(8).to(device)
optimizer = optim.SGD(model.parameters(), lr=MIN_LR)
flor.log("optimizer", str(type(optimizer)))
clr_scheduler = CLR_Scheduler(
    optimizer,
    net_steps=(len(train_iter) * EPOCHS),
    min_lr=MIN_LR,
    max_lr=4.0,
    tail_frac=0.0,
)
pred = train(model=model, optimizer=optimizer, num_epochs=EPOCHS)

# save result as .csv file
# test_data = pd.read_csv("data/test.csv")
# preds_df = pd.DataFrame({"id": test_data["id"], "target": pred})
# preds_df.to_csv(f"data/output_lstm_3.csv", index=False)
