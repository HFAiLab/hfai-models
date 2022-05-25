import os
import torch
import hfai

from bert import bert_base_CLS


class Trainer:
    def __init__(self, model, lr, batch_size, epochs):
        super().__init__()
        self.lr = lr
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.criterion = torch.nn.CrossEntropyLoss()

    def finetune(self, train_dataset, eval_dataset):
        self.optimizer = torch.optim.AdamW([
            {"params": [p for n, p in self.model.named_parameters() if "cls" not in n], "lr": self.lr / 10},
            {"params": [p for n, p in self.model.named_parameters() if "cls" in n], "lr": self.lr},
        ])
        train_dataloader = train_dataset.loader(self.batch_size, shuffle=True)
        eval_dataloader = eval_dataset.loader(self.batch_size, shuffle=False)
        for epoch in range(self.epochs):
            self.model.train()
            for step, batch in enumerate(train_dataloader):
                seq, mask, label = [data.cuda() for data in batch]
                loss = self.criterion(self.model(seq, mask.bool()), label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if step % 10 == 0:
                    print(f"Epoch: {epoch + 1}/{self.epochs}, Step: {step + 1}/{len(train_dataloader)}, Loss: {loss.item()}", flush=True)

            self.model.eval()
            correct, loss, total = [0] * 3
            with torch.no_grad():
                for step, batch in enumerate(eval_dataloader):
                    seq, mask, label = [data.cuda() for data in batch]
                    output = self.model(seq, mask)
                    predict = output.argmax(-1)
                    loss += self.criterion(output, label)
                    correct += torch.eq(predict, label).sum().item()
                    total += seq.size(0)
                print(f"Epoch: {epoch + 1}/{self.epochs}, Eval Loss: {loss.item() / len(eval_dataloader)}, Eval Acc: {100 * correct / total:.2f}%")


def finetune():
    seed = 12
    vocab_size = 8021
    max_length = 128
    batch_size = 200
    epochs = 5
    lr = 0.0005
    save_path = "output/bert"
    dataset_name = "iflytek"

    torch.manual_seed(seed)
    train_dataset = hfai.datasets.CLUEForCLS(dataset_name=dataset_name, split="train")
    eval_dataset = hfai.datasets.CLUEForCLS(dataset_name=dataset_name, split="dev")

    classes = len(train_dataset.classes)
    model = bert_base_CLS(vocab_size=vocab_size, max_length=max_length, classes=classes).cuda()

    state = torch.load(os.path.join(save_path, "model.pt"), map_location="cpu")
    model.load_state_dict(state['model'], strict=False)

    trainer = Trainer(model, lr, batch_size, epochs)
    trainer.finetune(train_dataset, eval_dataset)


if __name__ == "__main__":
    finetune()
