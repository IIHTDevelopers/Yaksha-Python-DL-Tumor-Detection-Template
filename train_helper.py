import torch
from loss import DiceBCELoss
from train import TrainingModule
import pandas as pd


class TrainHelper:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.__initialize_hyperparameters()
        self.training_module = TrainingModule(model, self.optimizer, self.loss_fn, self.device)
        self.df = pd.DataFrame(columns = ["Epoch", "Train Loss", "Jaccard Cofficient", "Dice Score", "Val Loss", "Jaccard Cofficient (Val)", "Dice Score (Val)"])


    def __initialize_hyperparameters(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5, verbose=True)
        self.loss_fn = DiceBCELoss()
    

    def run(self, args, train_dataloader, val_dataloader, dest_dir):
        val_loss_min = 10000.0
        for epoch in range(1, args.epochs+1):
            print("Epoch {}/{}".format(epoch, args.epochs))
            # training
            train_loss, train_jc, train_dice = self.training_module.train(train_dataloader)
            print("Loss: {}\tJC: {}\t\tDice Score: {}".format(train_loss, train_jc, train_dice))
            # evaluation
            val_loss, val_jc, val_dice = self.training_module.evaluation(val_dataloader)
            print("Val Loss: {}\tVal JC: {}\tVal Dice Score: {}".format(val_loss, val_jc, val_dice))
            # schedular
            self.scheduler.step(val_loss)
            if val_loss < val_loss_min:
                val_loss_min = val_loss
                self.training_module.save(dest_dir + "/model.pt".format(int(epoch)))
                with open(dest_dir + "/model_min_val_loss.txt", "a") as f:
                    f.write("Epoch {} -> {}\n".format(int(epoch), val_loss))
            # data entry and saving
            self.df.loc[len(self.df.index)] = [int(epoch), train_loss, train_jc, train_dice, val_loss, val_jc, val_dice]
            self.df.to_csv(dest_dir + "/loss.csv", index = False)

        # self.training_module.save(dest_dir + "/model.pt")