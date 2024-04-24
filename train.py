from tqdm.auto import tqdm
import torch
from torch import nn
from sklearn.metrics import jaccard_score as jsc
from metric import dice_cofficient


class TrainingModule(nn.Module):
    def __init__(self, model, optimizer, loss_fn, device):
        super(TrainingModule, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
    

    def __metric_calculation(self, pred, mask):
        pred = pred.detach().cpu()
        mask = mask.detach().cpu()
        ## getting dice cofficient
        dice_score = dice_cofficient(pred, mask)
        ## getting jaccard cofficient
        mask = mask.long()
        mask = mask.numpy().reshape(-1)
        pred[pred>=0.5] = 1.0
        pred[pred<0.5] = 0.0
        pred = pred.long()
        pred = pred.numpy().reshape(-1)
        jc = jsc(pred, mask)
        return (dice_score.item(), jc)
    


    def train(self, dataloader):
        epoch_loss = 0.0
        epoch_jc = 0.0
        epoch_dice = 0.0

        self.model.train()
        for image, mask in tqdm(dataloader):
            image = image.to(self.device)
            mask = mask.to(self.device)

            # running the model
            self.optimizer.zero_grad()
            pred = self.model(image)
            loss = self.loss_fn(pred, mask)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

            ## metrics
            dice_score, jc = self.__metric_calculation(pred, mask)
            epoch_jc += jc
            epoch_dice += dice_score
        return (epoch_loss/len(dataloader), epoch_jc/len(dataloader), epoch_dice/len(dataloader))



    def evaluation(self, dataloader):
        epoch_loss = 0.0
        epoch_jc = 0.0
        epoch_dice = 0.0

        self.model.eval()
        with torch.no_grad():
            for image, mask in tqdm(dataloader):
                image = image.to(self.device)
                mask = mask.to(self.device)

                # running the model
                pred = self.model(image)
                loss = self.loss_fn(pred, mask)
                epoch_loss += loss.item()

                ## metrics
                dice_score, jc = self.__metric_calculation(pred, mask)
                epoch_jc += jc
                epoch_dice += dice_score
        return (epoch_loss/len(dataloader), epoch_jc/len(dataloader), epoch_dice/len(dataloader))
    


    def save(self, save_path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_architecture': self.model,
            }, save_path)