import torch, os, cv2, shutil
from torch import nn
from dataloader import PolypDatasetLoader
from loss import DiceBCELoss
from metric import dice_cofficient
from model import PolypModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import jaccard_score as jsc
from test_functional import server_tests


class Test(nn.Module):
    def __init__(self, device):
       super(Test, self).__init__()
       self.device = device
       self.loss_fn = DiceBCELoss()
    
    def load_model(self, checkpoint_path):
        ## checking for model path
        if not os.path.exists(checkpoint_path):
            print("Inccorect model path")
            exit()
        self.model = PolypModel().to(self.device)
        self.model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])
    
    def load_dataset(self, root_dir, image_dir, mask_dir):
        val_dataset = PolypDatasetLoader(os.path.join(root_dir, image_dir) + "/", os.path.join(root_dir, mask_dir) + "/")
        self.val_dataloader = DataLoader(val_dataset, shuffle=True, drop_last=False, batch_size=2)
    

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
    

    def __denormalize_image(self, image, mean=[0,0,0], std=[1,1,1]):
        for channel in range(3):
            image[:,:,channel] = (image[:,:,channel]*std[channel] + mean[channel]) * 255.0
        return image
    

    def __save_image_func(self, image_tensor, save_dir, name, counter):
        try:
            image_tensor = torch.permute(image_tensor, (1,2,0))
        except:
            print(image_tensor.shape)
            raise Exception("Exception in permute")
        if self.device == "gpu":
            image_tensor = image_tensor.detach().cpu().numpy()
        else:
            image_tensor = image_tensor.cpu().numpy()
        
        if name == "image":
            image_tensor = self.__denormalize_image(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            image_tensor = cv2.cvtColor(image_tensor, cv2.COLOR_RGB2BGR)
        else:
            image_tensor[image_tensor>=0.5] = 1
            image_tensor[image_tensor<0.5] = 0
            image_tensor = image_tensor*255
        cv2.imwrite(save_dir + "{}_{}.png".format(counter, name), image_tensor)
    


    def __save_img(self, image_tensor, mask_tensor, pred_tensor, save_dir):   
        for i in range(image_tensor.size(0)):
            self.__save_image_func(image_tensor[i,:,:,:], save_dir, "image", self.counter)
            self.__save_image_func(mask_tensor[i,:,:,:], save_dir, "mask", self.counter)
            self.__save_image_func(pred_tensor[i,:,:,:], save_dir, "pred", self.counter)
            self.counter += 1
    
    
    def test(self, save_dir):
        epoch_loss = 0.0
        epoch_jc = 0.0
        epoch_dice = 0.0
        self.counter = 0

        self.model.eval()
        with torch.no_grad():
            for image, mask in tqdm(self.val_dataloader):
                image = image.to(self.device)
                mask = mask.to(self.device)

                # running the model
                pred = self.model(image)

                ## saving samples
                self.__save_img(image, mask, pred, save_dir)

                loss = self.loss_fn(pred, mask)
                epoch_loss += loss.item()

                ## metrics
                dice_score, jc = self.__metric_calculation(pred, mask)
                epoch_jc += jc
                epoch_dice += dice_score
        return (epoch_loss/len(self.val_dataloader), epoch_jc/len(self.val_dataloader), epoch_dice/len(self.val_dataloader), mask.detach().cpu().numpy(), pred.detach().cpu().numpy())


def test_helper(root_dir, image_dir, mask_dir, model_path, dest_folder):
    print("Making directory")
    save_dir = dest_folder + "/"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test = Test(device)
    print("Loading model")
    test.load_model(model_path)
    print("Loading dataset")
    test.load_dataset(root_dir, image_dir, mask_dir)
    print("Testing model on images")
    loss, jc, dice, mask, pred = test.test(save_dir)
    print("Loss: {}\tJC: {}\t\tDice Cofficient: {}".format(loss, jc, dice))
    print("Saving metric values")
    with open(save_dir + "loss_metrics.txt", "w") as file:
        file.write("Loss: {}\tJC: {}\t\tDice Cofficient: {}".format(loss, jc, dice))
    if jc > 0.8 and dice > 0.8:
        print("model is good")
    else:
        print("model is not good")
    
    print("Uploading Test results to server")
    server_tests(model_path, mask, pred, jc, dice)




if __name__ == "__main__":
    test_helper(root_dir="test-images/", image_dir="images", mask_dir="masks", model_path="results/model.pt", dest_folder = "test_results/")
