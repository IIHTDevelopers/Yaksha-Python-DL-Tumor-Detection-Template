# import argparse, os, torch, splitfolders
# from datetime import datetime
# from dataloader import PolypDatasetLoader
# from torch.utils.data import DataLoader
# from loss import DiceBCELoss
# from model import PolypModel
# from train import TrainingModule
# import pandas as pd
# from pathlib import Path

# parser = argparse.ArgumentParser()
# parser.add_argument('--data-dir', '-d', required = True, type = str, help = 'The directory where data is stored')
# parser.add_argument('--image-dir', '-i', required = True, type = str, help = 'Image directory name')
# parser.add_argument('--mask-dir', '-m', required = True, type = str, help = 'mask directory name')
# parser.add_argument('--batch-size', '-b', default = 1, type = int, help = 'Batch size for experinment')
# parser.add_argument('--epochs', '-e', default = 300, type = int, help = 'Number of epoch to run experinment')
# parser.add_argument('--name', '-n', default = "test", type = str, help = 'Name of Experinment')

# args = parser.parse_args()

# ## folder creation
# dest_dir = "results/"
# os.makedirs(dest_dir, exist_ok=True)

# time = datetime.now().strftime("%H_%M_%S")
# dest_dir = dest_dir + time + "_" + args.name + "/"
# os.makedirs(dest_dir, exist_ok=True)
# ## --------------------

# ## splitting data
# loc = Path(__file__).resolve().parent.parent
# save_loc = os.path.join(loc, "split_data")
# print(save_loc)
# splitfolders.ratio(args.data_dir, output=save_loc, seed=7, ratio=(.85, .15), group_prefix=None, move=False)
# ## ----------------------------

# ## dataloader
# train_dataset = PolypDatasetLoader(save_loc + "/train/" + args.image_dir + "/", save_loc + "/train/" + args.mask_dir + "/")
# train_dataloader = DataLoader(train_dataset, shuffle=True, drop_last=False, batch_size=args.batch_size)

# val_dataset = PolypDatasetLoader(save_loc + "/val/" + args.image_dir + "/", save_loc + "/val/" + args.mask_dir + "/")
# val_dataloader = DataLoader(val_dataset, shuffle=True, drop_last=False, batch_size=args.batch_size)
# ## --------------------------

# ## device info
# device = "cuda" if torch.cuda.is_available() else "cpu"
# ## ------------

# ## model
# model = PolypModel().to(device)
# ## ----------------
# data_sample_count = len(train_dataloader)
# print("Data Samples:", data_sample_count)

# ## Hyperparameter
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
# loss_fn = DiceBCELoss()
# ## ----------------

# ## Dataframe
# df = pd.DataFrame(columns = ["Epoch", "Train Loss", "Jaccard Cofficient", "Dice Score", "Val Loss", "Jaccard Cofficient (Val)", "Dice Score (Val)"])
# ## ----------------

# training_module = TrainingModule(model, optimizer, loss_fn, device)
# val_loss_min = 10000.0

# for epoch in range(1, args.epochs+1):
#     print("Epoch {}/{}".format(epoch, args.epochs))
#     # training
#     train_loss, train_jc, train_dice = training_module.train(train_dataloader)
#     print("Loss: {}\tJC: {}\t\tDice Score: {}".format(train_loss, train_jc, train_dice))
#     # evaluation
#     val_loss, val_jc, val_dice = training_module.evaluation(val_dataloader)
#     print("Val Loss: {}\tVal JC: {}\tVal Dice Score: {}".format(val_loss, val_jc, val_dice))
#     # schedular
#     scheduler.step(val_loss)
#     if val_loss < val_loss_min:
#         val_loss_min = val_loss
#         training_module.save(dest_dir + "/model_min_{}.pt".format(int(epoch)))
#     # data entry and saving
#     df.loc[len(df.index)] = [int(epoch), train_loss, train_jc, train_dice, val_loss, val_jc, val_dice]
#     df.to_csv(dest_dir + "/loss.csv", index = False)

# training_module.save(dest_dir + "/model.pt")



from model import CompNet
import arguments, utils, torch
from train_helper import TrainHelper


def main():
    args = arguments.get_arguments()
    ## folder creation
    dest_dir = "results/"
    utils.folder_creation(dest_dir)
    ## splitting data
    save_loc = utils.splitting_data(args)
    ## dataloader
    train_dataloader, val_dataloader = utils.get_dataloader(save_loc, args)
    ## device info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ## model
    model = CompNet().to(device)
    
    # sample count
    data_sample_count = len(train_dataloader)
    print("Data Samples:", data_sample_count)

    # train helper for training the model
    train_helper = TrainHelper(model, device)
    train_helper.run(args, train_dataloader, val_dataloader, dest_dir)


main()
