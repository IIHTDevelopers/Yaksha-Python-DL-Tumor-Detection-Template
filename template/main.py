from model import PolypModel
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
    model = PolypModel().to(device)
    
    # sample count
    data_sample_count = len(train_dataloader)
    print("Data Samples:", data_sample_count)

    # train helper for training the model
    train_helper = TrainHelper(model, device)
    train_helper.run(args, train_dataloader, val_dataloader, dest_dir)


main()