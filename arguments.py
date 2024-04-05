import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', '-d', required = True, type = str, help = 'The directory where data is stored')
    parser.add_argument('--image-dir', '-i', required = True, type = str, help = 'Image directory name')
    parser.add_argument('--mask-dir', '-m', required = True, type = str, help = 'mask directory name')
    parser.add_argument('--batch-size', '-b', default = 1, type = int, help = 'Batch size for experinment')
    parser.add_argument('--epochs', '-e', default = 300, type = int, help = 'Number of epoch to run experinment')
    parser.add_argument('--name', '-n', default = "test", type = str, help = 'Name of Experinment')
    return parser.parse_args()