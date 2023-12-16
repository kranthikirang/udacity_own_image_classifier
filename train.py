import sys
from learning import TRANING
import argparse

parser = argparse.ArgumentParser(
    description = 'Parser for train.py'
)
parser.add_argument('--data_dir', dest="data_dir", help="data directory to read the data files", default="./flowers/")
parser.add_argument('--save_dir', dest="save_dir", help="destination directory to save the model checkpoint", default=".")
parser.add_argument('--arch', dest="arch", help="pre-trained model architecture", default="vgg19_bn")
parser.add_argument('--learning_rate', dest="learning_rate", help="learning rate", type=float, default=0.001)
parser.add_argument('--hidden_units', dest="hidden_units", help="hidden layer units ',' separated", type=str, default="512")
parser.add_argument('--epochs', dest="epochs", default=10, help="Number of loops that you want to train the model", type=int)
parser.add_argument('--dropout', dest="dropout", help="dropout strategy for forward pass while learning", type=float, default=0.5)
parser.add_argument('--category_names', dest="category_names", help="category names or labels as json file path", type=str, default='cat_to_name.json')
parser.add_argument('--gpu', action=argparse.BooleanOptionalAction, dest="gpu", default=True, type=bool)
parser.add_argument('--debug', action=argparse.BooleanOptionalAction, dest="debug", default=False)

args = parser.parse_args()
data_dir = args.data_dir
save_dir = args.save_dir
learning_rate = args.learning_rate
arch = args.arch
hidden_units = args.hidden_units
gpu = args.gpu
epochs = args.epochs
category_names = args.category_names
dropout = args.dropout
debug = args.debug

def main():
    traning = TRANING(data_dir=data_dir, save_dir=save_dir, model_name=arch, learning_rate=learning_rate, hidden_units=hidden_units, dropout=dropout, 
                      epochs=epochs, category_names=category_names, gpu_flag=gpu, debug=debug)
    # traning.train(train_network=False)
    traning.train()

if __name__== "__main__":
    main()