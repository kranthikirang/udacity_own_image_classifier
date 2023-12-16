import sys
from learning import PREDICT
import argparse

parser = argparse.ArgumentParser(
    description = 'Parser for train.py'
)
parser.add_argument('--image_path', dest="image_path", help="path to the image", default="", required=True)
parser.add_argument('--checkpoint', dest="checkpoint", help="model checkpoint path", default="./vgg19bn_checkpoint.pth")
parser.add_argument('--category_names', dest="category_names", help="category names or labels as json file path", type=str, default='cat_to_name.json')
parser.add_argument('--top_k', dest="top_k", help="top classes and probabilities", type=int, default=5)
parser.add_argument('--gpu', action=argparse.BooleanOptionalAction, dest="gpu", default=True, type=bool)
parser.add_argument('--debug', action=argparse.BooleanOptionalAction, dest="debug", default=False)

args = parser.parse_args()
image_path = args.image_path
checkpoint = args.checkpoint
gpu = args.gpu
category_names = args.category_names
top_k = args.top_k
debug = args.debug

def main():
    prediction = PREDICT(image_path=image_path, checkpoint=checkpoint, top_k=top_k, gpu_flag=gpu, debug=debug)
    prediction.load_checkpoint(category_names=category_names)
    prediction.predict(ret_dict=True)
if __name__== "__main__":
    main()