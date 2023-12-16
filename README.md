# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

## Command Line Arguments for train.py
- --data_dir DATA_DIR   data directory to read the data files
- --save_dir SAVE_DIR   destination directory to save the model checkpoint
- --arch ARCH           pre-trained model architecture (default=vgg19bn)
- --learning_rate LEARNING_RATE
                        learning rate (default=0.001)
- --hidden_units HIDDEN_UNITS
                        hidden layer units ',' separated (default=512) - only one hidden layer
- --epochs EPOCHS       Number of loops that you want to train the model (default=10)
- --dropout DROPOUT     dropout strategy for forward pass while learning (default=0.5)
- --category_names CATEGORY_NAMES
                        category names or labels as json file path (default=cat_to_name.json)
- --gpu, --no-gpu (default=True)
- --debug, --no-debug (default=False)

### train.py Example
```
python3 train.py --data_dir ./flowers --hidden_units 4096,1024 --arch vgg19 --epochs 10 --debug
```

## Command Line Arguments for predict.py

- --image_path IMAGE_PATH
                    path to the image (Mandatory)
- --checkpoint CHECKPOINT
                    model checkpoint path (default="./vgg19bn_checkpoint.pth")
- --category_names CATEGORY_NAMES
                    category names or labels as json file path (default=cat_to_name.json)
- --top_k TOP_K         top classes and probabilities (default=5)
- --gpu, --no-gpu (default=True)
- --debug, --no-debug (default=False)

### predict.py Example
```
python3 predict.py --checkpoint ./vgg19_checkpoint.pth --image_path ./flowers/test/31/image_08070.jpg --top_k=3 --debug
```
### Output Example
```
2023-12-15 19:23:41,472 - predict - INFO - Top 3 classes: predictions are
2023-12-15 19:23:41,472 - predict - INFO - ================================
2023-12-15 19:23:41,472 - predict - INFO - love in the mist: 98.6073911190033
2023-12-15 19:23:41,472 - predict - INFO - stemless gentian: 0.17923658015206456
2023-12-15 19:23:41,472 - predict - INFO - desert-rose: 0.1562096062116325
```