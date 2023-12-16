import sys, json

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from collections import OrderedDict
import numpy as np
from PIL import Image

import logging

class PREDICT():
    """
    PREDICT class to support model checkpoint loading and predicting methods
    Parameters:
        checkpoint - Model checkpoint path
        top_k - Defined top classes and probabilities of the image after prediciton
        gpu_flag - gpu/mps(Mac) flag
    """
    def __init__(self, image_path=None, checkpoint=None, top_k=5, gpu_flag=False, debug=False):
        self.image_path=image_path
        self.checkpoint=checkpoint
        self.top_k = top_k
        self.debug=debug

        device="cpu"
        if gpu_flag:
            if torch.cuda.is_available():
                device="cuda"
            elif torch.backends.mps.is_built():
                device="mps"
        self.device = torch.device(device)

        self.logger = logging.getLogger('predict')
        ch = logging.StreamHandler()
        log_level='INFO'
        if debug:
            log_level='DEBUG'
        self.logger.setLevel(getattr(logging, log_level))
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def predict(self, ret_dict=False):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        self.model.to(self.device)
        img = self.process_image()
        img = torch.unsqueeze(img,0).to(self.device).float()
        
        self.model.eval()
        logps = self.model(img)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(self.top_k, dim=1)
        class_list= top_class.tolist()[0]
        p_list = top_p.tolist()[0]

        idx_to_flower = {v:self.category_names[k] for k, v in self.model.class_to_idx.items()}
        predicted_flowers_list = [idx_to_flower[i] for i in class_list]

        if ret_dict:
            self.logger.debug(f'Preditecd items are: {predicted_flowers_list}')
            classes = {}
            for index, item in enumerate(predicted_flowers_list):
                classes[item]=p_list[index]*100
            self.logger.info(f'Top {self.top_k} classes: predictions are')
            self.logger.info(f'==================================')
            for key, val in classes.items():
                self.logger.info(f'{key}: {val}')
            # return classes
        else:
            return p_list, predicted_flowers_list

    def process_image(self):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''
        new_width,new_height = 224,224 #https://stackoverflow.com/questions/16646183/crop-an-image-in-the-centre-using-pil
        # TODO: Process a PIL image for use in a PyTorch model
        with Image.open(self.image_path) as im:
            image_width, image_height = im.size
            if image_width > 256 or image_height > 256:
                modified_image = im.resize(size=(256,256))
            else:
                modified_image = im

            image_width, image_height = modified_image.size
            left = (image_width - new_width)/2
            top = (image_height - new_height)/2
            right = (image_width + new_width)/2
            bottom = (image_height + new_height)/2
            modified_image = modified_image.crop((left, top, right, bottom))
            
        image_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                            ])
        image_tensor = image_transform(modified_image)
    #     np_image_tensor = np.array(image_tensor)
    #     return np_image_tensor
        return image_tensor
    
    def load_checkpoint(self, category_names):
        """
        This method will load from model checkpoint and handover the information to/from TRAINING class.
        """
        self.logger.info(f'Using device {self.device}')
        try:
            self.logger.info(f'Loading the model checkpoint: {self.checkpoint}')
            checkpoint = torch.load(self.checkpoint, map_location=self.device)
            self.model_name = checkpoint['name']
            self.hidden_units = checkpoint['hidden_units']
            self.dropout = checkpoint['dropout']
            self.learning_rate = checkpoint['learning_rate']
            self.epochs = checkpoint['epochs']
            self.model_arch = checkpoint['arch']
        except Exception as e:
            self.logger.error(e)
    
        self.model = getattr(models, self.model_name)(pretrained=True)
        self.logger.info(f'Model {self.model_name} classifier is: {self.model.classifier}')

        #Let's freeze the parameters again to start
        for param in self.model.parameters():
            param.requires_grad = False

        #read the json to use in prediction label mapping
        with open(category_names, 'r') as f:
            self.category_names = json.load(f, strict=False)

        # Now let's set the classfier from the received model information from checkpoint
        traning = TRANING(model_name=self.model_arch, learning_rate=self.learning_rate, hidden_units=self.hidden_units, dropout=self.dropout, 
                      epochs=self.epochs, category_names=category_names, debug=self.debug)
        try:
            print("calling set_classifier at load_checkpoint")
            self.model = traning.set_classifier(model=self.model)
        except Exception as e:
            self.logger.error(e)
        
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.class_to_idx = checkpoint['class_2_idx']

        self.logger.info(f'After loading from checkpoint classifier is: {self.model.classifier}')

class TRANING():
    """
    Traning Class for different type of pre-trained models.
    Parameters:
        data_dir - Full path to the data directory to read the data/image files
        save_dir - Full path or relative path to store the model checkpoint
        model_name - Name of the pre-trained model
        hidden_units - A command separated string of hidden layer units
        learning_rate - Learning rate constant to train the model
        dropout - Dropout rate to train the model
        epochs - Number of loops to train the model
        gpu_flag - gpu/mps(Mac) flag
    Returns:
        None
    """
    def __init__(self, data_dir=None, save_dir=None, model_name=None, hidden_units=512, learning_rate=0.001, dropout=0.5, epochs=10, category_names=None, gpu_flag=False, debug=False):
        self.data_dir=data_dir
        self.save_dir=save_dir
        self.hidden_units=hidden_units
        self.hidden_units_list=hidden_units.split(',') #We use list to make sure we get number of hidden layers that we want
        self.learning_rate=learning_rate
        self.dropout=dropout
        self.epochs=epochs
        self.model_arch=model_name

        if model_name.upper() == 'VGG':
            self.model_name = 'vgg19'
            self.in_features = 25088
        elif model_name.upper() == 'DENSENET':
            self.model_name = 'densenet121'
            self.in_features = 1024
        elif model_name.upper() == "ALEXNET":
            self.model_name = 'alexnet'
            self.in_features = 9216

        print(f'model_name is: {model_name}')

        device="cpu"
        if gpu_flag:
            if torch.cuda.is_available():
                device="cuda"
            elif torch.backends.mps.is_built():
                device="mps"
        self.device = torch.device(device)

        with open(category_names, 'r') as f:
            self.category_names = json.load(f)

        self.logger = logging.getLogger('train')
        ch = logging.StreamHandler()
        log_level='INFO'
        if debug:
            log_level='DEBUG'
        self.logger.setLevel(getattr(logging, log_level))
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def load_data(self):
        """
        Load_data method will assume data directory have correct data sets present
        in appropriate directories. The pre-trained networks you'll use were trained 
        on the ImageNet dataset where each color channel was normalized separately. 
        For all three sets you'll need to normalize the means and standard deviations 
        of the images to what the network expects. For the means, it's [0.485, 0.456, 0.406] 
        and for the standard deviations [0.229, 0.224, 0.225], calculated from the ImageNet 
        images. These values will shift each color channel to be centered at 0 and range from 
        -1 to 1.
        """
        train_dir = self.data_dir + '/train'
        valid_dir = self.data_dir + '/valid'
        test_dir = self.data_dir + '/test'

        self.train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(size=(224, 224), antialias=True),
                                       transforms.RandomHorizontalFlip(p=0.5),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
        self.valid_transforms = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                    [0.229, 0.224, 0.225])])

        self.test_transforms = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                    [0.229, 0.224, 0.225])])

        # TODO: Load the datasets with ImageFolder
        self.train_data = datasets.ImageFolder(train_dir, transform=self.train_transforms)
        self.valid_data = datasets.ImageFolder(valid_dir, transform=self.valid_transforms)
        self.test_data = datasets.ImageFolder(test_dir, transform=self.test_transforms)

        # TODO: Using the image datasets and the trainforms, define the dataloaders
        self.trainloader = torch.utils.data.DataLoader(self.train_data, batch_size=64, shuffle=True)
        self.validloader = torch.utils.data.DataLoader(self.valid_data, batch_size=64, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(self.test_data, batch_size=32, shuffle=True)

        self.logger.debug(f'Train data is: {self.train_data}')
        self.logger.debug(f'Tets data is: {self.test_data}')
        self.logger.debug(f'Validation data is: {self.valid_data}')

    def train(self, train_network=True):
        """
        This method will try to download the pre-trained model and train based on hyper parameters.
        This will also load the data using load_data method and save the model checkpoint based on 
        provided save directory.
        """
        self.logger.info(f'Using device {self.device}')

        self.model = getattr(models, self.model_name)(pretrained=True)
        self.logger.info(f'Model {self.model_name} classifier is: {self.model.classifier}')

        for param in self.model.parameters():
            param.requires_grad = False
        try:
            self.load_data()
            self.set_classifier()
            if train_network:
                self.train_network()
            self.save_checkpoint()
        except Exception as e:
            self.logger.error(e)

    def save_checkpoint(self):
        """
        This method will save the model to checkpoint onm defined save_dir
        """
        checkpoint = {'arch': self.model_arch,
                    'name': self.model_name,
                    'in_features': self.in_features,
                    'hidden_units': self.hidden_units,
                    'dropout': self.dropout,
                    'learning_rate': self.learning_rate,
                    'epochs': self.epochs,
                    'state_dict': self.model.state_dict(),
                    'class_2_idx': self.train_data.class_to_idx,
                    'optimizer_dict': self.optimizer.state_dict()
                }
        # for key, val in self.model.classifier():
        #     print(key, val)
        self.checkpoint_filename = f'{self.save_dir}/{self.model_name}_checkpoint.pth'
        self.logger.info(f'Saving model checkpoint to {self.checkpoint_filename}')
        try:
            torch.save(checkpoint, self.checkpoint_filename)
        except Exception as e:
            self.logger.error(e)
        self.logger.debug(f'checkpoint details are: {checkpoint}')

    def train_network(self):
        """
        This method will run loop for defined epochs and train the network
        """
        steps = 0
        running_loss = 0
        running_accuracy = 0
        print_every = 15

        train_losses, validation_losses = [], []

        for epoch in range(self.epochs):
            for inputs, labels in self.trainloader:
                steps += 1
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                logps = self.model(inputs)
                loss = self.criterion(logps, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                
                ps = torch.exp(logps)
                top_ps, top_class = ps.topk(1,dim=1)
                matches = (top_class == labels.view(*top_class.shape)).type(torch.FloatTensor)
                accuracy = matches.mean()
                running_accuracy += accuracy.item()
                
                if steps % print_every == 0:
                    validation_loss = 0
                    validation_accuracy = 0
                    self.model.eval()
                    with torch.no_grad():
                        for inputs, labels in self.validloader:
                            inputs, labels = inputs.to(self.device), labels.to(self.device)
                            logps = self.model(inputs)
                            batch_loss = self.criterion(logps, labels)
                            
                            validation_loss += batch_loss.item()
                            
                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            validation_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                            
                    train_losses.append(running_loss/len(self.trainloader))
                    validation_losses.append(validation_loss/len(self.validloader))
                
                    self.logger.info(f"Epoch {epoch+1}/{self.epochs}.. "
                        f"Train loss: {train_losses[-1]:.3f}.. "
                        f"Train accuracy: {(running_accuracy/print_every)*100:.3f}%.. "
                        f"Validation loss: {validation_losses[-1]:.3f}.. "
                        f"Validation accuracy: {(validation_accuracy/len(self.validloader))*100:.3f}%")
                    running_loss = 0
                    running_accuracy = 0
                    self.model.train()

    def set_classifier(self, model=None):
        """
        This method will set the classfier to the model based on the provided hiidden layers and
        hyper parameters.
        """
        external_call=False
        if model: #this is required if we only want to use set_classifier method from outside class
            self.logger.info(f'Using the model requested')
            self.model=model
            external_call=True
        classifier_list=[]
        #For the first hidden layer
        in_features=self.in_features
        for index, layer in enumerate(self.hidden_units_list):
            out_features=int(layer)
            self.logger.debug(f'in_features: {in_features}, out_features: {out_features}, index: {index}')
            if index == 0 and index == len(self.hidden_units_list) - 1:
                classifier_list.append((f'fc{index}', nn.Linear(in_features, out_features)))
                classifier_list.append((f'relu{index}', nn.ReLU()))
                classifier_list.append((f'dropout{index}', nn.Dropout(self.dropout)))
                in_features = out_features
                out_features = len(self.category_names)
                classifier_list.append((f'fc{index}-last', nn.Linear(in_features, out_features)))
                classifier_list.append(('output', nn.LogSoftmax(dim=1)))
            elif index == len(self.hidden_units_list) - 1:
                classifier_list.append((f'fc{index}', nn.Linear(in_features, out_features)))
                classifier_list.append((f'relu{index}', nn.ReLU()))
                classifier_list.append((f'dropout{index}', nn.Dropout(self.dropout)))
                in_features = out_features
                out_features = len(self.category_names)
                classifier_list.append((f'fc{index}-last', nn.Linear(in_features, out_features)))
                classifier_list.append(('output', nn.LogSoftmax(dim=1)))
            else:
                classifier_list.append((f'fc{index}', nn.Linear(in_features, out_features)))
                classifier_list.append((f'relu{index}', nn.ReLU()))
                classifier_list.append((f'dropout{index}', nn.Dropout(self.dropout)))
            in_features = out_features #this is make sure we configure properly for next hidden layer
        
        self.classifier = nn.Sequential(OrderedDict(classifier_list))

        self.model.classifier = self.classifier
        if external_call:
            return self.model

        self.criterion = nn.NLLLoss()

        # Only train the classifier parameters, feature parameters are frozen
        self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=0.001)

        self.model.to(self.device)
        self.logger.info(f'After calculation classifier is: {self.model.classifier}')