import torch
import os
from torchvision import datasets, transforms

def get_train_dataloaders(train_dir, batch_size=32):
    data_transforms = {
        'train': transforms.Compose([
            #transforms.Resize(size=(224,224)),
            transforms.Resize(280),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.493, 0.493, 0.493], [0.246, 0.246, 0.246])
        ]),
    }
    
    image_datasets = datasets.ImageFolder(train_dir, data_transforms['train'])
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, shuffle=True, num_workers=4) 
    dataset_sizes = len(image_datasets)
    train_class_names = image_datasets.classes
    print(train_class_names)
    
    return dataloaders
    

def get_dataloaders(val_dir, batch_size):
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'test': transforms.Compose([
            #transforms.Resize(size=(224,224)),
            transforms.Resize(280),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.493, 0.493, 0.493], [0.246, 0.246, 0.246])
        ]),
    }

    # Create the data loader
    image_datasets = datasets.ImageFolder(val_dir, data_transforms['test'])
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=1,
                                              shuffle=False, num_workers=4) 
    dataset_sizes = len(image_datasets)
    val_class_names = image_datasets.classes

    # Show the dataset distributions
    print("Validation dataset size: " + str(dataset_sizes))
    print(val_class_names)

    covid_data_dir = val_dir + 'covid-19'
    normal_data_dir = val_dir + 'normal'
    pneumonia_data_dir = val_dir + 'pneumonia'

    print("######### Validation Dataset #########")
    val_covid_num = len(os.listdir(covid_data_dir))
    val_normal_num = len(os.listdir(normal_data_dir))
    val_pneumonia_num = len(os.listdir(pneumonia_data_dir))
    print("covid-19 size: " + str(val_covid_num))
    print("normal size: " + str(val_normal_num))
    print("pneumonia size: " + str(val_pneumonia_num))
    
    return dataloaders