import torch
from torch.utils.data import DataLoader, TensorDataset 
import time
import os
import sys
import configparser
import random

import utils 
import network

# Program starts here if pickle folders are not updated
train_images, train_labels  = utils.pickle_data(file = './Data/pickled_data/gray32_train_dataset')
test_images, test_labels  = utils.pickle_data(file = './Data/pickled_data/gray32_test_dataset')
val_images, val_labels  = utils.pickle_data(file = './Data/pickled_data/gray32_val_dataset')

# Collect hyper params
config = configparser.ConfigParser()
config.read(
    './src/capsnet_config.ini'
    )
batch_size = int(config['network']['batch_size'])
learning_rate = float(config['network']['lr'])
epochs = int(config['network']['epochs'])

# Create log directory
t = time.localtime()
log_format = time.strftime("%Y%m%d_%H%M%S", t)
log_dir =  os.path.join("./Results/", log_format)
os.mkdir(log_dir)

# Create log dict 

def create_data_dict():
    reporter = utils.Reporter(f"./{log_dir}/model_log.xlsx")
    data = {"Epoch": '',
            "Batch ID": '', 
            "Loss":{"Margin": '',
                    "Recon": '',
                    "Total": ''},
            "Accuracy": ''
            }
    
    return data, reporter


def train(model, train_loader, optimizer, device, epoch):
    log_batch = 20
    data, report_writer = create_data_dict()
    model.train() # set the model to training mode

    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        x_batch = x_batch.permute(0, 3, 1, 2) # [batch, C, H, W]
        y_batch_one_hot = torch.nn.functional.one_hot(y_batch.to(torch.int64), num_classes=43)

        # aug_bool = random.randint(0,1)
        # if aug_bool:
        #     augmentor = utils.ImgAug(x_batch, 0.3)
        #     x_batch = augmentor.apply_trans()

        if device == 'cuda':
            x_batch, y_batch_one_hot = x_batch.to('cuda'), y_batch.to('cuda')

        pred_idx, recons_img = model(x_batch, y_batch_one_hot) # recons_img = [batch, C, H, W]
        margin_loss, recon_loss, total_loss = model.loss_fn(recons_img, y_batch_one_hot)
        total_loss.backward()
        optimizer.step() # update the params to be optimized (weights, biases, routing weights)

        if (batch_idx+1) % log_batch == 0:
            correct_pred = torch.eq(pred_idx.view_as(y_batch), y_batch).sum().item()
            correct_pred /= batch_size
            data["Epoch"] = epoch
            data["Batch ID"] = batch_idx+1
            data["Loss"]["Margin"] = margin_loss.item()
            data["Loss"]["Recon"] = recon_loss.item()
            data["Loss"]["Total"] = total_loss.item()
            data["Accuracy"] = correct_pred*100 # From training batch_images
            report_writer.record(data, sheet="TRAIN")
            print('Epoch: {}, Batch: {}/{}, Loss: {:.4f}, Accuracy: {:.4f}'
                    .format(epoch, batch_idx+1, len(train_loader), total_loss.item(), data["Accuracy"]))
            correct_pred = 0

def test(model, test_loader, device, epoch):
    """
    Can be used for both test and validation
    """
    # set the model to testing mode
    data, report_writer = create_data_dict()
    model.eval() 
    loss, correct_pred = 0, 0
    for batch_idx, (x_batch, y_batch) in enumerate(test_loader):
        x_batch = x_batch.permute(0, 3, 1, 2)
        y_batch_one_hot = torch.nn.functional.one_hot(y_batch.to(torch.int64), num_classes=43)
        if device == 'cuda':
            x_batch, y_batch_one_hot = x_batch.to('cuda'), y_batch_one_hot.to('cuda')
        
        pred_idx, recons_img = model(x_batch)
        _, _, total_loss = model.loss_fn(recons_img, y_batch_one_hot)

        loss += total_loss.item()
        correct_pred += torch.eq(pred_idx.view_as(y_batch), y_batch).sum().item()
        
    loss /= len(test_loader)
    correct_pred /= len(test_loader.dataset)
    accuracy = correct_pred * 100

    data["Epoch"] = epoch
    data["Loss"]["Total"] = loss
    data["Accuracy"] = accuracy
    report_writer.record(data, sheet="TEST")
    print("Epoch {} - Loss: {:.4f}; Accuracy: {:.4f} %". format(epoch, loss, accuracy))

    return loss, accuracy
            
def data_loader(images, 
                labels, 
                batch_size: int,
                shuffle: bool=False) -> DataLoader:
    """
    :param images: 3D array [H, W, Channels]
    :param labels: 1D array
    :param batch_size: number of images per batch

    :return: an iterable object
    """
    images_tensor = torch.Tensor(images)
    labels_tensor = torch.Tensor(labels)

    dataset = TensorDataset(images_tensor, labels_tensor)
    data_loader = DataLoader(dataset, batch_size, shuffle=shuffle)

    return data_loader


if __name__ == '__main__':
    model = network.CapsNet()
    if torch.cuda.is_available():
        device_as_str = 'cuda'
        model = network.CapsNet().to('cuda') 
    else:
        device_as_str = 'cpu'
    print(f'{device_as_str} will be used')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Reduce learning rate when a metric has stopped improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-5) 

    # Prepare dataset
    val_loader = data_loader(val_images, val_labels, batch_size)
    train_loader = data_loader(train_images, train_labels, batch_size, shuffle=True)

    for epoch in range(epochs):  
        train(model, train_loader, optimizer, device_as_str, epoch) 
        val_loss, val_accuracy = test(model, val_loader, device_as_str, epoch) # validate the model at each epoch
        scheduler.step(val_loss)
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'loss': val_loss
            }, f"{log_dir}/epoch{epoch+1}.pth")