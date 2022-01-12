import models
import dataloaders
import trainers


def prepare_train(opt):
    model = models.make(opt)
    train_dataloader, val_dataloader = dataloaders.make(opt)
    train_func = trainers.make(opt)
    return model, train_dataloader, val_dataloader, train_func


