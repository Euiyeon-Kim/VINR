import models
import dataloaders
import trainers


def prepare_train(opt):
    if opt.model_type == 'liif_flow':
        assert not opt[opt.model_type].sample_q, "LIIF Flow cannot be trained with sampling"
    model = models.make(opt)
    train_dataloader, val_dataloader = dataloaders.make(opt)
    train_func = trainers.make(opt)
    return model, train_dataloader, val_dataloader, train_func


