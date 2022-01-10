from torch.utils.data import DataLoader

datasets = {}


def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator


def make(opt):
    train_dataset, val_dataset = datasets[opt.model_type](opt['common'], opt[opt.model_type])
    train_dataloader = DataLoader(train_dataset, batch_size=opt.common.batch_size, shuffle=True,
                                  drop_last=False, num_workers=opt.common.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, drop_last=False, num_workers=1)
    return train_dataloader, val_dataloader
