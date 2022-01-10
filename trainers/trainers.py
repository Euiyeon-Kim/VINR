trainers = {}


def register(name):
    def decorator(cls):
        trainers[name] = cls
        return cls
    return decorator


def make(opt):
    train_f = trainers[opt.model_type](opt['common'], opt.exp_name)
    return train_f
