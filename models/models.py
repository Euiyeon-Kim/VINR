models = {}


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(opt):
    model = models[opt.model_type](opt['common'], opt[opt.model_type])
    return model
