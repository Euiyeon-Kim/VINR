models = {}


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(opt):
    m_type = 'liif_flow' if 'liif_flow' in opt.model_type else opt.model_type
    model = models[opt.model_type](opt['common'], opt[m_type])
    return model
