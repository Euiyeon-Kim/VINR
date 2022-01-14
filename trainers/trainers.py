trainers = {}


def register(name):
    def decorator(cls):
        trainers[name] = cls
        return cls
    return decorator


def make(opt):
    m_type = 'liif_flow_v2' if opt.model_type == 'liif_flow_v3' else opt.model_type
    train_f = trainers[m_type](opt['common'], opt.exp_name)
    return train_f
