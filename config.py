class Config:
    def __init__(self):
        self.exp_name = 'debug'
        self.exp_dir = f'exps/{self.exp_name}'
        self.mode = 'train'

        self.data_root = 'x4k'
        self.num_frames = 5
        self.z_dim = 256

        self.epochs = 1000
        self.save_epoch = 20
        self.viz_step = 500
        self.batch_size = 4
        self.patch_size = 64
        self.num_workers = 8

        self.lr = 1e-4
        self.min_lr = 1e-6
