class Config:
    def __init__(self):
        self.exp_name = 'liif_debug'
        self.exp_dir = f'exps/{self.exp_name}'
        self.mode = 'train'     # 'train' or 'test'
        self.model = 'liif'     # 'mod' or 'liif'

        self.data_root = 'x4k'
        self.num_frames = 5
        self.z_dim = 256

        self.epochs = 1000
        self.val_epoch = 1
        self.viz_step = 1
        self.batch_size = 2
        self.patch_size = 96
        self.num_workers = 8

        self.lr = 1e-4
        self.T_max = 5
        self.min_lr = 1e-6

        self.liif_lambda = 0.5
        self.save_epoch = 20
        self.lr_size = 24
        self.scale_max = 4