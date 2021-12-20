class Config:
    def __init__(self):
        self.exp_name = 'mod_debug'
        self.exp_dir = f'exps/{self.exp_name}'
        self.mode = 'train'     # 'train' or 'test'

        self.z_dim = 128
        self.num_frames = 5

        # Dataset
        self.data_root = 'x4k'
        self.patch_size = 96
        self.num_workers = 8

        # Train setting
        self.epochs = 1000
        self.viz_step = 1
        self.batch_size = 2
        self.lr = 1e-4
        self.min_lr = 1e-4

        # LIIF
        self.liif_lambda = 0.5
        self.save_epoch = 20
        self.lr_size = 24
        self.scale_max = 4