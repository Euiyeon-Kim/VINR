from dotmap import DotMap

config = DotMap(
    {
        'exp_name': 'db',
        'model_type': 'liif_flow',    # ['mod', 'liif', 'xvfi', 'liif_flow']

        'common': {
            'data_root': 'x4k',
            'patch_size': 96,
            'num_workers': 8,

            'num_frames': 5,
            'z_dim': 32,

            'epochs': 1500,
            'viz_steps': 500,
            'val_epoch': 5,
            'save_epoch': 20,

            'batch_size': 2,
            'lr': 2e-4,
            'factor': 0.5,
            'patience': 10,
            'min_lr': 1e-6,
        },

        # Config model arch option
        'mod': {
            'hidden': 256,
            'depth': 5,
            'w0': 200
        },

        'liif': {
            'encoder_blocks': 2,
            'sample_q': None,
            'multiscale': True,
            'hidden': 256,
            'depth': 5,
            'w0': 200
        },

        'liif_flow': {
            'encoder_blocks': 2,
            'sample_q': None,
            'multiscale': False,
            'hidden': 256,
            'depth': 5,
            'w0': 200
        },
    }
)

# class Config:
#     def __init__(self):
#         self.exp_name = 'modrgb_debug'
#         self.model_type = 'mod'                 # ['mod', 'liif', 'xvfi']
#         self.output_type = 'rgb'                # ['rgb', 'flow']
#         self.exp_dir = f'exps/{self.exp_name}'
#
#         self.z_dim = 64
#         self.num_frames = 5
#
#         # Dataset
#         self.data_root = 'x4k'
#         self.patch_size = 96
#         self.num_workers = 8
#
#         # Train setting
#         self.epochs = 1000
#         self.viz_step = 500
#         self.val_epoch = 5
#         self.save_epoch = 20
#
#         # Hyper params
#         self.batch_size = 2
#         self.lr = 1e-4
#         self.min_lr = 1e-6
#
#         # Flow
#         self.flow_lambda = 1.
#
#         # LIIF
#         self.sample_q = None
