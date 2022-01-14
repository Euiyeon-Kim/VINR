from dotmap import DotMap

config = DotMap(
    {
        'exp_name': 'db',
        'model_type': 'liif_flow',          # ['mod', 'liif', 'liif_flow', 'liif_flow_v2']
        'data_type': 'liif',                # ['mod', 'liif', 'liif_zero_centered']

        'common': {
            'data_root': 'x4k',
            'patch_size': 96,
            'num_workers': 8,

            'num_frames': 5,
            'z_dim': 32,

            'epochs': 1500,
            'viz_steps': 500,
            'val_save_epoch': 5,
            'save_epoch': 20,

            'batch_size': 2,
            'lr': 1e-4,
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
            'multiscale': False,
            'hidden': 256,
            'depth': 5,
            'w0': 200
        },

        'liif_flow': {
            'encoder_blocks': 2,
            'hidden': 256,
            'depth': 5,
            'w0': 200
        },
    }
)