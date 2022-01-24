from dotmap import DotMap

config = DotMap(
    {
        'exp_name': 'debug',
        'model_type': 'liif_rel',
        # ['mod']
        # ['liif', 'liif_flow', 'liif_flow_v2', 'liif_flow_v3', 'liif_flow_mask_refine']
        # ['liif_rel']
        'data_type': 'liif_rel',
        # ['mod', 'liif', 'liif_zero_centered', 'liif_rel']

        'common': {
            'data_root': 'x4k',
            'patch_size': 48,
            'test_patch': 64,
            'num_workers': 8,

            'num_frames': 5,
            'z_dim': 32,

            'epochs': 1500,
            'viz_steps': 500,
            'val_save_epoch': 5,
            'save_epoch': 20,

            'batch_size': 3,
            'lr': 1e-4,
            'factor': 0.5,
            'patience': 10,
            'min_lr': 1e-6,
        },

        # Config model and data arch option
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
        'liif_rel': {
            'encoder_blocks': 2,
            'hidden': 256,
            'depth': 5,
            'w0': 200
        },
    }
)