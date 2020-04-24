configs = [
    # pure mode
    {'mode': 'pure', 'model_name': 'model_epochs100_percent0_pure_vis', 'pure_ratio': 1.0,
     'synthetic_ratio': 0.0, 'augmented_ratio': 0.0},
    # none_only
    {'mode': 'none', 'model_name': 'model_epochs100_percent100_none_only_vis', 'pure_ratio': 0.0,
     'synthetic_ratio': 1.0, 'augmented_ratio': 0.0},
    # scanner class mode
    {'mode': 'fine_tune_scanner', 'model_name': 'model_epochs100_percent0_pure_vis', 'pure_ratio': 1.0,
     'synthetic_ratio': 1.0, 'augmented_ratio': 0.0},
    {'mode': 'train_scanner', 'model_name': 'model_epochs100_percent100_scanner_vis', 'pure_ratio': 1.0,
     'synthetic_ratio': 1.0, 'augmented_ratio': 0.0},
]
