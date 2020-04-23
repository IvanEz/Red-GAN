configs = [
    # pure mode
    {'mode': 'pure', 'model_name': 'model_epochs100_percent0_pure_vis', 'pure_ratio': 1.0,
     'synthetic_ratio': 0.0, 'augmented_ratio': 0.0},
    # augmented mode
    {'mode': 'augmented', 'model_name': 'model_epochs100_percent50_augmented_vis', 'pure_ratio': 1.0,
     'synthetic_ratio': 0.0, 'augmented_ratio': 0.5},
    {'mode': 'augmented', 'model_name': 'model_epochs100_percent100_augmented_vis', 'pure_ratio': 1.0,
     'synthetic_ratio': 0.0, 'augmented_ratio': 1.0},
    {'mode': 'augmented', 'model_name': 'model_epochs100_percent200_augmented_vis', 'pure_ratio': 1.0,
     'synthetic_ratio': 0.0, 'augmented_ratio': 2.0},
    # elastic mode
    {'mode': 'elastic', 'model_name': 'model_epochs100_percent50_elastic_vis', 'pure_ratio': 1.0,
     'synthetic_ratio': 0.5, 'augmented_ratio': 0.0},
    {'mode': 'elastic', 'model_name': 'model_epochs100_percent100_elastic_vis', 'pure_ratio': 1.0,
     'synthetic_ratio': 1.0, 'augmented_ratio': 0.0},
    {'mode': 'elastic', 'model_name': 'model_epochs100_percent200_elastic_vis', 'pure_ratio': 1.0,
     'synthetic_ratio': 2.0, 'augmented_ratio': 0.0},
    # coregistration mode
    {'mode': 'coregistration', 'model_name': 'model_epochs100_percent50_coregistration_vis', 'pure_ratio': 1.0,
     'synthetic_ratio': 0.5, 'augmented_ratio': 0.0},
    {'mode': 'coregistration', 'model_name': 'model_epochs100_percent100_coregistration_vis', 'pure_ratio': 1.0,
     'synthetic_ratio': 1.0, 'augmented_ratio': 0.0},
    {'mode': 'coregistration', 'model_name': 'model_epochs100_percent200_coregistration_vis', 'pure_ratio': 1.0,
     'synthetic_ratio': 2.0, 'augmented_ratio': 0.0},
    # none mode

    # {'mode': 'none', 'model_name': 'model_epochs100_percent0_none_vis_old', 'pure_ratio': 1.0,
    #  'synthetic_ratio': 0.5, 'augmented_ratio': 0.0},
    # {'mode': 'none', 'model_name': 'model_epochs100_percent100_none_vis', 'pure_ratio': 1.0,
    # 'synthetic_ratio': 1.0, 'augmented_ratio': 0.0},
    # {'mode': 'none', 'model_name': 'model_epochs100_percent200_none_vis', 'pure_ratio': 1.0,
    # 'synthetic_ratio': 2.0, 'augmented_ratio': 0.0},
    # augmented_coregistration
    {'mode': 'augmented_coregistration', 'model_name': 'model_epochs100_percent50_augmented_coregistration_vis',
     'pure_ratio': 1.0, 'synthetic_ratio': 0.5, 'augmented_ratio': 0.5},
    {'mode': 'augmented_coregistration', 'model_name': 'model_epochs100_percent100_augmented_coregistration_vis',
     'pure_ratio': 1.0, 'synthetic_ratio': 1.0, 'augmented_ratio': 1.0},
    {'mode': 'augmented_coregistration', 'model_name': 'model_epochs100_percent200_augmented_coregistration_vis',
     'pure_ratio': 1.0, 'synthetic_ratio': 2.0, 'augmented_ratio': 2.0},
    # none_only
    {'mode': 'none', 'model_name': 'model_epochs100_percent100_none_only_vis', 'pure_ratio': 0.0,
     'synthetic_ratio': 1.0, 'augmented_ratio': 0.0},
    # scanner class mode
    {'mode': 'fine_tune_scanner', 'model_name': 'model_epochs100_percent0_pure_vis', 'pure_ratio': 1.0,
     'synthetic_ratio': 1.0, 'augmented_ratio': 0.0},
    {'mode': 'train_scanner', 'model_name': 'model_epochs100_percent100_scanner_vis', 'pure_ratio': 1.0,
     'synthetic_ratio': 1.0, 'augmented_ratio': 0.0},
]
