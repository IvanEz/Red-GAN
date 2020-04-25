import torch
import segmentation_models_pytorch as smp
import os


class Model:
    def __init__(self, args):
        self.fold = 'fold_' + str(args.fold)
        self.model_dir = os.path.join(args.root_dir, 'models', self.fold, args.model_name)
        self.mode = args.mode
        self.encoder = args.encoder
        self.activation = args.activation
        self.classes = 3

    def create_model(self):
        if self.mode == 'continue_train':
            model = torch.load(self.model_dir)
            model.decoder.layer1.require_grad = False
            optimizer = torch.optim.Adam([
                {'params': model.decoder.parameters(), 'lr': 1e-4},
                {'params': model.encoder.parameters(), 'lr': 1e-6},
            ])
        elif self.mode == 'fine_tune':
            model = torch.load(self.model_dir)
            freeze_layers_encoder = [
                 model.encoder.conv1,
                 model.encoder.bn1,
                 model.encoder.relu,
                 model.encoder.maxpool,
                 model.encoder.layer1,
                 model.encoder.layer2,
                 model.encoder.layer3
            ]

            fine_tune_layers_encoder = list(model.encoder.layer4.parameters())

            freeze_layers_decoder = [model.decoder.layer1.block,
                                     model.decoder.layer2.block,
                                     model.decoder.layer3.block,
                                     model.decoder.layer4.block]

            fine_tune_layers_decoder = list(model.decoder.layer5.
                                            parameters()) + list(model.decoder.final_conv.parameters())
            for layer in freeze_layers_encoder:
                layer.require_grad = False
            for layer in freeze_layers_decoder:
                layer.require_grad = False
            optimizer = torch.optim.Adam([
                {'params': fine_tune_layers_decoder, 'lr': 1e-4},
                {'params': fine_tune_layers_encoder, 'lr': 1e-6},
            ])
        else:
            model = smp.Unet(
                encoder_name=self.encoder,
                encoder_weights=None,
                classes=self.classes,
                activation=self.activation,
            )
            model.decoder.layer1.require_grad = False
            optimizer = torch.optim.Adam([
                {'params': model.decoder.parameters(), 'lr': 1e-4},
                {'params': model.encoder.parameters(), 'lr': 1e-6},
            ])

        return model, optimizer
