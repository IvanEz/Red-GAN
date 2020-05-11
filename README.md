# Red-GAN

The repository contains code used for the *"Red-GAN: Attacking class imbalance via conditioned generation. Yet another medical imaging perspective"* accepted at MIDL2020 (http://arxiv.org/abs/2004.10734 ).

### Training
To use the three player design (SPADE design equiped with the segmentor) introduced in the paper, please step to the `unet` folder first and check out the instruction. Once the unet is trained, step to the `gan` folder.

To use the two-player design (vanila [SPADE](https://github.com/NVlabs/SPADE/blob/master/README.md) architecture), step to the `gan` folder right away and specify `--segmentator` option as `None`

### Requirements
Both GAN and U-Net parts are written in Pytorch. For further installation requirements, please check the corresponding folders.

### Citation

To cite the repository/paper please use:

`@article{qasim2020red,
  title={Red-GAN: Attacking class imbalance via conditioned generation. Yet another medical imaging perspective},
  author={Qasim, Ahmad B and Ezhov, Ivan and Shit, Suprosanna and Schoppe, Oliver and Paetzold, Johannes C and Sekuboyina, Anjany and Kofler, Florian and Lipkova, Jana and Li, Hongwei and Menze, Bjoern},
  journal={arXiv preprint arXiv:2004.10734},
  year={2020}
}`

