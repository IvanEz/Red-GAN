# Red-GAN

The repository contains code used for the *"Red-GAN: Attacking class imbalance via conditioned generation. Yet another medical imaging perspective"* accepted at MIDL2020 (http://arxiv.org/abs/2004.10734 ).
The paper proposes one more data augmentation protocol based on generative adversarial networks. It is based on the [SPADE](https://github.com/NVlabs/SPADE/blob/master/README.md) (that is capable to learn a mapping between label maps and images), and is equiped with two additional components: (a) network conditioning at a global-level information (e.g. acquisition environment or lesion type), (b) a passive player in a form of segmentor introduced into the the adversarial game. The method validate the approach on two medical datasets: BraTS, ISIC. 

### Getting started
To use the three player design (SPADE design equiped with the segmentor) introduced in the paper, one has to first train the segmentor. For that, please step to the `unet` folder and check out the *readme* there. Once the segmentor is trained, step to the `gan` folder.

To use only the two-player design (vanila [SPADE](https://github.com/NVlabs/SPADE/blob/master/README.md) architecture), step to the `gan` folder right away (in this case case `--segmentator` option should be specified as `None`).

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

