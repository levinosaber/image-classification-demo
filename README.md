# Demo of multimodel training code of image classification
1. Use of the argparse class to allow specifying hyperparameters directly in the training launch command.
2. Ability to fix the network's initialization method by specifying --seed in the launch command to achieve reproducible results.
3. Implementation of a more advanced learning strategy, cosine warm-up: uses a smaller learning rate (lr) in the first round of training, and then gradually decreases the lr from the second epoch as training progresses.
4. Ability to choose the model to use by specifying --model in the launch command.
5. Use of the amp package to implement half-precision training, which minimizes training costs while ensuring accuracy.
6. Custom implementation of the data loading class.
7. Tensorboard visualization can be enabled by specifying --tensorboard in the launch command, which is not enabled by default.
Note, before using tensorboard, it must be started with the command 'tensorboard --logdir=log_path', and results can be viewed through the webpage 'http://localhost:6006/'.

Optional hyperparameters for the --model are as follows:
alexnet   vgg   vgg_tiny   vgg_small   vgg_big    googlenet   resnet_small   resnet   resnet_big   resnext   resnext_big
densenet_tiny   densenet_small   densenet   densenet_big   mobilenet_v1   mobilenet_v2   convnext_tiny   convnext_small   convnext   convnext_big   convnext_big

Example training command "python train.py --model mobilenet_v2 --num_classes 5  --epochs 10  --batch_size 16"