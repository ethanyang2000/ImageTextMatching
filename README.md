# Image Text Matching
## Introduction
This project tackles the following image-text-matching problem: Given a product on Taobao(such as a skirt or a pair of shoes), usually there will be a few different colors of this product on sell. And the seller would give a few phrases to describe them. However, there are no strict requirements for the description, so the same color could be linked to thousands of phrases. Your mission is to assign a phrases to a product within a group of texts and a group of products. The phrases may not be so accurate, just find the one that describe the product as best as possible.

## Environments
   + python 3.8.0
   + torch 1.7.0+cu110
   + torchvision 0.8.1
   + transformers 4.20.0
   + numpy 1.21.5

## Codes
Detailed information can be found in the [report](https://github.com/ethanyang2000/ImageTextMatching/blob/main/report.pdf).

   + Args: hyperparameters
   + Accuracy: metrics of our algorithm
   + BasicBlock，ResNet，ResNet18: architecture of the image feature extractor
   + myDataset: data argumentation
   + trainer: main class, including data read, training, and inference. You can start training and inference based on the defined path by calling `train` or `inference`.
   + SupConLoss: loss of supervised contrastive learning
   + MatchingNet: the code of the main matcher.
   + InitNet: feature extractor, including BERT and ResNet.  

## Usage & Hyperparameters
Put the `dataset/medium` and the `project.py` in the same directory.
You need to set the `inference` by any strings during inference. Detailed parameters can be found in my [wandb.ai](https://wandb.ai/ethanyang/course_project?workspace=user-ethanyang%E3%80%82)  

   + hiddensize=512
   + batch_size = 256
   + extra_crop = True
   + crop = False
   + normalize='batch'
   + use_attn = True
   + use_gru = True
   + fix_resent=True
   + train_resnet = True
   + train_both = False