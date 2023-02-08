<img src="https://user-images.githubusercontent.com/15837806/101306054-f28f2900-3811-11eb-99eb-cf6bc0d56b9c.png"
     alt="Input image"
     style="float: left; margin-right: 10px;" />
<img src="https://user-images.githubusercontent.com/15837806/101306056-f327bf80-3811-11eb-9a38-1fd7d0bc7bae.gif"
     alt="Optimization"
     style="float: left; margin-right: 10px;" />

# Field of Junctions Initialization

Exploration of feedforward methods for predicting junction parameters given patches of an image.

## Description of Folders

1. Junction Networks Preliminary: CNNs for learning junction parameters from junction image samples with fixed sizes.
2. Line Networks: CNNs for learning line parameters from line image samples with fixed sizes.
3. Line Networks with Parameter Transfer: CNNs for learning line parameters from patches of arbitrarily sized images.
4. Line To Junction Networks: CNNs/MLPs for learning junction parameters from junction image samples that have been processed as lines.
5. Junction Mixer Networks: MLP-Mixer networks for learning junction parameters from junction image samples with fixed sizes, with and without preprocessing as lines. 

## Requirements

The code is implemented in pytorch. It has been tested using pytorch 1.6 but it should work for other pytorch 1.x versions. The following packages are required:

- python 3.x
- pytorch 1.x
- numpy >= 1.14.0
- torch-summary
