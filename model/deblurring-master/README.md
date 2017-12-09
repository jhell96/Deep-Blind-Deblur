deblurring
==============================

Huawei image deblurring competition entry by Dirk Brink and Xav Kearney (Imperial College London EEE).

### Introduction

The project directory structure loosely follows the [cookiecutter data science](https://drivendata.github.io/cookiecutter-data-science) format.

The goal of this project is to remove motion blur from images of license plates in order to improve the accuracy of OCR.

### Project Summary

The first step in the process was to generate training data that accurately represented the problem to be solved. Given 4000 clear images (no blur), we designed a generation script (`random_blur.py`) which takes an input image and generates 100 different copies of it, each blurred in a different direction and by a different amount. For testing we also included random Gaussian blur, and added random noise.

In practice, the noise and random Gaussian blurring did not aid the training, so were removed from the process.

Once 400,000 images had been generated, they are fed into the training of a Convolutional Neural Network with 15 layers. The input is the randomly blurred image, and the desired output is the un-blurred image of a license plate!

A validation set of 33% of the training data was taken, and in addition the network was tested against the 100 test images supplied.

After training on millions of images and creating hundreds of models with different parameters, we found what we believe to be the optimum model architecture given the data at hand.

Due to the fixed size of the input to the network, each image is split up into a number of overlapping images of size 60x20 (the smallest image size in the dataset). To recombine the images, they are blended together using the Python Pillow image library.

### Usage

#### NOTE - Images are cropped to 40x20px for training.  We pushed this close to the limit of the smallest image received for performance benefits. However, this means the script will crash if trying to run on images smaller than this.

Raw input data (un-blurred images) should be placed in `/data/raw/pre-blur/`. Test data (100 blurred images) should be placed in `/data/raw/test`.

IMPORTANT - Please make sure to remove the `.gitkeep` files from all sub-directories in the `data` directory before running any scripts.  They are there to make folder structure apparent but will crash the scripts.

After installing the `requirements.txt` with `pip install -r requirements.txt`, run the `random_blur.py` script in `/src/data/`. This will generate 400,000 cropped image pairs - blurred and un-blurred.

Once image generation is complete, train the model using `/src/models/train_model.py`. We used a batch size of 20 and found that the model converges after around 10,000 batches.

Finished models will be saved in `/models/`. It's possible to use them to de-blur the images in `/data/raw/test` by running the `/src/models/predict_model.py` script with the filename of the model (without `.hdf5`) as a command line parameter.
e.g. `python predict_model.py 20171120-210447_final`

Finalised images will be saved to `/data/predictions/`.

### Further Work

Ideally we would have spent more time analysing the test images to ascertain exactly how to produce equivalent training data, but the sample size was so small that simple motion blur was likely optimal.

With more hardware capacity, it would be interesting to see the performance improvement given more layers + training data.
