# Training a clip-art image classifier in Torch-7 with VGG19

This code shows how to fine-tune the net VGG-19 so it can precisely predict clip-art images into 23 classes. We fine-tunned the network based on the assumption that the different between natural images (photographs) and illustrations (cliparts) rely on the low-level features of the images.

The classification is done using a Support Vector Machine (SVM) whose input is the second Fully-Connected layer (FC) of the fine-tunned VGG19.

## Required libraries and implementation

The following libraries have been used and are required to run the code:

- cutorch
- cudnn
- cunn
- hdf5
- matio


## Options
```lua
cmd:option('-option', "", '[classify] to classify images.
                           [path] to get the paths from a group of images.')

cmd:option('-net',         "trainf",  'CNN to use [trainf (optimized vgg19)|vgg19]')
cmd:option('-backend',     "cudnn",   '[cudnn|nn]')
cmd:option('-svm',         false,     '[true] to use the SVM to classify features')
cmd:option('-layer',       42,        'Layer of the network where we will extract the features')
cmd:option('-batch',       10,        'Size of the batch')

cmd:option('-dataset',     "",        'Name of the dataset we are using')
cmd:option('-path',        "",        'Path to the image paths')
cmd:option('print_output', false,     'Either to print the predictions output or to keep it clear')
cmd:option('-test',        "",        'Some string to make a difference between test and final results')
cmd:option('-out',         "",        'Name of the outputfile. Do not give any extension to it.'')
cmd:option('-keep',        false,     'Either to keep the temporary files or to remove them')
cmd:option('-nresults',    10,        'Number of predictions to show, sorted in decreasing order')

cmd:option('-image_paths', "",        'Path to the folder with the images to extract. The structure has to be folder/train/class/image')
cmd:option('-image_out',   "",        'Path to store the files where the images splitted in train, val and all together')
```

## Extract the paths

In order to extract the paths from each image of the dataset (assuming our dataset is located at `../data/curated/`) we have to run the following commands (NOTE that the ouput should be `*_train.txt` and `*_val.txt`):
```
th main.lua -option path -image_out ../data/paths/paths_train.txt -image_paths ../data/curated/train/

th main.lua -option path -image_out ../data/paths/paths_val.txt -image_paths ../data/curated/val/
```

## Classify Images

To classify a group of images we have to especify the dataset name, the network to use and the path to the file with the paths of the images in that dataset. Also we can add the flag `-svm` to classify it using the SVM. Without this flag the predicitions will be done using the CNN.

```
th main.lua -dataset ../data/curated -option classify -net trainf  -path ../data/paths/paths -svm

th main.lua -dataset ../data/curated -option classify -net vgg19  -path ../data/paths/paths
```

## Citation and details

Details about the method used can be found in the following paper: [paper](http://giga.cps.unizar.es/~mlagunas/downloads/ceig_2017.pdf).

If you find it useful for your research please cite:
```
@inproceedings {ceig.20171213,
    booktitle = {Spanish Computer Graphics Conference (CEIG)},
    title = {{Transfer Learning for Illustration Classification}},
    author = {Lagunas, Manuel and Garces, Elena},
    year = {2017},
    publisher = {The Eurographics Association},
    ISSN = {-},
    ISBN = {978-3-03868-046-8},
    DOI = {10.2312/ceig.20171213}
}
```

This software is published for academic and non-commercial use only.
