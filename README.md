# Training a clip-art image classifier in Torch-7 with VGG19

This code shows how to fine-tune the net VGG-19 so it can precisely predict clip-art images into 23 classes. We fine-tunned the network based on the assumption that the different between natural images (photographs) and illustrations (cliparts) rely on the low-level features of the images.

The classification is done using a Support Vector Machine (SVM) whose input is the second Fully-Connected layer (FC) of the fine-tunned VGG19.

## Required libraries

- cutorch
- cudnn
- cunn
- hdf5
- matio

## Options

    cmd:option('-option', "", '[classify] to classify images.
                               [path] to get the paths from a group of images.')

    cmd:option('-net',         "trainf",  'CNN to use [trainf (optimized vgg19)|vgg19]')
    cmd:option('-backend',     "cudnn",   '[cudnn|nn]')
    cmd:option('-svm',         false,     '[true] to use the SVM to classify features')
    cmd:option('-layer',       42,        'Layer of the network where we will extract the features')
    cmd:option('-batch',       10,        'Size of the batch')

    cmd:option('-dataset',     "",        'Data to feed the network')
    cmd:option('-path',        "",        'Path to the image paths')
    cmd:option('print_output', false,     'Either to print the predictions output or to keep it clear')
    cmd:option('-test',        "",        'Some string to make a difference between test and final results')
    cmd:option('-out',         "",        'Name of the outputfile. Do not give any extension to it.'')
    cmd:option('-keep',        false,     'Either to keep the temporary files or to remove them')
    cmd:option('-nresults',    10,        'Number of predictions to show, sorted in decreasing order')

    cmd:option('-image_paths', "",        'Path to the folder with the images to extract. The structure has to be folder/train/class/image')
    cmd:option('-image_out',   "",        'Path to store the files where the images splitted in train, val and all together')

## Extract the paths

In order to extract the paths from each image of the dataset (assuming our dataset is located at `../data/curated/`) we have to run the following commands (NOTE that the ouput should be `*_train.txt` and `*_val.txt`):

    th main.lua -option path -image_out ../data/paths/paths_train.txt -image_paths ../data/curated/train/

    th main.lua -option path -image_out ../data/paths/paths_val.txt -image_paths ../data/curated/val/

## Classify Images
