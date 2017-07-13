-- File to gather all the options together
local M = { }

function M.parse(arg)
  local cmd = torch.CmdLine()
  --------------------------------------------------------------------------------
  -- BASIC OPTIONS ---------------------------------------------------------------
  --------------------------------------------------------------------------------

  cmd:option('-option', "", '[classify] to classify images.\n[extract] to extract features of a group of images.\n[path] to get the paths from a group of images.')

  cmd:option('-net', "trainf", "CNN to use [trainf (optimized vgg19)|vgg19]")
  cmd:option('-backend', "cudnn", '[cudnn|nn]')
  cmd:option('-svm', false, '[true] to use the SVM to classify features')
  cmd:option('-layer', 42, 'Layer of the network where we will extract the features')
  cmd:option('-batch', 10, 'Size of the batch, in case we classify a group of images')

  cmd:option('-dataset', "", 'Data to feed the network')
  cmd:option('-path', "", 'Path to the image paths')
  cmd:option('print_output', false, "Either to print the predictions output or to keep it clear")
  cmd:option('-test', "", 'Some string to make a difference between test and final results')
  cmd:option('-out', "", "Name of the outputfile. Do not give any extension to it.")
  cmd:option('-keep', false, "Either to keep the temporary files or to remove them")
  cmd:option('-nresults', 10, "Number of predictions to show, sorted in decreasing order")
  -- cmd:option('-dataPath', '/media/mlagunas/a0148b08-dc3a-4a39-aee5-d77ee690f196/TFG', "Path to the dataset")
  -- cmd:option('-dataPath', '/media/mlagunas/a0148b08-dc3a-4a39-aee5-d77ee690f196', "Path the dataset")

  cmd:option('-image_paths', "", 'Path to the folder with the images to extract. The structure has to be folder/train/class/image and folder/val/class/image')
  cmd:option('-image_out', "", 'Path to store the files where the images splitted in train, val and all together')


  -- "Output file with the predictions: dataset_subdata_nettype_measure"

  cmd:text()
  local opt = cmd:parse(arg or {})

  -- Code configuration stuff
  opt.name = opt.dataset

  if opt.name == "" then
    print ("WARNING: You did not introduce any dataset.")
  end

  if opt.option == "" then
    print ("WARNING: You did not introduce any option to do.")
  end

  return opt
end

return M

-- end of options
