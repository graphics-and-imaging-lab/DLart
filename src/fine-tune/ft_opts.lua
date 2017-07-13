-- File to gather all the options together
local M = { }

function M.parse(arg)
  local cmd = torch.CmdLine()
  --------------------------------------------------------------------------------
  -- BASIC OPTIONS ---------------------------------------------------------------
  --------------------------------------------------------------------------------
  cmd:option('-learningRate', 1e-3, 'Learning rate')
  cmd:option('-weightDecay', 5e-4, 'Weight decay')
  cmd:option('-momentum', 0.9, 'Momentum')
  cmd:option('-batchSize', 10, 'Batch size')
  cmd:option('-channels', 3, 'Number of channels of the images')
  cmd:option('-imgSize', {224,224}, 'Image width and height {width, height}')
  cmd:option('-learningRateDecay', 1e-7, 'learning Rate Decay')
  cmd:option('-optimization', 'SGD', 'Optimization method (SGD, CG, LBFGS, ASGD, adam)')
  cmd:option('-ttype', 'cuda', 'Tensor type that is gonna be used (cuda, nn)')
  cmd:option('-model', 'vgg', 'Pretrained model to fine-tune (vgg, vgg_bn, vgg_classbn, retrain)')
  cmd:option('-retrain', 'path', 'Path to the pre-trained model')
  cmd:option('-plot', false, 'Show graphs while training')
  cmd:option('-nBatchs', 10000, 'Number of batchs per epoch')
  cmd:option('-save', '/test', 'Where to save the process [datapath .. save]')
  cmd:option('-nEpochs', 55, 'Number of epochs for training')
  cmd:option('-execute', 1, 'Either to train the net or to see its architecture')
  cmd:option('-numberEpoch', -1, 'Useful to start again a training')
  cmd:option('-dataPath', '/media/mlagunas/a0148b08-dc3a-4a39-aee5-d77ee690f196/TFG', "Path to the folder with the data (models, logs...)")

  cmd:text()
  local opt = cmd:parse(arg or {})
  return opt
end

return M
-- end of options
