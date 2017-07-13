require 'loadcaffe';
require 'cudnn';
require 'cutorch' ;
require 'nn';
require 'cunn';
require 'optim';
require 'image';

path = 'fine-tune/'
upath = 'utils/'

function main()

  local opts = dofile(path ..'ft_opts.lua')
  opt = opts.parse(arg)

  dofile (upath .. "utils.lua");
  ----------------------------------------------------------------------

  print '==> defining some tools'
  opt.nBatchs = opt.batchSize * opt.nBatchs

  if opt.numberEpoch == -1 then epoch = 1
  else epoch = opt.numberEpoch print ("==> epoch number set to " .. epoch) end

  ----------------------------------------------------------------------
  print '==> loading network model...'
  if opt.retrain == 'path' then
    if opt.model == "vgg" then model = torch.load(opt.dataPath .. '/models/vgg_log_.t7')
    else if opt.model == "vgg_bn" then model = torch.load(opt.dataPath .. '/models/vgg_log_bn.t7')
    else if opt.model == "vgg_classbn" then model = torch.load(opt.dataPath .. '/models/vgg_log_classBN.t7')
      if not opt.retrain == '' then print '==> retraining model' model = torch.load(opt.retrain)
      else print "[ERROR] model not recognized" end end end end
    else
      print '==> loading pre-trained model'
      model = torch.load(opt.retrain)
    end

    -- Removing backpropagation on the top layers
    -- 20-36 VGG and VGG_classBN
    --[[for i=1, 18 do
    c = model:get(i)
    c.updateGradInput = function(self, inp, out) end
    c.accGradParameters = function(self,inp, out) end
    end]]

    ----------------------------------------------------------------------
    -- opt = {}
    -- opt.dataPath = "/media/mlagunas/a0148b08-dc3a-4a39-aee5-d77ee690f196/TFG"
    -- Retrieve cached files
    print '==> loading training, validation data and dataset caches...'
    train_data = torch.load(opt.dataPath .. '/datasets/noisy_train_n.t7')
    test_data = torch.load(opt.dataPath .. '/datasets/noisy_test_n.t7')
    class_cache = torch.load(opt.dataPath .. '/datasets/class_cache.t7')

    -- function test_data:data(idx)
    --   data = image.load(self.path[idx])
    --   img = torch.Tensor(3, data:size(2), data:size(3))
    --   img[1], img[2], img[3] = data[1], data[2], data[3]
    --
    --   if data:size(1) == 4 then
    --     return preprocess(img, data[4])
    --   else
    --     return preprocess(img, false)
    --   end
    -- end
    -- torch.save(opt.dataPath .. '/datasets/noisy_test_n.t7', test_data)
    -- Log results to files
    trainLogger = optim.Logger(paths.concat(opt.dataPath .. opt.save, 'train.log'))
    testLogger = optim.Logger(paths.concat(opt.dataPath .. opt.save, 'test.log'))

    ----------------------------------------------------------------------

    print "==> defining criterion and changing to cuda if needed"
    criterion = nn.ClassNLLCriterion()

    if (opt.ttype == "cuda") then
      model = model:cuda()
      criterion = criterion:cuda()
    end

    if opt.execute == 0 then print (model) end
    ----------------------------------------------------------------------

    print '==> configuring optimizer'

    if opt.optimization == 'CG' then
      optimState = {
        maxIter = opt.nEpochs
      }
      optimMethod = optim.cg

    elseif opt.optimization == 'LBFGS' then
      optimState = {
        learningRate = opt.learningRate,
        maxIter = opt.nEpochs,
        nCorrection = 10
      }
      optimMethod = optim.lbfgs

    elseif opt.optimization == 'SGD' then
      optimState = {
        learningRate = opt.learningRate,
        weightDecay = opt.weightDecay,
        momentum = opt.momentum,
        learningRateDecay = opt.learningRateDecay
      }
      optimMethod = optim.sgd
    elseif opt.optimization == "adam" then
      print "Using ADAM optimization"
      optimState = {
        learningRate = opt.learningRate,
        beta1 = 0.9,
        beta2 = 0.99,
      }
      optimMethod = optim.adam
    elseif opt.optimization == 'ASGD' then
      optimState = {
        eta0 = opt.learningRate,
        t0 = trsize * t0
      }
      optimMethod = optim.asgd

    else
      error('unknown optimization method')
    end

--------------------------------------------------------------------------------

    print "==> reseting layers and initializing weights with uniform distribution"

    if opt.model == "vgg" then linear_layers = {39, 42, 45}end
    if opt.model == "vgg_bn" then linear_layers = {54,58,62} end
    if opt.model == "vgg_classbn" then linear_layers = {39, 43, 47} end

    if opt.model == "vgg" or model == "vgg_classbn" then
      conv_layers = {1,3,6,8,11,13,15,17,20,22,24,26,29,31,33,35} end
      if opt.model == "vgg_bn" then conv_layers = {1,4,8,11,15,18,21,24,28,31,34,37,40,43,46,49} end

      lrs_model = model:clone()
      lrs = lrs_model:getParameters()
      lrs:fill(opt.learningRate) -- setting the base learning rate to opt.learningRate

      -- conv_layer index til all the weights and LR will be reset
      reset = 11

      -- set the learning rate factor of the bias
      -- reset weights of the layers we want to retrain
      for i = reset, #conv_layers do
        lrs_model:get(conv_layers[i]).bias:fill(1e-2)
        -- Set its weight random using a uniform distribution
        lrs_model:get(conv_layers[i]).weight:uniform()
      end

      for i = 1, #linear_layers do
        -- Set learning rate on linear layers of 1e-2
        lrs_model:get(linear_layers[i]).bias:fill(1e-2)
        -- Set its weight random using a uniform distribution
        lrs_model:get(linear_layers[i]).weight:uniform()
      end
      -- now pass lrs_model to optimState, which was created previously
      optimState.learningRates = lrs
      -- free the memory
      lrs_model = nil
      lrs = nil

      print '==> getting pretrained model parameters'
      parameters, gradParameters = model:getParameters()
      collectgarbage()
      if opt.execute == 1 then
        while (epoch <= opt.nEpochs) do
          -- TRAIN --
          dofile(path .. 'ft_train.lua')
          -- VALIDATE --
          dofile(path .. 'ft_test.lua')
          -- next epoch
          epoch = epoch + 1
        end
      end
    end

    main()
