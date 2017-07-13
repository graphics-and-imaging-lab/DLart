-- local vars
local time = sys.clock()

-- set model to training mode (for modules that differ in training and testing, like Dropout)
model:training();

-- shuffle at each epoch
shuffle = torch.randperm(train_data:size())

-- do one epoch
print('==> doing epoch on training data:')
print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

local top1_epoch = 0
local loss_epoch = 0

for t = 1, opt.nBatchs, opt.batchSize do
  -- create mini batch
  local inputs = torch.CudaTensor(opt.batchSize, opt.channels, opt.imgSize[1], opt.imgSize[2])
  local targets = torch.CudaTensor(opt.batchSize)
  for i = t, math.min(t + opt.batchSize - 1, train_data:size()) do
    -- Get image and groundtruth randomly
    randi = shuffle[i]
    local input = train_data:data(randi)
    local target = class_cache[train_data:labels(randi)]

    if opt.ttype == 'double' then input = input:double()
    elseif opt.ttype == 'cuda' then input = input:cuda() end

    --Test if the data is being loaded correctly
    assert(input, "Error in data with index ".. randi)
    assert(target, "Error in label with index ".. randi)

    inputs[{i - t + 1}] = input
    targets[{i - t + 1}] = target
  end

  local regimes = {
    -- start, end, LR, WD,
    { 1, 17, 5e-2, 5e-4 },
    { 17, 36, 1e-2, 5e-4 },
    { 36, 55, 5e-3, 5e-5 },
    { 55, 85, 1e-3, 5e-6 },
    { 85, 120, 5e-4, 5e-7 },
    { 120, 160, 1e-4, 5e-7 },
    { 160, 1e8, 5e-5, 5e-7 },

  }
  offset = 1e-1
  for _, row in ipairs(regimes) do
    if epoch >= row[1] and epoch <= row[2] then
      params= { learningRate=row[3]*offset, weightDecay=row[4], nesterov=opt.nesterov }, epoch == row[1]
    end
  end

  optimState={
      learningRate=params.learningRate,
      weightDecay = params.weightDecay,
  }

  local eval = function(X)
    if X ~= parameters then
      parameters:copy(X)
    end

    -- Reset gradients
    gradParameters:zero()

    -- evaluate function for complete mini batch
    output = model:forward(inputs)
    err = criterion:forward(output, targets)
    -- Estimate error
    local df_do = criterion:backward(output, targets)
    model:backward(inputs, df_do)

    loss_epoch = loss_epoch + err
    -- normalize gradients and f(X)
    gradParameters:div(inputs:size(1))

    return err, gradParameters
  end

  -- optimize on current mini-batch
  if optimMethod == optim.asgd then
    _,_,average = optimMethod(eval, parameters, optimState)
  else
    optimMethod(eval, parameters, optimState)
  end

  local top1 = 0
  do
    -- Get the output and change log probabilities to percentages
    output = output:exp()
    -- Sort the prediction in descending order
    local _,prediction_sorted = output:float():sort(2, true)
    for i = 1, opt.batchSize do
      if prediction_sorted[i][1] == targets[i] then
        top1_epoch = top1_epoch + 1;
        top1 = top1 + 1
      end
    end
    top1 = top1 * 100 / opt.batchSize;
  end

  print(('Epoch: [%d][%d/%d]\tLoss %.4f\tTop1-%%: %.2f')
    :format(epoch, math.floor(t/opt.batchSize)+1, math.floor(opt.nBatchs/opt.batchSize), err, top1))
end

-- time taken
time = sys.clock() - time
print("\n==> time to learn 1 epoch: " .. time )
time = time / train_data:size()
print ("==> time to learn 1 sample: " .. (time*1000) .. 'ms')

top1_epoch = top1_epoch/opt.nBatchs*100
loss_epoch = loss_epoch/opt.nBatchs*opt.batchSize

-- Save the logs of the training and plot them
trainLogger:add{
  ['% top1 accuracy (train set)'] = top1_epoch,
  ['avg loss (train set)'] = loss_epoch}
trainLogger:style{['% top1 accuracy (train set)'] = '+-'}
trainLogger:style{['avg loss (train set)'] = '+-'}
trainLogger:plot()

-- save/log current net
print('==> (mean) loss = ' .. loss_epoch)
print('==> (mean) top1(%) = ' .. top1_epoch)

model:clearState()
collectgarbage()

print('==> saving model to '..opt.dataPath..opt.save .. '\n')
torch.save(opt.dataPath..opt.save .. 'model-' .. epoch .. '.t7', model)

-- end of traning
