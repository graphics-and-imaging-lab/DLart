
print('==> doing epoch on validation data:')
print("==> online epoch # " .. epoch)

--Get time
local time = sys.clock()

-- set the dropouts to evaluate mode
model:evaluate()

local loss_test = 0
local top1_test = 0
for t = 1, test_data:size(), opt.batchSize do
 xlua.progress(t,test_data:size())
 -- create mini batch
 local inputs = torch.CudaTensor(opt.batchSize, opt.channels, opt.imgSize[1], opt.imgSize[2])
 local targets = torch.CudaTensor(opt.batchSize)
 for i = t, math.min(t + opt.batchSize -1, test_data:size()) do
   local input = test_data:data(i)
   local target = class_cache[test_data:labels(i)]

   if opt.ttype == 'double' then input = input:double()
   elseif opt.ttype == 'cuda' then input = input:cuda() end

   --Test if the data is being loaded correctly
   assert(input, "Error in data with index ".. i)
   assert(target, "Error in label with index ".. i)

   inputs[{i - t + 1}] = input
   targets[{i - t + 1}] = target
 end

 local outputs = model:forward(inputs)
 local err = criterion:forward(outputs, targets)
 local pred = outputs:exp():float()

 loss_test = loss_test + err

 local _, pred_sorted = pred:sort(2, true)
 for i=1,pred:size(1) do
    local g = targets[i]
    if pred_sorted[i][1] == g then top1_test = top1_test + 1 end
 end
end

top1_test = top1_test * 100 / test_data:size()
loss_test = loss_test / (test_data:size()/opt.batchSize) -- because loss_test is calculated per batch
testLogger:add{
  ['% top1 accuracy (test set)'] = top1_test,
  ['avg loss_test (test set)'] = loss_test  }
testLogger:style{['% top1 accuracy (test set)'] = '+-'}
testLogger:style{['avg loss_test (test set)'] = '+-'}
testLogger:plot()

-- save/log current net
print('\n==> (mean) loss = ' .. loss_test)
print('==> (mean) top1(%) = ' .. top1_test .. '\n')
collectgarbage()
-- end of testing
