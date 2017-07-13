local M = {}

---------------------------------------------------------------------------------------------
------------------------------- PRIVATE FUNCTIONS -------------------------------
---------------------------------------------------------------------------------------------

-- This is an auxiliar function used to format a synset in order to get
-- the position of all the different matches
function formatSynset(synset, mode)
  local sy = synset
  local res = {}
  if mode == 1 then
    for i = 1, #sy do
      res[i] = table.concat(sy[i]:lower():split(","),"")
    end
    return table.concat(res, ", ")
  end
  if mode == 2 then
    for i = 1, #sy do
      res[i] = table.concat(sy[i]:lower():split(" |,|'"),"-")
    end
    return res
  end
end

-- Auxiliar function that recieves an array and formats it. It returns
-- a string with the elements of the array ignoring the ones that are
-- stopwords, 1 char words or numbers.
function formatIClass(p)
  words = ""
  --remove stop-words and number values
  --TO-DO remove words with lenght < N
  for _,v in pairs(p) do
    if not v:match("%d+")
    and not contains(stop_words,v)
    and v:len()>1 then
      words = words .. v .. " "
    end
  end
  return words
end

-- This functions reads a HDF5 file and returns a table with its contect
function HDF5read(h5file)
  local myFile = hdf5.open(h5file, 'r')
  local prob = myFile:read('prediction'):all();
  myFile:close();
  return prob
end

---------------------------------------------------------------------------------------------
------------------------------- PUBLIC FUNCTIONS -------------------------------
---------------------------------------------------------------------------------------------

-- This function calculates the cross entropy error of a matrix of predictions (images x predictionsArray)
-- each line of the matrix is related with a line in the image_paths file. It iterates through each row
-- in this matrix and calculates the cross entropy error retrieving the probability which is in the position
-- where the net prediction has done a match, we have to calculate every match between the ground truth and
-- the net prediction.
function HDF5entropy(pathFile, h5probabilities, outputFile, nElements, classes, imagenet)
  --pathFile = pathp .. '.txt'
  -- classes = synset_words
  -- h5probabilities = predictions .. '.h5'

  local lines = fileToArray(pathFile, 0)
  local prob = HDF5read(h5probabilities):float()
  local synset = formatSynset(classes, 2)

  local entropy = {}
  local elements = {}
  local match1 = {}
  local match5 = {}
  local globalEntropy = 0

  for i = 1, #lines do
    xlua.progress(i, #lines)

    local path = ""
    -- Get the target class.
    if imagenet then
      bl,_ = contains(blacklist, tostring(i))
      path = classes_from_mat(groundtruth[i])
      path = path:split(" |,|'")
      path = table.concat(path,"-")
    else
      bl = false
      path = lines[i]:split("/") path = path[#path-1]
    end
    if not bl then
      -- Get match position
      local _,match = contains(synset,path)

      -- Calculate the cross-entropy for each element and adds it
      -- to the class that the image belongs.
      local add = math.log(prob[i][match])
      globalEntropy = globalEntropy + add
      if entropy[path] == nil then
        entropy[path] = add
        elements[path] = 1
      else
        entropy[path] = entropy[path] + add
        elements[path] = elements[path] + 1
      end
    end
  end

  for k,v in pairs(entropy) do
    entropy[k] = -entropy[k]/elements[k]
  end
  print ("writing ", outputFile)

  file = io.open(outputFile, "w")
  if imagenet then
    file:write("GLOBAL | " .. -globalEntropy/(#lines-#blacklist) .. " | " .. (#lines-#blacklist) .. '\n')
  else
    file:write("GLOBAL | " .. -globalEntropy/#lines .. " | " .. #lines .. '\n')
  end
  for k,v in spairs(entropy, function(t,a,b) return t[b] > t[a] end) do
    if(elements[k] > nElements) then
      file:write(k .. " | " .. entropy[k] .. " | " .. elements[k] .. '\n')
    end
  end
  file:close()
end

--
--
--
function getProb(h5probabilities, m,n, synset)
  local prob = HDF5read(h5probabilities)
  local pos = 0
  if m > 1 then
    prob, pos = prob[m]:view(-1):sort(true)
  else
    prob, pos = prob:view(-1):sort(true)
  end
  for i = 1, n do
    print (synset[pos[i]] .. "\t".. prob[i])
  end
end

--
--
--
function printResults(img_paths, predictions, groundt, class)
  local imgs = fileToArray(img_paths, 0)
  local prob = HDF5read(predictions)
  local auxgt = groundt
  for i = 1, #groundt do
    auxgt[i] = table.concat(groundt[i]:split(" |,|'"),"-"):lower()
  end
  _, synset_index = contains(auxgt, opt.name)
  for i = 1, prob:size()[1] do
    pred, pos = prob[i]:view(-1):sort(true)
    _, index = contains(torch.totable(pos), synset_index)
    aux_img = imgs[i]:split("/") aux_img = aux_img[#aux_img -1] .. "/".. aux_img[#aux_img]
    print (aux_img)
    print (groundt[pos[1]] .. " (1) \t".. pred[1])
    if index ~= 1 then
      print (groundt[pos[index]] .. " (" .. index .. ") \t".. pred[index])
    end
    print ("====")
  end
end

-- Predicts the probability for each class in the synset of each image
-- in the dataset
function HDF5prediction_1(path, outputFile, cnn, nclasses, exp)
  ok, img = pcall(image.load, path, 3)
  if not ok then
    print('[error] ' .. path)
    --os.execute("rm ".. lines[i])
  else
    img = image.scale(preprocess(image.load(path, 3)), 224, 224, 'bilinear'):cuda()
  end
  local res
  if exp then res = cnn:forward(img):exp()
  else res = cnn:forward(img) end
  res = res:float();

  --Writes the result in a hdf5 format file
  hdf5_file = hdf5.open(outputFile, 'w');
  hdf5_file:write('prediction', res);
  hdf5_file:close();

  return res
end

-- Predicts the probability for each class in the synset of each image
-- in the dataset
function HDF5prediction(pathFile, cnn, outputFile, nclasses, size, exp)

  --pathFile = 'data/paths/curated/curated_paths_test.txt'
  --predictions = dataPath ..'/h5/curated/vgg19/Pcurated_test_probabilities.h5'
  --nclasses = 1000
  --size = 5
  print ("Classifying dataset...")
  print (pathFile)
  lines = fileToArray(pathFile, 0)
  probabilityResult = torch.FloatTensor(#lines,nclasses)
  mini_batch = torch.CudaTensor(size,3,224,224)

  -- Iterate through the dataset obtaining the probabilities array for each class
  i = 1
  while i < #lines do
    xlua.progress(i,#lines)
    --j = 1
    for j = 1, size do
      --print ((i+j-1))
      if (i+j-1) <= #lines then ok, img = pcall(image.load, lines[i+j-1], 3) end
      if not ok then
        print('[error] ' .. lines[i+j-1])
        --os.execute("rm ".. lines[i])
      else
        if (i+j-1)<= #lines then mini_batch[j] = image.scale(preprocess(image.load(lines[i+j-1],3)), 224, 224, 'bilinear'):cuda() end
      end
    end
    local res
    if exp then res = cnn:forward(mini_batch):exp()
    else res = cnn:forward(mini_batch) end
    res = res:float();
    for j = 1, size do
      if (i+j-1) <= #lines then probabilityResult[i+j-1] = res[j]
      else print ('ended at ' .. j) break end
    end
    res = nil
    collectgarbage()
    i = i + size
  end

  --Writes the result in a hdf5 format file
  print ("Writing file...")
  print(outputFile)
  hdf5_file = hdf5.open(outputFile, 'w');
  hdf5_file:write('prediction', probabilityResult);
  hdf5_file:close();
end

--
--
function HDF5features(pathFile, cnn, outputFile, size, layer)
  -- print ("Extracting features to one matrix from layer ", layer)
  -- pathFile = '/home/mlagunas/Bproject/DLart/data/paths/curated/curated_paths_train.txt'
  print (pathFile)
  lines = fileToArray(pathFile, 0)
  --size = 2
  --layer = 42
  --Create a tensor to store data (images x features)
  local featureResult = torch.DoubleTensor(#lines, 4096)
  local mini_batch = torch.CudaTensor(size,3,224,224)

  -- Iterate through the dataset obtaining the probabilities array for each class
  i = 1
  while i < #lines do
    if (i + size) > #lines then
      size = #lines - i + 1
    end

    for j = 1, size do
      xlua.progress(i+j-1, #lines)
      ok, data = pcall(image.load, lines[i+j-1])
      if not ok then
        print('[error] ' .. lines[i+j-1])
      else
        if (i+j-1)<= #lines then
          data = image.load(lines[i+j-1])
          img = torch.Tensor(3, data:size(2), data:size(3))

          if data:size(1) == 1 then
            for w = 1, 3 do
              img[w] = data[1]
            end
          else
            for w = 1, 3 do
              img[w] = data[w]
            end
          end

          if data:size(1) == 4 then
            img = preprocess(img, data[4])
          else
            img = preprocess(img, false)
          end

          mini_batch[j] = image.scale(img, 224, 224, 'bilinear'):cuda()
        end
      end
    end
    cnn:forward(mini_batch);
    feat = cnn.modules[layer].output:float();

    for j = 1, size do
      if (i+j-1)<=#lines then featureResult[i+j-1] = feat[j]
      else print ('ended at ' .. j) break end
    end

    feat = nil
    collectgarbage()
    i = i + size
  end

  --Writes the result in a hdf5 format file
  hdf5_file = hdf5.open(outputFile, 'w');
  hdf5_file:write('features', featureResult);
  hdf5_file:close();

  featureResult = nil
  mini_batch = nil
  lines = nil
  collectgarbage()
end

function HDF5features_1(path, outputFile, cnn, layer)
  ok, img = pcall(image.load, path, 3)
  if not ok then
    print('[error] ' .. path)
  else
    img = image.scale(preprocess(image.load(path)), 224, 224, 'bilinear'):cuda()
  end

  cnn:forward(img);
  feat = cnn.modules[layer].output:float();

  hdf5_file = hdf5.open(outputFile, 'w');
  hdf5_file:write('features', feat);
  hdf5_file:close();
  collectgarbage()
end

-- This function calculate the average precision in N for each
-- class in synset_words. It returns another file with all the
-- probabilities calculated.
function HDF5topN(pathFile, h5probabilities, outputFile, n, classes, imagenet)
  local lines = fileToArray(pathFile, 0)
  local prob = HDF5read(h5probabilities)
  local ap_global = 0
  local ap_class = {}
  local elements = {}
  for i = 1, #lines do
    xlua.progress(i, #lines)

    --Get position and result of the prediction
    local _,pos = prob[i]:view(-1):sort(true)
    local target = ""

    -- Get the target class.
    if imagenet then
      bl,_ = contains(blacklist, tostring(i))
      target = classes_from_mat(groundtruth[i])
      target = target:split(" |,|'")
      target = table.concat(target,"-"):lower()
    else
      bl = false
      target = lines[i]:split("/") target = target[#target-1]:lower()
    end

    if not bl then
      -- get the first n predictions and format it as the folders name
      local pred = {}
      for j = 1, n do
        pred[j] = table.concat(classes[pos[j]]:split(" |,|'"),"-"):lower()
      end

      -- if target == "velvet" then print (pred, target) end
      --
      -- if i == 10 then
      -- do return end
      -- end

      --Add 1 to the correct predictions
      if contains(pred,target) then
        ap_global = ap_global+1
        if ap_class[target] == nil then ap_class[target] = 1
        else ap_class[target] = ap_class[target] + 1 end
      else
        if ap_class[target] == nil then ap_class[target] = 0 end
      end

      -- Count elements in each class
      if elements[target] == nil then elements[target] = 1
      else elements[target] = elements[target] + 1 end
    end
  end

  -- Export the results as a readable txt file
  print ("writing ", outputFile)
  local file = io.open(outputFile, "w")
  file:write("IMAGE | AP | PREDICTED | ALL\n")
  if imagenet then
    file:write("global | " ..ap_global/(#lines-#blacklist) .. " | " .. ap_global .. " | " .. (#lines-#blacklist).. '\n')
  else
    file:write("global | " ..ap_global/#lines .. " | " .. ap_global .. " | " .. #lines.. '\n')
  end
  local map = 0
  local classes = 0
  for k,v in spairs(ap_class, function(t,a,b) return t[b]/elements[b] < t[a]/elements[a] end) do
    if(elements[k] > 3) then -- Take only classes with more than 3 elemnts
      map = map + ap_class[k]/elements[k]
      classes = classes + 1
      file:write(k .. " | " .. ap_class[k]/elements[k] .. " | " .. ap_class[k] .. " | " .. elements[k] .. '\n')
    end
  end
  file:write("mAP | " .. map/classes .. " | " .. map .. " | " .. classes .. '\n')
  file:close()
end

M.HDF5topN = HDF5topN
M.HDF5entropy = HDF5entropy
M.HDF5prediction = HDF5prediction
M.HDF5prediction_1 = HDF5prediction_1
M.HDF5features = HDF5features
M.HDF5read = HDF5read
M.getProb = getProb

return M
