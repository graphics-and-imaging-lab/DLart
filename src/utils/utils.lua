local M = {}

upath = "utils/"
if upath == "utils/" then offset = "../"
else offset = "" end

-- dataPath='/media/mlagunas/a0148b08-dc3a-4a39-aee5-d77ee690f196'

-- Load all the global variables needed to execute the functions provided by this application
proto_file = '../models/VGG_ILSVRC_19_layers_deploy.prototxt'
model_file = '../models/VGG_ILSVRC_19_layers.caffemodel'
image_size = 224

require_path = {upath .. "FILEutils.lua", upath .. "/HDF5utils.lua"}
dofile (require_path[1]);
dofile (require_path[2]);
require 'image';

matio = require 'matio'
-- words : CharTensor - size: 1x9
-- gloss : CharTensor - size: 1x156
-- num_children : DoubleTensor - size: 1x1
-- ILSVRC2014_ID : DoubleTensor - size: 1x1
-- wordnet_height : DoubleTensor - size: 1x1
-- children : DoubleTensor - empty
-- num_train_images : DoubleTensor - size: 1x1
-- WNID : CharTensor - size: 1x9
ILSVRC_mat = matio.load(offset .. 'data/data_utils/meta_clsloc.mat', 'synsets')[1]
groundtruth = fileToArray(offset .. 'data/data_utils/ILSVRC2014_clsloc_validation_ground_truth.txt', 0)
blacklist = fileToArray(offset .. 'data/data_utils/ILSVRC2014_clsloc_validation_blacklist.txt', 0)

synset = offset .. 'data/data_utils/synset_words.txt'
synset_words = fileToArray(synset ,11)

swords = offset .. 'data/data_utils/stop_words.txt'
stop_words = fileToArray(swords,0)

curated = offset .. 'data/data_utils/synset_curated.txt'
synset_curated = fileToArray(curated,0)

require 'hdf5'

-- This functions maps an ILSVRC_2014ID to its class to be easily accesed with the file synset_words
--
function classes_from_mat(pos)
  name = ILSVRC_mat[tonumber(pos)]["words"]
  i,j = 1,1
  res = {}
  while i <= name:size()[1] do
    string = ""
    while j <= name:size(2) do
      s = string.format("%c", name[i][j])
      string = string .. s
      j = j+1
    end
    res[i] = string
    i = i+1
    j = 1
  end
  return res[1]
end

function ssd(t1, t2)
  t1 = t1:double();
  t2 = t2:double();
  -- element-wise t1-t2
  x = torch.Tensor.csub(t1,t2)
  -- absolute value of Tensor X
  x:abs();
  -- element-wise sqrt of X
  x:sqrt();
  -- sum up all the elements in X
  return torch.sum(x)
end

function getFeatures(path, cnn, out, batch, layer, use_imagenet)
  if not use_imagenet then
    print ("Extracting training...")
    HDF5features(path .. '_train.txt', cnn, out .. 'train_42.h5', opt.batch, opt.layer)
    print ("Extracting all...")
    HDF5features(path .. '.txt',cnn, out ..'42.h5', opt.batch,opt.layer )
  end
  print ("Extracting test...")
  HDF5features(path .. '_test.txt',cnn, out .. 'test_42.h5', opt.batch, opt.layer)

end

function getMeasures(path, predictions, out, synset, imagenet)
  print ("writing measures", out)
  HDF5topN(path, predictions, out.."top1.txt", 1, synset, imagenet);
  HDF5topN(path, predictions, out.."top5.txt", 5, synset, imagenet);
  HDF5entropy(path, predictions, out.."ce.txt", 1, synset, imagenet);
end

function datasetToFile(path, out, format)
  arrayToFile(getImageList(path, format), out .. '.txt');
  arrayToFile(getImageList(path.. '/train', format), out .. '_train.txt');
  arrayToFile(getImageList(path.. '/val', format), out .. '_test.txt');
end

--It removes all the elements that appears twice or more inside a table
function deleteDuplicate(array)
  local hash = {}
  local res = {}
  for _,v in ipairs(array) do
    if (not hash[v]) then
      res[#res+1] = v -- you could print here instead of saving to result table if you wanted
      hash[v] = true
    end
  end
  return res
end

function arrayToFile(array, outputFile)
  file = io.open(outputFile, 'w')
  for _,x in pairs(array) do
    file:write(x..'\n')
  end
  file:close()
end

--Split a String using a given sep, it returns
--an array with the subStrings of the Split
--function
function string:split(sep)
  local sep, fields = sep or ":", {}
  local pattern = string.format("([^%s]+)", sep)
  self:gsub(pattern, function(c) fields[#fields+1] = c end)
  return fields
end

-- Preprocess an image before passing it to a Caffe model.
-- We need to rescale from [0, 1] to [0, 255], convert from RGB to BGR,
-- and subtract the mean pixel.
function preprocess(img, A)

  -- Substract the mean_pixel value
  mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):mul(256.0)
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img = img:add(-1, mean_pixel)
  img = img:clamp(0, 255)

  -- Return the img in case it has three channels
  if A then
    -- Make the "transparent" background white
    w = torch.ne(A, torch.ones(#A)):double()*255.0
    img[1] = img[1]:add(w)
    img[2] = img[2]:add(w)
    img[3] = img[3]:add(w)
  end

  return image.scale(img, 224, 224, 'bilinear')
end

function visualizeWeight(weight, rescale)
  w = weight:permute(4,3,2,1)
  w = w:float()
  res = torch.Tensor(w:size(1), w:size(2), w:size(3)*rescale, w:size(4)*rescale)
  for i = 1, res:size(1) do
    res[i] = image.scale(w[i], w:size(3)*rescale, w:size(4)*rescale, 'simple')
  end
  itorch.image(res)
end

function subrange(t, first, last)
  local sub = {}
  for i=first,last do
    sub[#sub + 1] = t[i]
  end
  return sub
end

-- Returns an array with all the elements that contain totally or partially
-- item
function getContains(array, item)
  res = {}
  for _,v in pairs(array) do
    --Split the class given by array and change all its elements to lower case
    class = v:split(" |,|'")
    for k,x in pairs(class) do
      class[k] = x:lower()
    end
    --Split the item name so we get the given classes from a full path
    -- the given classes will be in an array
    words = item:split("/")
    file = words[#words]
    words = words[table.getn(words)]
    words = words:split(".")
    words = words[1]:split("-")
    -- Loop through the given classes checking which ones are contained
    -- in the predicted classes, if contained then we add it to the result
    for i = 1, #words do
      if (not words[i]:match("%d+")
        and not utils.contains(stop_words,words[i])
        and utils.contains(class,words[i]:lower())) then
        table.insert(res,table.concat(class,"-"))
        break;
      end
    end
  end
  return res, file
end

-- True if item is an element which is inside table
function contains(array, item)
  for i = 1, #array do
    if array[i] == item then
      return true, i
    end
  end
  return false , -1
end

-- Given a table: synset and a string: sep it returns another table with each
-- element splitted using sep.
function split(synset, sep)
  res = {}
  pos = {}
  for i = 1, #synset do
    words = synset[i]:split(sep)
    for j = 1, #words do
      table.insert(pos,i)
      table.insert(res,words[j])
    end
  end
  return res, pos
end

-- It returns an array with the positions of all the
-- matches that words has made in synset.
-- words format: words1 word2 word3 ...
-- synset format: class1 class1, class2 class2 class2,..
function getMatchPosition (iclass, nclass)
  iclass = iclass:lower()
  nclass = nclass:lower()
  m1 = iclass:split(" ")
  res = {}
  for i, v in ipairs(trim(nclass:split(","))) do
    for j=1, #m1 do
      bool, x = contains(v:split(" |'"),m1[j]:lower())
      if bool then
        table.insert(res,i)
        break;
      end
    end
  end
  return res
end

-- It recieves an string and removes the \t it has at the beginning or at the end
function trim(s)
  if (type(s) == "table") then
    for i = 1, table.getn(s) do
      s[i] = trim(s[i])
    end
    return s
  else
    local from = s:match"^%s*()"
    return from > #s and "" or s:match(".*%S", from)
  end
end

function spairs(t, order)
  -- collect the keys
  local keys = {}
  for k in pairs(t) do keys[#keys+1] = k end

  -- if order function given, sort by it by passing the table and keys a, b,
  -- otherwise just sort the keys
  if order then
    table.sort(keys, function(a,b) return order(t, a, b) end)
  else
    table.sort(keys)
  end

  -- return the iterator function
  local i = 0
  return function()
    i = i + 1
    if keys[i] then
      return keys[i], t[keys[i]]
    end
  end
end

M.preprocess = preprocess
M.spairs = spairs
M.trim = trim
M.contains = contains
M.deleteDuplicate = deleteDuplicate
M.split = split
M.getContains = getContains
M.getMatchPosition = getMatchPosition
M.subrange = subrange
M.arrayToFile = arrayToFile
M.visualizeWeight = visualizeWeight
M.classes_from_mat = classes_from_mat

return M
