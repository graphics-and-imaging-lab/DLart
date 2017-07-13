
require 'cutorch';
require 'cudnn';
require 'cunn';

--------------------------------------------------------------------------------
-- MAIN ------------------------------------------------------------------------
--------------------------------------------------------------------------------
local function main()

  -- read the arguments file
  local opts = dofile('opts.lua')
  opt = opts.parse(arg)
  opt.name = opt.dataset:split("/")
  opt.name = opt.name[#opt.name]
  paths = opt.path
  print(opt)

  dofile ('utils/utils.lua');

  -- Load the network regarding the backend we are gonna use
  print ("========> Loading")
  -- load the model
  if opt.net == "vgg19" then
    require 'loadcaffe';
    opt.exp = false
    cnn = loadcaffe.load(proto_file, model_file, 'cudnn');
    print ("========> Using optimized VGG19")

  end
  if opt.net == "trainf" then
    opt.exp = true
    cnn = torch.load("../models/testmodel-55.t7");
    print ("========> Using optimized VGG19")
  end
  cnn:cuda();
  cnn:evaluate();

  -- Some configuration stuff
  use_imagenet, single_image, directory = false, false, false
  if not (opt.out == "") then
    out = opt.out
  end

  -- Get the image path and name if it is not a previously used dataset
  if opt.name == "imagenet" then
    use_imagenet = true
  else
    if isDir(opt.dataset) then directory = true
    else single_image = true end
    path_name = opt.dataset
    opt.name = path_name:split("/") opt.name = opt.name[#opt.name]
  end

  -- Load the groundtruth
  if opt.svm then groundt = synset_curated
  else groundt = synset_words end
  nclasses = #groundt

  -- Extract features option
  if opt.option == "extract" then
    if (opt.out == "") then
      out = '../data/h5/' .. opt.dataset .. '_features_' .. opt.net ..'_' .. opt.name.. opt.test ..'_' .. opt.net ..'_'
    end
    paths = '../data/paths/' .. opt.dataset .. '_' .. opt.name .. '_paths'
    getFeatures(paths, cnn, out, opt.batch, opt.layer, use_imagenet)
  end

  -- Clasification option
  if opt.option == "classify" then
    -- -- Give a name to the file with the prediction results
    -- predictions = "temp/predictions_"..opt.name..".h5"
    -- -- Give a name to the file with the features
    -- features = "temp/features_"..opt.name..".h5"
    --   if directory then
    --     -- Create a temporary folder
    --     if not isDir("temp") then
    --       os.execute("mkdir temp")
    --     end
    --
    --     -- Get the image paths from the given dataset
    --     -- print ("========> Getting image paths")
    --     -- img_paths = getImageList(path_name, false)
    --     -- arrayToFile(img_paths, "temp/paths.txt")
    --
    --     -- Call the SVM if neccsary
    --     if opt.svm then
    --     print ("========> Extracting features (layer " .. opt.layer .. ")")
    --     HDF5features(opt.path, cnn, features, opt.batch, opt.layer)
    --     print ("========> Classifying the features (SVM)")
    --     os.execute("python SVM/SVM_test.py " ..
    --       "notused " ..
    --       opt.net .. " " ..
    --       opt.name .. " " ..
    --       opt.dataset .. " " ..
    --       features .. " " ..
    --       predictions .. " " ..
    --     1)
    --     if opt.print_output then
    --       printResults("temp/paths.txt", predictions, groundt, opt.name)
    --     end
    --   else
    --     print ("========> Classifying the images contained in the folder ".. opt.name)
    --     HDF5prediction("temp/paths.txt", cnn, predictions, nclasses, opt.batch, opt.exp)
    --     if opt.print_output then
    --       printResults("temp/paths.txt", predictions, groundt, opt.name)
    --     end
    --   end
    --
    --   -- If necessary remove the temporary files
    --   if opt.keep then
    --     -- os.execute("rm ".. predictions)
    --     os.execute("rm -r temp")
    --   end
    --
    --   -- The image name typed by the user is a single image, not a dataset
    --   if single_image then
    --     print ("========> Clasifying image")
    --
    --     -- Create a temporary folder
    --     if not isDir("temp") then
    --       os.execute("mkdir temp")
    --     end
    --
    --     if opt.svm then
    --       -- Calculate features if needed
    --       if not isFile(features) then
    --         print ("========> Creating features file")
    --         HDF5features_1(path_name, features, cnn, opt.layer)
    --       else
    --         print ("========> Features file already created")
    --       end
    --
    --       os.execute("python SVM/SVM_test.py " ..
    --         "notinuse" .. " " ..
    --         opt.net .. " " ..
    --         opt.name .. " " ..
    --         opt.dataset .. " " ..
    --         features .. " " ..
    --         predictions .. " " ..
    --       0)
    --     else
    --       -- Calculate predictions if needed
    --       if not isFile(predictions) then
    --         print ("========> Creating predictions file")
    --         print (path_name)
    --         HDF5prediction_1(path_name, predictions, cnn, nclasses, opt.exp)
    --       else
    --         print ("========> Predictions file already created")
    --       end
    --     end
    --
    --     -- Write the 10 first predictions obtained by the CNN
    --     print ("====================================")
    --     getProb(predictions, 1, opt.nresults, groundt)
    --     print ("====================================")
    --
    --     -- If necessary remove the temporary files
    --     if opt.keep then
    --       -- os.execute("rm ".. predictions)
    --       os.execute("rm -r temp")
    --     end
    --   end
    -- end

    -- build the predictions file, it has to be extracted before
    if opt.svm then
      predictions = '../data/h5/' ..'_svm_' .. opt.net .. '_' .. opt.name .. opt.test .. '_' .. opt.net .. '_probabilities'
    else
      predictions = '../data/h5/' ..'_' .. opt.net .. '_' .. opt.name .. opt.test.. '_' ..opt.net .. '_probabilities'
    end

    -- If the images have not been predicted yet we predict them
    print (opt.svm)
    if not opt.svm then
      -- if not isFile(predictions .. '.h5') and not use_imagenet then
      -- HDF5prediction(paths ..'.txt', cnn, predictions .. '.h5', nclasses, opt.batch, opt.exp);
      -- end

      if not isFile(predictions ..'_val.h5') then -- We only evaluate the test dataset of ImageNet
        print ("========> Classifying data val")
        HDF5prediction(paths .. '_val.txt', cnn, predictions .. '_val.h5', nclasses, opt.batch, opt.exp);
      end
      if not isFile(predictions ..'_train.h5') and not use_imagenet then
        print ("========> Classifying data train")
        HDF5prediction(paths .. '_train.txt', cnn, predictions .. '_train.h5', nclasses, opt.batch, opt.exp);
      end
    else
      features = '../data/h5/' ..'_svm_' .. opt.net .. '_' .. opt.name .. opt.test.. '_' ..opt.net .. '_features'
    print ("========> Extracting features (layer " .. opt.layer .. ")")
    if not isFile(features ..'_val.h5') then
      HDF5features(paths .. '_val.txt', cnn, features .. '_val.h5', opt.batch, opt.layer)
    end
    if not isFile(predictions ..'_val.h5') then
      os.execute("python SVM/SVM_test.py " ..
        "notused " ..
        opt.net .. " " ..
        opt.name .. " " ..
        opt.dataset .. " " ..
        features .. '_val.h5' .. " " ..
        predictions .. '_val.h5' .. " " ..
      1)
    end
    if not isFile(features ..'_train.h5') and not use_imagenet then
      HDF5features(paths .. '_train.txt', cnn, features.. '_train.h5', opt.batch, opt.layer)
    end
    if not isFile(predictions ..'_train.h5') then
      os.execute("python SVM/SVM_test.py " ..
        "notused " ..
        opt.net .. " " ..
        opt.name .. " " ..
        opt.dataset .. " " ..
        features.. '_train.h5' .. " " ..
        predictions.. '_train.h5' .. " " ..
      1)
    end
  end
  -- print ("========> Classifying the features (SVM)")

  -- After the prediction we get the array and evaluate them getting some statistical results as error or predictions
  -- which is going to be saved in a results folder
  -- if (opt.out == "") then
  -- if svm then out = '../data/results/' .. opt.dataset .. "/svm/" .. opt.net .."/" .. opt.name .. opt.test.. "_all_" .. opt.net .. "_"
  -- else out = "../data/results/" .. opt.dataset .. "/" .. opt.net .."/" .. opt.name.. opt.test .. "_all_" .. opt.net .. "_" end
  -- end
  -- if not use_imagenet then
  -- getMeasures(paths ..'.txt', predictions .. '.h5', out, groundt, use_imagenet);
  -- end

  --VAL
  print "inin"

  if (opt.out == "") then
    if opt.svm then out = '../data/results/' .. "_svm_" .. opt.net .."_" .. opt.name.. opt.test .. "_test_" .. opt.net .. "_"
    else out = '../data/results/' .. "_" .. opt.net .."_" .. opt.name.. opt.test .. "_test_" .. opt.net .. "_" end
  end
  getMeasures(paths .. '_val.txt', predictions .. '_val.h5', out, groundt, use_imagenet);

  --TRAIN
  if (opt.out == "") then
    if opt.svm then out = '../data/results/' .. "_svm_" .. opt.net .. "_" .. opt.name.. opt.test .. "_train_" .. opt.net .. "_"
    else out = '../data/results/' .. "_" .. opt.net .. "_" .. opt.name.. opt.test .. "_train_" .. opt.net .. "_" end
  end
  if not use_imagenet then
    getMeasures(paths .. '_train.txt', predictions .. '_train.h5', out, groundt, use_imagenet);
  end
end


-- Option to extract the paths of a given dataset
if opt.option == "path" then

  if opt.image_out == "" then
    print ("WARNING: You did not introduce any output path.")
  end

  l = getImageList(opt.image_paths, false)
  arrayToFile(l, opt.image_out)
end

-- Clean variables from memory
cnn = nil
collectgarbage();
end

main()
