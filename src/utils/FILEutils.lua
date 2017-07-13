local M = {}

--Open directory look for files, save data in p. By giving '-type f' as parameter, it returns all files.
--Loop through all files
--WARNING: root directories might have lots of files
function dirLookup(dir, escapeSeq)
  local stop_words = fileToArray('data/stop_words.txt',0)
  local pr = io.popen('find "'..dir..'" -type f')
  paths = {}
  class = {}
  for file in pr:lines() do
    if (file:find("[.jpg][.png][.jpeg]") ~= nil) then
      words = file:split("/")
      words = words[table.getn(words)]
      words = words:split(".")
      words = words[1]:split("-")
      for i = 1, table.getn(words) do
        if(not (words[i]:match("%d+"))
        and not (contains(stop_words,words[i]))) then
          table.insert (class, words[i]:lower())
          if(escapeSeq) then
            p = file
            p = p:split(" ")
            p = table.concat(p, "\\ ")
            p = p:split("(")
            p = table.concat(p,"\\(")
            p = p:split(")")
            p = table.concat(p,"\\)")
            p = p:split("&")
            p = table.concat(p,"\\&")
            p = p:split("'")
            p = table.concat(p,"\\'")
            table.insert (paths, p)
          else
            table.insert (paths,file)
          end
        end
      end
    end
  end
  return paths, class;
end

function formatPath(input)
  p = input
  p = p:split(" ")
  p = table.concat(p, "\\ ")
  p = p:split("(")
  p = table.concat(p,"\\(")
  p = p:split(")")
  p = table.concat(p,"\\)")
  p = p:split("&")
  p = table.concat(p,"\\&")
  p = p:split("'")
  p = table.concat(p,"\\'")
  return p
end

function getImageList(dir, format)
  p = io.popen('find '.. dir ..' -type f')
  a = {}
  for l in p:lines() do
    if (l:find('.jpg') ~= nil or
        l:find('.png') ~= nil or
        l:find('.jpeg') ~= nil or
        l:find('.JPEG') ~= nil ) then
      if format then l = formatPath(l) end
      table.insert(a,l)
    end
  end
  return a
end

-- It returns a file which have all the classes that dir is containing in its
-- images. This classes are also inside synset_words file.
-- dir MUST have an structure with subpredictedFiles and each subfolder has to be extracted
-- using the functions given in file_functions
function getDirClass(dir, outputFile)
  arr = scandir(dir)
  syn = {}
  for i = 1, #synset_words do
    splitted = synset_words[i]:split(" |,|'")
    path = table.concat(splitted,"-")
    table.insert(syn, path:lower())
  end
  count = {}
  for i = 3, #arr do
    p = io.popen('find "'..dir .. "/".. arr[i] ..'" -type f')
    c = 0
    for lines in p:lines() do
      c = c +1
    end
    count[i] = c
  end
  result = ""
  for v=3,  #arr do
    if(not (arr[v]:match(".txt"))) then
      bool, pos = utils.contains(syn,arr[v])
      result = result .. arr[v].. " | " .. pos .. " | " .. count[v] ..'\n'
    end
  end
  file = io.open("data/".. outputFile .. ".txt", "w")
  file:write(result)
  file:close()
end

--Parse a file and put each line of it into an array
function fileToArray(path, sub)
  local file = io.open(path, "r");
  local arr = {}
  for line in file:lines() do
    table.insert (arr, line:sub(sub));
  end
  return (arr)
end

--Duplicates the files that are in more than one class
function moreThanAClass(dir)
  -- find all the files in a given dir
  p = io.popen('find "'..dir..'" -type f')
  for file in p:lines() do
    -- Loop through the files and add them to the predictedFiles that they are supposed
    -- to be in in case they are not.
    if (file:find("[.jpg][.png][.jpeg]") ~= nil) then
      mkdir = false
      -- Get all the predicted classes that an item could be
      array, name = getContains(synset_words, file)
      for i=1, #array do
        -- Create new directory if needed
        if(not mkdir and not isDir(dir .. '/' ..array[i])) then
          os.execute("mkdir " .. dir .. '/' ..array[i])
          mkdir = true
        end
        --Create new file if needed
        if(not isFile(dir .. '/' ..array[i] .. '/' .. name)) then
          os.execute("cp " .. file .. " " .. dir .. '/' ..array[i] .. '/' .. name)
        end
      end
    end
  end
  print("done")
end


--Search a set of classes in a set of images
function searchClass(words, class, paths)
  local path;
  local added;
  for j = 1, table.getn(class) do
    added = false;
    for i = 1, table.getn(words) do
      splitted = words[i]:split(" |,|'")
      path = table.concat(splitted,"-")
      for w = 1, table.getn(splitted) do
        if (class[j] == splitted[w]) then
          if(not(isDir("inputs/"..path))) then
            os.execute("mkdir " .. "inputs/" .. path)
          end
          os.execute("cp " .. paths[j] .. " inputs/" .. path)
          print("cp " .. paths[j] .. " inputs/" .. path)
          added = true;
          break;
        end
      end

      if added then
        break;
      end
    end
  end
end

-- True if name is a path to a directory, false otherwise
function isDir(name)
  if type(name)~="string" then return false end
  local cd = lfs.currentdir()
  local is = lfs.chdir(name) and true or false
  lfs.chdir(cd)
  return is
end

-- True if name is a path to a file, false otherwise
function isFile(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

function scandir(directory)
    local i, t, popen = 0, {}, io.popen
    local pfile = popen('ls -lrt -d -R "'..directory..'"')
    for filename in pfile:lines() do

        if (filename:find('.jpg') ~= nil or
            filename:find('.png') ~= nil or
            filename:find('.jpeg') ~= nil) then
          i = i + 1
          t[i] = filename
        end
    end
    pfile:close()
    return t
end

-- This function splits in 3 folders the dataset. This folders are train, crossv and
-- test. You can set the % of images that will go to train, the rest will go
-- to the val dataset. Finally it writes all the paths to the files in three different
-- files
function splitTrainTest(path, outDir, t)
  local lines = fileToArray(path,0)
  local folders = {}

  -- Create output Directory in case it doesnt exist
  if (not isDir(outDir)) then os.execute("mkdir " .. outDir) end
  -- Create train y validation dirs in the Dataset
  if (not isDir(outDir .. "/train")) then os.execute("mkdir " .. outDir .. "/train") end
  if (not isDir(outDir .. "/val")) then os.execute("mkdir " .. outDir .. "/val") end

  --Split the whole dataset by classes
  for i =1, #lines do
    class = lines[i]:split("/") class = class[#class - 1]
    if folders[class] == nil then folders[class] = {} end
    table.insert(folders[class], lines[i])

    -- Make dataset structure, only use classes with 3 examples or more
    if #folders[class] >= 3 then
      --Create folders for each class if they are not already created
      if (not isDir(outDir.. "/train/" .. class)) then os.execute("mkdir " .. outDir .. "/train/" .. class) end
      if (not isDir(outDir.. "/val/" .. class)) then os.execute("mkdir " .. outDir .. "/val/" .. class) end
    end
  end

  -- Split each class separately in train, test regarding
  -- the given parameters
  local training, test = {}, {}
  for k,v in pairs(folders) do
    if (#folders[k] >= 3 ) then
      local train = math.floor(#folders[k]*t)
      print(k .. ":: " .. "Train 1-" .. train .. " | Test " .. (train+1) .. "-" .. #folders[k])
      training[k] = subrange(folders[k], 1, train)
      test[k] = subrange(folders[k], train+1, #folders[k])
    else
      print ("Class doesn't have enough samples ".. k)
    end
  end

  local function copySubset(subset,set)
    -- Copy images to their respective folders
    for class, v in pairs(subset) do
      for _,fpath in pairs(v) do
        name = fpath:split("/")
        name = name[#name]
        os.execute("cp " .. fpath .. " " .. outDir .. "/" .. set .."/" .. class .. "/".. name)
      end
    end
  end --end of copySubset

  copySubset(training,"train")
  copySubset(test,"val")
end

M.splitTrainTest = splitTrainTest
M.dirLookup = dirLookup
M.fileToArray = fileToArray
M.searchClass = searchClass
M.isDir = isDir
M.testingDir = testingDir
M.isFile = isFile
M.scandir = scandir
M.moreThanAClass = moreThanAClass
M.getDirClass = getDirClass
M.getImageList = getImageList
M.formatPath = formatPath

return M
