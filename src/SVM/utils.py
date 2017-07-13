import h5py as h5

def getFeatures(inputFile, h5Matrix):
	h5f = h5.File(inputFile,"r")
	feat = h5f[h5Matrix][:]
	h5f.close()
	return feat

def getClasses(inputFile):
	with open(inputFile) as f:
		content = f.readlines()
	return [i.split("/")[-2] for i in content], content
