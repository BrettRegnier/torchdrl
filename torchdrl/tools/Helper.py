import os
import torch

def AppendExtension(filepath, ext):
    if ext[0] != ".":
        ext = "." + ext

    if ext.lower() != filepath[-5:].lower():
        filepath += ext
    
    return filepath

def AppendSlash(folderpath):
    folderpath = folderpath if folderpath[len(folderpath) - 1] == "/" else folderpath + "/"
    return folderpath

def CreateSaveLocation(folderpath):
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)

def CheckFileLocation(filepath):
    return os.path.exists(filepath):

def SplitFilename(filepath):
    idx = filepath.rfind("/")
    filename = filepath[idx+1:]
    folderpath = filepath[:idx]
    return (filename, folderpath)

def SaveAgent(folderpath, filename, agent_dict):
    filename = AppendExtension(filename, "pt")
    folderpath = AppendSlash(folderpath)
    CreateSaveLocation(folderpath)

    torch.save(agent_dict, folderpath + filename)

def LoadAgent(filepath):
    if CheckFileLocation(filepath):
        checkpoint = torch.load(filepath)
        return checkpoint 
    else:
        filename, folderpath = Helper.GetFileName(filepath)
        raise Exception(f"{filename} does not exist at {folderpath}")