import json
import os

import torchdrl.tools.Helper as Helper

def Save(folderpath, filename, config, ext=".json"):
    Helper.CreateSaveLocation(folderpath)
    filename = Helper.AppendExtension(filename, ext)
    json.dump(config, open(filepath, "w"), indent=4)

def Load(filepath):
    if Helper.CheckFileLocation(filepath):
        return json.load(open(filepath))
    else:
        filename, folderpath = Helper.GetFileName(filepath)
        raise Exception(f"{filename} does not exist at {folderpath}")

