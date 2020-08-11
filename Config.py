import json
import os

def Save(filepath, config, ext=".txt"):
    last_slash = 0
    for i in range(len(filepath)):
        if filepath[i] == "/":
            last_slash = i

    sub_path = filepath[:last_slash]

    if not os.path.exists(sub_path):
        os.mkdir(sub_path)

    if ext not in filepath:
        filepath += ext

    json.dump(config, open(filepath, "w"), indent=4)

def Load(filepath, ext=".txt"):
    if ext not in filepath:
        filepath += ext
    return json.load(open(filepath))