import os
from os import listdir
from os.path import isfile, join

from ..data_structures import Shared


class Commandline:
    def __init__(self):
        self._commands = ["show_training", "show_log", "save", "exit"]

    def Begin(self):
        choice = ""
        while choice != "exit":
            choice = input().lower()
            if choice == "show_training":
                shared.show_training = not shared.show_training
            elif choice == "show_log":
                shared.show_log = not shared.show_log
            elif choice == "save":
                shared.save = True
                
    def GetConfig(self):
        config_path = "configs"
        files = [f for f in listdir(
            config_path) if isfile(join(config_path, f))]
            
        for i in range(len(files)):
            print(str(i) + ". " + files[i])
        
        choice = ""
        while choice not in [str(i) for i in range(len(files))]:
            choice = input("Choose a valid number \n>")
        choice = int(choice)

        return config_path + "/" + files[choice]
