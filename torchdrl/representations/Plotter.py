import matplotlib.pyplot as plt
import numpy as np
import json
import os

class Plotter:
    def __init__(self):
        self._figures = {}

    def AddFigure(self, figure:str, label: str, colour="green"):
        if figure not in self._figures:
            self._figures[figure] = {}

        self._figures[figure][label] = {"x": [], "y": [], "colour": colour}
    
    def AddPoint(self, figure:str, label: str, point:tuple):
        if not isinstance(figure, str):
            raise AssertionError("Figure must be of type string")
        if not isinstance(label, str):
            raise AssertionError("Label must be of type string")
        if not isinstance(point, tuple) or len(point) != 2 or not isinstance(point[0], (int,float)) or not isinstance(point[1], (int, float)):
            raise AssertionError("Point must be a tuple of length 2 in form (x:int, y:int)")

        if figure not in self._figures:
            self.AddFigure(figure, label)

        x, y = point
        self._figures[figure][label]['x'].append(x)
        self._figures[figure][label]['y'].append(y)

    def AddPoints(self, figure:str, label:str, points:list):
        for point in points:
            self.AddPoint(figure, label, point)
    
    def ShowFigure(self, figure, label):
        plt.figure(figure, constrained_layout=True)
        plt.plot(self._figures[figure][label]['x'], self._figures[figure][label]['y'], self._figures[figure][label]['colour'])
        plt.pause(0.05)		
        plt.show(block=False)

    def ShowAll(self):
        for figure in self._figures:
            plt.figure(figure, figsize=(12, 10), constrained_layout=True) 
            plt.clf()
            plt.xlabel("Episode") #TODO parameter
            for label in self._figures[figure]:
                plt.plot(self._figures[figure][label]['x'], self._figures[figure][label]['y'], color=self._figures[figure][label]['colour'], label=label)
            plt.legend(bbox_to_anchor=(1.05, 1), borderaxespad=0, fontsize='xx-small')

        plt.pause(0.05)		
        plt.show(block=False)

    def SaveImage(self, figure, path):
        try:
            last_slash = 0
            for i in range(len(path)):
                if path[i] == "/" or path[i] == "\\":
                    last_slash = i

            sub_path = path[:last_slash]
            filename = path[last_slash+1:] #maybe

            if not os.path.exists(sub_path):
                os.mkdir(sub_path)

            if ".png" not in filename:
                path += ".png"
            
            plt.figure(figure)
            plt.savefig(path)
        except:
            print("Figure name: " + figure + " not found in figures")

    def SaveAllImages(self, path):
        for figure in self._figures:
            if path[-1] != "/" or path[-1] != "\\":
                path += "/"
            path += figure
            self.SaveImage(figure, path)

    def SavePoints(self, figure, path):
        try:
            last_slash = 0
            for i in range(len(path)):
                if path[i] == "/" or path[i] == "\\":
                    last_slash = i

            sub_path = path[:last_slash]
            filename = path[last_slash+1:] #maybe

            if not os.path.exists(sub_path):
                os.mkdir(sub_path)

            if ".json" not in filename:
                path += ".json"
            
            json.dump(self._figures[figure], open(path, "w"), indent=4)
        except:
            print("Figure name: " + figure + " not found in figures")

    def SaveAllPoints(self, path):
        for figure in self._figures:
            if path[-1] != "/" or path[-1] != "\\":
                path += "/"
            path += figure
            self.SavePoints(figure, path)

    def RemovePoint(self, figure:str, label: str, point:tuple):
        if figure in self._figures:
            if label in self._figures[label]:
                x1, y1 = point
                for p in self._figures[figure][label]:
                    x2, y2 = p
                    if x1 == x2 and y1 == y2:
                        self._figures[figure][label]['x'].remove(x1)
                        self._figures[figure][label]['y'].remove(x2)
                        return
    
    def RemovePointIdx(self, figure: str, label:str, idx:int):
        if figure in self._figures:
            if label in self._figures[label]:
                self._figures[figure][label]['x'].pop(idx)
                self._figures[figure][label]['y'].pop(idx)
    
    def ClearPlot(self, name):
        self.NewPlot(name)

    def ClearAll(self):
        for plot in self._plots:
            self.NewPlot(plot)

    def DeletePlot(self, name):
        self._plots.pop(name, None)

    def DeleteAll(self):
        for plot in self._plots:
            self._plots.pop(name, None)