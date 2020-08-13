import matplotlib.pyplot as plt
import numpy as np

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
        plt.figure(figure)
        plt.plot(self._figures[figure][label]['x'], self._figures[figure][label]['y'], self._figures[figure][label]['colour'])
        plt.pause(0.05)		
        plt.show(block=False)

    def ShowAll(self):
        for figure in self._figures:
            plt.figure(figure) 
            plt.clf()
            plt.xlabel("Episode") #TODO parameter
            for label in self._figures[figure]:
                plt.plot(self._figures[figure][label]['x'], self._figures[figure][label]['y'], self._figures[figure][label]['colour'], label=label)
            plt.legend(loc="upper left")

        plt.pause(0.05)		
        plt.show(block=False)
        

    def Save(self, path):
        pass

    def SaveAll(self, path):
        pass

    def RemovePoint(self, x, y):
        pass
    
    def RemovePointIdx(self, idx):
        pass
    
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

# plot = Plotter()
# plot.AddPoint("test", (1, 1))
# plot.AddPoint("test", (2, 2))

# plot.AddPoint("x", (1, 1))
# plot.AddPoint("x", (2, 2))
# plot.ShowAll()

# input("press enter")
# plot.AddPoint("test", (3, 2))
# plot.ShowAll()
# input("press enter")
# plot.AddPoint("test", (4, 6))
# plot.ShowAll()
# input("press enter")