class Agent:
    def Train(self):
        raise NotImplementedError

    def TrainNoYield(self):
        raise NotImplementedError

    def Evaluate(self):
        raise NotImplementedError

    def Act(self):
        raise NotImplementedError

    def GetAction(self):
        raise NotImplementedError

    def Save(self):
        raise NotImplementedError

    def Load(self):
        raise NotImplementedError