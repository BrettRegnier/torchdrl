class Agent:
    def Train(self):
        raise NotImplementedError("Error. Agent must have Train function implemented")

    def TrainNoYield(self):
        raise NotImplementedError("Error. TrainNoYield function not implemented")

    def Learn(self):
        raise NotImplementedError("Error. Learn function not implemented")

    def Evaluate(self):
        raise NotImplementedError("Error. Evaluate not implemented")

    def GetAction(self, state, evaluate=False):
        raise NotImplementedError("Error. Agent must have GetAction function implemented")
    
    def Act(self, state):
        raise NotImplementedError("Error. Agent must have Act function implemented")

    def Save(self, folderpath, filename):
        raise NotImplementedError("Error. Save function not implemented")

    def Load(self, filepath):
        raise NotImplementedError("Error. Load function not implemented")