class Agent:
    def PlayEpisode(self):
        raise NotImplementedError("Error. PlayEpisode not impleneted")

    def EvaluateEpisode(self):
        raise NotImplementedError("Error. Evaluate not implemented")

    def GetAction(self, state, evaluate=False):
        raise NotImplementedError("Error. Agent must have GetAction function implemented")

    def Learn(self):
        raise NotImplementedError("Error. Learn function not implemented")

    def Update(self):
        raise NotImplementedError("Error. Learn function not implemented")

    def Save(self, folderpath, filename):
        raise NotImplementedError("Error. Save function not implemented")

    def Load(self, filepath):
        raise NotImplementedError("Error. Load function not implemented")