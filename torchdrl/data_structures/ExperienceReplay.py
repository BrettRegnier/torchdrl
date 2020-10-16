class ExperienceReplay:
    def __init__(self):
        self._ready = True

    @property
    def Ready(self):
        return self._ready