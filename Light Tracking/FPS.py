import datetime

class FPS:
    def __init__(self):
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        #start the timer
        self._start() = datetime.datetime.now()
        return self
    
    def stop(self):
        #stop the timer
        self._end = datetime.datetime.now()

    def update(self):
        #update number of frames since start
        self._numFrames += 1

    def elapsed(self):
        #return time elapsed
        return (self._end - self._start).total_seconds()

    def fps(self):
        #calculate and return FPS
        return self._numFrames/self.elapsed()
