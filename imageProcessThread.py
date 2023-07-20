from threading import Thread

class imageProcessThread(Thread):
    def __init__(self):
        super().__init__()
        self.stat = "runing"
    
    def setHandlerAndPath(self, handle, path):
        self.imageProcess = handle
        self.filepath = path

    def run(self):
        self.imageProcess(self.filepath)
        print("Done")

    def stop(self):
        self.terminate()