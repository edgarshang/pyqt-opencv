from Iimageprocess import Process

from yolov5.detect import run as yolov5_run

class YOLOV5_Process(Process):

    def setShowImage(self, callback):
        self.showImage = callback
    
    def imageprocess(self, imageInfo):
        print("yolo v5")
        yolov5_run(source=imageInfo, showCallBack=self.showImage)
        pass