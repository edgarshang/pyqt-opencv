from Iimageprocess import Process

from yolov5.detect import run as yolov5_run

class YOLOV5_Process(Process):

    def setShowImage(self, callback):
        self.showImage = callback
    
    def imageprocess(self, imageInfo):
        print("yolo v5")
        imagePath = imageInfo["namePath"]
        if imageInfo["typeCal"] == "Pt":
            model = "yolov5s.pt"
        elif imageInfo["typeCal"] == "onnx":
            model = "yolov5s.onnx"
        yolov5_run(weights=model, source=imagePath, showCallBack=self.showImage)
