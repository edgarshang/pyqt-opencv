import cv2 as cv
import time
import numpy as np
import onnxruntime
from openvino.runtime import Core

from Iimageprocess import Process, Show
from UIQt import log

class YOLOV8_Process(Process):

    def __init__(self):
        self.imageShow = Show()
        self.colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]
        self.class_list = self.load_classed()
    
    def setImageShower(self, imshow):
        self.imageShow  = imshow

    def setShowImage(self, callback):
        self.showImage = callback
    def imageprocess(self, imageInfo):
        print("yolo v8")
        imagePath = imageInfo["namePath"]
        if imageInfo["typeCal"] == "Pt":
            model = "yolov8s.onnx"
        elif imageInfo["typeCal"] == "onnx":
            model = "yolov8s.onnx"

        log.debug(f'the typeCal is {imageInfo["typeCal"]}')
        log.debug(f'the deploy is {imageInfo["deploy"]}')


        if imageInfo["deploy"] == "Openvino":
            ie = Core()
            for device in ie.available_devices:
                print(device)

            # Read IR
            model = ie.read_model(model="yolov8n.onnx")
            self.mode = ie.compile_model(model=model, device_name="CPU")
            output_layer = self.mode.output(0)
        elif imageInfo["deploy"] == "onnxruntime":
            self.mode = onnxruntime.InferenceSession("yolov8n.onnx",
                                                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            
            #  providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            # session = onnxruntime.InferenceSession(w, providers=providers)
            

            
            output_names = [x.name for x in self.mode.get_outputs()]
            log.debug(output_names)
            meta = self.mode.get_modelmeta().custom_metadata_map  # metadata
            self.names = eval(meta['names'])
    
            # return
        elif imageInfo["deploy"] == "tensorrt":
            pass
        elif imageInfo["deploy"] == "opencv":
            pass
        
        capture = cv.VideoCapture(imagePath)
        while True:
            _, frame = capture.read()
            if frame is None:
                print("End of stream")
                break
            bgr = self.format_yolov8(frame)
            img_h, img_w, img_c = bgr.shape

            start = time.time()
            image = cv.dnn.blobFromImage(bgr, 1 / 255.0, (640, 640), swapRB=True, crop=False)

            if imageInfo["deploy"] == "Openvino":
                res = self.mode([image])[output_layer]
            elif imageInfo["deploy"] == "onnxruntime":
                ort_inputs = {self.mode.get_inputs()[0].name: image}
                res = self.mode.run(None, ort_inputs)[0]

           

            # matrix transpose from 1x84x8400 => 8400x84
            rows = np.squeeze(res, 0).T

            # post-process
            class_ids = []
            confidences = []
            boxes = []
            x_factor = img_w / 640
            y_factor = img_h / 640

            for r in range(rows.shape[0]):
                row = rows[r]
                classes_scores = row[4:]
                _, _, _, max_indx = cv.minMaxLoc(classes_scores)
                class_id = max_indx[1]
                if (classes_scores[class_id] > .25):
                    confidences.append(classes_scores[class_id])
                    class_ids.append(class_id)
                    x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                    left = int((x - 0.5 * w) * x_factor)
                    top = int((y - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = np.array([left, top, width, height])
                    boxes.append(box)

            # NMS
            indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)
            for index in indexes:
                box = boxes[index]
                color = self.colors[int(class_ids[index]) % len(self.colors)]
                cv.rectangle(frame, box, color, 2)
                cv.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
                cv.putText(frame, self.class_list[class_ids[index]], (box[0], box[1] - 10), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))
                # cv.putText(frame, self.names[(class_ids[index])], (box[0], box[1] - 10), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))
                # log.debug(self.class_list[(class_ids[index])])
                # log.debug(self.names[(class_ids[index])])
            end = time.time()
            inf_end = end - start
            fps = 1 / inf_end
            fps_label = "FPS: %.2f" % fps
            cv.putText(frame, fps_label, (20, 45), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.imageShow.imageshow(frame)



    def load_classed(self):
        with open("classes.txt", "r") as f:
            class_list = [cname.strip() for cname in f.readlines()]
        return class_list
    
    def format_yolov8(self, frame):
        row, col, _ = frame.shape
        _max = max(col, row)
        result = np.zeros((_max, _max, 3), np.uint8)
        result[0:row, 0:col] = frame
        return result

