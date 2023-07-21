import cv2
import numpy as np
import torch
from pathlib import Path
import os
import glob
import time

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes

#  images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
#         videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]


from Iimageprocess import Process, Show
from landmark.landmark_cnn import Net, ChannelPool
from UIQt import log



class landMark_Process(Process):
    
    def __init__(self):
        self.imageShow = Show()
        self.face_model = "./common_model/face_model/opencv_face_detector_uint8.pb"
        self.face_model_protxt = "./common_model/face_model/opencv_face_detector.pbtxt"
        self.face_model_detect = self.load_face_detect_model()
        self.land_mark_detect = self.load_land_mark_model()
      
        
    
    def setImageShower(self, imshow):
        self.imageShow  = imshow

    def load_face_detect_model(self):
        return cv2.dnn.readNetFromTensorflow(self.face_model, config=self.face_model_protxt)

    def load_land_mark_model(self):
        # return torch.load("./landmark/model_landmarks_new.pt")
        new_mode = Net()
        if torch.cuda.is_available():
            new_mode.cuda()
        new_mode.load_state_dict(torch.load("./landmark/model_landmarks_new.pt"))
        new_mode.eval()
        return new_mode
    
    def frameHandle(self, frame):
        start_time = time.time()
        h,w,c = frame.shape
        blobImage = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
        self.face_model_detect.setInput(blobImage)
        cvOut = self.face_model_detect.forward()
        for detection in cvOut[0,0,:,:]:
            score = float(detection[2])
            if score > 0.5:
                left = detection[3]*w
                top = detection[4]*h
                right = detection[5]*w
                bottom = detection[6]*h
                if left < 0:
                    left = 0
                if top < 0:
                    top = 0
                if right >= w:
                    right = w - 1
                if bottom >= h:
                    bottom = h - 1

                # roi and detect landmark
                roi = frame[np.int32(top):np.int32(bottom),np.int32(left):np.int32(right),:]
                rw = right - left
                rh = bottom - top
                img = cv2.resize(roi, (64, 64))
                img = (np.float32(img) / 255.0 - 0.5) / 0.5
                img = img.transpose((2, 0, 1))
                x_input = torch.from_numpy(img).view(1, 3, 64, 64)
                if torch.cuda.is_available():
                    probs = self.land_mark_detect(x_input.cuda())
                else:
                    probs = self.land_mark_detect(x_input)
                lm_pts = probs.view(5, 2).cpu().detach().numpy()
                for x, y in lm_pts:
                    x1 = x * rw
                    y1 = y * rh
                    cv2.circle(roi, (np.int32(x1), np.int32(y1)), 2, (0, 0, 255), 2, 8, 0)
                inf_end = time.time() - start_time
                fps = 1 / inf_end
                fps_label = "FPS: %.2f" % fps
                cv2.putText(frame, fps_label, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # 绘制
                cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), thickness=2)
                cv2.putText(frame, "score:%.2f"%score, (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                self.imageShow.imageshow(frame)
    def imageprocess(self, info):
        print("hello, world, this is the landmark")
        iscamera =info["camera"]
        print(iscamera)
        path = info["namePath"]
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            p = str(Path(p).resolve())
            if '*' in p:
                files.extend(sorted(glob.glob(p, recursive=True)))  # glob
            elif os.path.isdir(p):
                files.extend(sorted(glob.glob(os.path.join(p, '*.*'))))  # dir
            elif os.path.isfile(p):
                files.append(p)  # files
            else:
                raise FileNotFoundError(f'{p} does not exist')
            
        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]

        ni, nv = len(images), len(videos)
        print(images)
        for i in range(ni):
            frame = cv2.imread(images[i])
            self.frameHandle(frame)
            
        for j in range(nv):
            capture = cv2.VideoCapture(videos[j])
            while True:
                ret, frame = capture.read()
                if ret is not True:
                    break
                self.frameHandle(frame)
        if iscamera:
            capture = cv2.VideoCapture(0)
            while True:
                ret, frame = capture.read()
                if ret is not True:
                    break
                self.frameHandle(frame)                

        

    











# def video_landmark_demo():
#     cnn_model = torch.load("./model_landmarks.pt")
#     # capture = cv.VideoCapture(0)
#     capture = cv.VideoCapture("../age_gentle/age_gender_cnn/example_dsh.mp4")

#     # load tensorflow model
#     net = cv.dnn.readNetFromTensorflow(model_bin, config=config_text)
#     while True:
#         ret, frame = capture.read()
#         if ret is not True:
#             break
#         frame = cv.flip(frame, 1)
#         h, w, c = frame.shape
#         blobImage = cv.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
#         net.setInput(blobImage)
#         cvOut = net.forward()
#         # print(cvOut.shape)
#         # 绘制检测矩形
#         index = 0
#         for detection in cvOut[0,0,:,:]:
#             print(detection.shape)
#             index += 1
#             print(f'index = {index}')
#             score = float(detection[2])
#             if score > 0.5:
#                 left = detection[3]*w
#                 top = detection[4]*h
#                 right = detection[5]*w
#                 bottom = detection[6]*h
#                 if left < 0:
#                     left = 0
#                 if top < 0:
#                     top = 0
#                 if right >= w:
#                     right = w - 1
#                 if bottom >= h:
#                     bottom = h - 1

#                 # roi and detect landmark
#                 roi = frame[np.int32(top):np.int32(bottom),np.int32(left):np.int32(right),:]
#                 rw = right - left
#                 rh = bottom - top
#                 img = cv.resize(roi, (64, 64))
#                 img = (np.float32(img) / 255.0 - 0.5) / 0.5
#                 img = img.transpose((2, 0, 1))
#                 x_input = torch.from_numpy(img).view(1, 3, 64, 64)
#                 probs = cnn_model(x_input.cuda())
#                 lm_pts = probs.view(5, 2).cpu().detach().numpy()
#                 for x, y in lm_pts:
#                     x1 = x * rw
#                     y1 = y * rh
#                     cv.circle(roi, (np.int32(x1), np.int32(y1)), 2, (0, 0, 255), 2, 8, 0)

#                 # 绘制
#                 cv.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), thickness=2)
#                 cv.putText(frame, "score:%.2f"%score, (int(left), int(top)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
#                 c = cv.waitKey(1)
#                 if c == 27:
#                     break
#                 cv.imshow("face detection + landmark", frame)

#     cv.waitKey(0)
#     cv.destroyAllWindows()


# if __name__ == "__main__":
#     video_landmark_demo()
#     # image_landmark_demo()