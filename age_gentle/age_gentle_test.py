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
from age_gentle.age_gender_cnn import MyMulitpleTaskNet
from UIQt import log

from openvino.inference_engine import IECore
genders = ['male', 'female']



class AgeGentle_Process(Process):
    
    def __init__(self):
        self.imageShow = Show()
        self.face_model = "./common_model/face_model/opencv_face_detector_uint8.pb"
        self.face_model_protxt = "./common_model/face_model/opencv_face_detector.pbtxt"
        self.face_model_detect = self.load_face_detect_model()
        self.age_gentle_detect = self.load_age_gentle_model()
      
        
    
    def setImageShower(self, imshow):
        self.imageShow  = imshow

    def load_face_detect_model(self):
        return cv2.dnn.readNetFromTensorflow(self.face_model, config=self.face_model_protxt)

    def load_age_gentle_model(self):
        # return torch.load("./landmark/model_landmarks_new.pt")
        new_mode = MyMulitpleTaskNet()
        if torch.cuda.is_available():
            new_mode.cuda()
        new_mode.load_state_dict(torch.load("./age_gentle/age_gentle_new.pt"))
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
                img = cv2.resize(roi, (64, 64))
                img = (np.float32(img) / 255.0 - 0.5) / 0.5
                img = img.transpose((2, 0, 1))
                x_input = torch.from_numpy(img).view(1, 3, 64, 64)
                age_, gender_ = self.age_gentle_detect(x_input.cuda())
                predict_gender = torch.max(gender_, 1)[1].cpu().detach().numpy()[0]
                gender = "Male"
                if predict_gender == 1:
                    gender = "Female"
                predict_age = age_.cpu().detach().numpy()*116.0
                print(predict_gender, predict_age)

                # 绘制
                cv2.putText(frame, ("gender: %s, age:%d"%(gender, int(predict_age[0][0]))), (int(left), int(top)-15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
                cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), thickness=2)
                self.imageShow.imageshow(frame)
    def face_age_gender_demo(self, path):
        ie = IECore()
        for device in ie.available_devices:
            print(device)

        model_xml = "./common_model/face_model/face-detection-0202.xml"
        model_bin = "./common_model/face_model/face-detection-0202.bin"

        net = ie.read_network(model=model_xml, weights=model_bin)
        input_blob = next(iter(net.input_info))
        out_blob = next(iter(net.outputs))

        n, c, h, w = net.input_info[input_blob].input_data.shape
        print(n, c, h, w)

        cap = cv2.VideoCapture(path)
        # cap = cv.VideoCapture(0)
        exec_net = ie.load_network(network=net, device_name="CPU")

        # 加载性别与年龄预测模型
        em_net = ie.read_network(model="./age_gentle/age_gender_model.onnx")
        em_input_blob = next(iter(em_net.input_info))
        em_it = iter(em_net.outputs)
        em_out_blob1 = next(em_it)
        em_out_blob2 = next(em_it)
        en, ec, eh, ew = em_net.input_info[em_input_blob].input_data.shape
        print(en, ec, eh, ew)

        em_exec_net = ie.load_network(network=em_net, device_name="CPU")

        while True:
            ret, frame = cap.read()
            if ret is not True:
                break
            image = cv2.resize(frame, (w, h))
            image = image.transpose(2, 0, 1)
            inf_start = time.time()
            res = exec_net.infer(inputs={input_blob:[image]})
            inf_end = time.time() - inf_start
            # print("infer time(ms)：%.3f"%(inf_end*1000))
            ih, iw, ic = frame.shape
            res = res[out_blob]
            for obj in res[0][0]:
                if obj[2] > 0.75:
                    xmin = int(obj[3] * iw)
                    ymin = int(obj[4] * ih)
                    xmax = int(obj[5] * iw)
                    ymax = int(obj[6] * ih)
                    if xmin < 0:
                        xmin = 0
                    if ymin < 0:
                        ymin = 0
                    if xmax >= iw:
                        xmax = iw - 1
                    if ymax >= ih:
                        ymax = ih - 1
                    roi = frame[ymin:ymax,xmin:xmax,:]
                    roi_img = cv2.resize(roi, (ew, eh))
                    roi_img = (np.float32(roi_img) / 255.0 - 0.5) / 0.5
                    roi_img = roi_img.transpose(2, 0, 1)
                    em_res = em_exec_net.infer(inputs={em_input_blob: [roi_img]})
                    gender_prob = em_res[em_out_blob1].reshape(1, 2)
                    prob_age = em_res[em_out_blob2].reshape(1, 1)[0][0] * 116
                    label_index = np.int32(np.argmax(gender_prob, 1))
                    age = np.int32(prob_age)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2, 8)
                    cv2.putText(frame, "infer time(ms): %.3f"%(inf_end*1000), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255),
                            2, 8)
                    cv2.putText(frame, ("gender: %s, age:%d"%(genders[label_index[0]], age)), (int(xmin), int(ymin)-15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2,8)
                    self.imageShow.imageshow(frame)
    def imageprocess(self, info):
        

        log.debug("hello, world, this is the ageGentle")
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
        if 0:
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
            # if iscamera:
            #     capture = cv2.VideoCapture(0)
            #     while True:
            #         ret, frame = capture.read()
            #         if ret is not True:
            #             break
            #         self.frameHandle(frame)        
            # 
        else:
            self.face_age_gender_demo(videos[0])
    # def face_age_gender_demo(self, path):


        

    











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