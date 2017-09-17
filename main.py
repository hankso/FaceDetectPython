#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 19:30:21 2017

@author: hank

Attention: This script needs file 'haarcascade_frontalface_default.xml' in the same folder to run smoothly.
"""

import cv2, keras
import sys, os, time, threading, numpy as np
import detector

class CamControl(object):
    def __init__(self, cam_index = 0, xml_path = 'haarcascade_frontalface_default.xml'):
        #大标题
        self.winname = 'Face Dectection (by Hank)'
        
        #提取面部工具
        self.xml = self._load_xml(xml_path)
        
        #打开摄像头
        self.cam = self._open_cam(cam_index)
        
        #机器学习模型
        self.detector = detector.Detector()
        
        #标志位
        self.FLAG_Detect = False
        self.FLAG_Mirror = False
        self.FLAG_Vmiror = False
        self.FLAG_Display = True
        self.FLAG_Light = np.array(0, dtype = 'uint8')
        
        #训练参数
        self.train_params \
            = dict(loss = 'categorical_crossentropy',
                   optimizer = keras.optimizers.rmsprop(lr=0.0001,decay=1e-6),
                   metrics = ['accuracy'],
                   batch_size = 32,
                   epochs = 10,
                   verbose = 0,
                   generator = False,
                   use_saved_model = True)
        # use_saved_model is for debugging.
        
        #帧率
        self.frame_num = 48
        
        self.run()

    
    def _load_xml(self, xml_path):
        if not os.path.exists(xml_path):
            print("\33[1mAttention!\33[0m\n    Can not find xml file(e.g. \33[1m'haarcascade_frontalface_default.xml'\33[0m]) in the same folder to run script smoothly.")
            sys.exit(0)
        return cv2.CascadeClassifier(xml_path)
    
    def _open_cam(self, cam_index):
        cam = cv2.VideoCapture(cam_index)
        for i in range(3):
            if cam.isOpened():
                for _ in range(10):
                    cam.read()
                return cam
            try:
                cam.open()
            except:
                print('Can not open camera! Try again after 3 seconds.')
                time.sleep(3)
        print("\n\33[1mFailed open camera at index %d !\33[0m\nAbort." % cam_index)
        sys.exit(0)
        
    def _train(self):
        img = cv2.GaussianBlur(self.frame, (11, 11), 0)
        cv2.putText(img = img,
                    text = 'Training data...',
                    org = (80, 250),
                    fontFace = cv2.FONT_ITALIC,
                    fontScale = 3,
                    color = (255, 255, 255),
                    thickness = 3,
                    lineType = cv2.LINE_AA)
        
        pre_name, self.winname = self.winname, 'temp window when training data'
        
        cv2.imshow(pre_name, img)
        
        threading.Thread(target = self.run, args = (True,)).start()
        
        self.detector.train(**self.train_params)
        
        self.winname = pre_name + '  with model trained'
        
        
    def _process_frame(self, temp_frame):
        if self.FLAG_Mirror:
            self.frame = cv2.flip(self.frame, 1)
        if self.FLAG_Vmiror:
            self.frame = cv2.flip(self.frame, 0)
        self.frame += self.FLAG_Light
        self.frame[self.frame > 255] = 255
        self.frame[self.frame < 0  ] = 0
        
        if temp_frame:
            self.frame = cv2.resize(self.frame, None, fx = 0.5, fy = 0.5)
            
    def _display_frame(self, temp_frame = False):
        if self.FLAG_Display and not temp_frame:
            
            #detect_face
            frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            faces = self.xml.detectMultiScale(frame_gray, 1.3, 5)
            
            for (c, r, w, h) in faces:
                ratio = w / 230.0 * 0.8
                if self.FLAG_Detect:
                    faith, text = self.detector.detect(self.frame[c-10:c+w+10, r-10:r+h+10])
                    if faith < 0.5:
                        color = (0, 0, 255)
                    else:
                        color = (0, 255, 0)
                else:
                    text = 'unable'
                    color = (255, 255, 255)
                    
                #small_rect
                cv2.rectangle(img = self.frame,
                              pt1 = (c, r-1),
                              pt2 = (c + int(19*ratio)*len(text), r - int(30*ratio)),
                              color = color,
                              thickness = 1,
                              lineType = cv2.LINE_AA)
                #big_rect
                cv2.rectangle(img = self.frame,
                              pt1 = (c, r),
                              pt2 = (c + w, r + h),
                              color = color,
                              thickness = 1,
                              lineType = cv2.LINE_AA)
                #text
                cv2.putText(img = self.frame,
                            text = text,
                            org = (c, r-5),
                            fontFace = cv2.FONT_ITALIC,
                            fontScale = ratio,
                            color = color,
                            thickness = 1,
                            lineType = cv2.LINE_AA)
                
        cv2.imshow(self.winname, self.frame)
                
    def _watch_key(self, key):
        if key == 32:
            # SPACE --> 32
            self.FLAG_Display = not self.FLAG_Display
            
        # Left --> 81  Up --> 82  Right --> 83  Down --> 84
        elif key == 81:
            pass
        elif key == 82:
            self.FLAG_Light += np.array(10, dtype = 'uint8')
        elif key == 83:
            pass
        elif key == 84:
            self.FLAG_Light -= np.array(10, dtype = 'uint8')
            
        elif key in [100, 10]:
            # ENTER --> 10  Detect --> 100
            if self.detector.trained: self.FLAG_Detect = not self.FLAG_Detect
        
        elif key == 109:
            # Mirror --> 109
            self.FLAG_Mirror = not self.FLAG_Mirror
                
        elif key in [113, 27]:
            # Quit --> 113  ESC --> 27
            cv2.destroyAllWindows()
            self.cam.release()
            
        elif key == 114:
            # Reload --> 114
            reload(detector)
            self.detectot = detector.Detector()
            print('\nReloaded modules: \33[1mdetector\33[0m')
            
        elif key == 115:
            # Save --> 115
#==============================================================================
#             threading.Thread(target = detector.save_photo, args = ([frame],)).start()
#             # 因为python传参时数字、字符或元组满足传值规则，字典或列表满足传址规则
#             # 为了新线程中能够多次存储照片，所以传一个列表，相当于传了一个对象
#==============================================================================
            threading.Thread(target = self.detector.save_photo, args = (self.cam,)).start()
            # why not deliver a camera object directly?
    
        elif key == 116:
            # Train --> 116
            self._train()
            
#==============================================================================
#             threading.Thread(target = self.detector.train,
#                              kwargs = self.train_params,
#                              ).start()
#==============================================================================
            
        elif key == 118:
            # Vertical --> 118
            self.FLAG_Vmiror = not self.FLAG_Vmiror
            
        
    def run(self, temp_frame = False):
        while self.cam.isOpened():
            
            boolean, self.frame = self.cam.read()
            
            self._process_frame(temp_frame)
            
            self._display_frame(temp_frame)
            
            self._watch_key(cv2.waitKey(1000/self.frame_num))

            if temp_frame and self.detector.trained:
                cv2.destroyWindow(self.winname)
                break

if __name__ == '__main__':
    app = CamControl()
