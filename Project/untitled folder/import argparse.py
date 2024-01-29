import argparse
import Models
import queue
import cv2
import numpy as np
from PIL import Image, ImageDraw

class BallDetector:
    def __init__(self, path_weights=None, n_classes =  256, device='cuda'):
        self.n_classes = n_classes
        self.width , self.height = 640, 360
        # model definition
        modelFN = Models.TrackNet.TrackNet
        m = modelFN(n_classes,input_height=self.height,input_width=self.width)
        m.compile(loss='categorical_crossentropy', optimizer= 'adadelta' , metrics=['accuracy'])
        m.load_weights(path_weights)
        self.output_width = 0
        self.output_height = 0
        self.model = m
        self.device = device

    def read_video(self,path_video):
        """ Read video file    
        :params
            path_video: path to video file
        :return
            frames: list of video frames
            fps: frames per second
        """
        cap = cv2.VideoCapture(path_video)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.output_width =  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                break
        cap.release()
        return frames, fps
    
    def resize_frames(self,frames, width, height):
        f = []
        for frame in frames:
            new_frame = cv2.resize(frame, ( width , height ))
            new_frame = new_frame.astype(np.float32)
            f.append(new_frame)
        return f


    def preset_output_video(self,f1,f2):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_video = cv2.VideoWriter('output.mp4',fourcc, 30.0, (self.output_width,self.output_height))
        # first two frames
        output_video.write(f1)
        output_video.write(f2)
        return output_video
    

    def infer_model(self, frames):

        output_video = self.preset_output_video(frames[0],frames[1])
        oldframes = frames.copy()
        frames = self.resize_frames(frames, self.width, self.height)
        q = queue.deque()
        for i in range(0,8):
            q.appendleft(None)
        
        ballpoints = [None,None]
        for i in range(2,len(frames)):
            X =  np.concatenate((frames[i], frames[i-1], frames[i-2]),axis=2)
            X = np.rollaxis(X, 2, 0)
            pr = self.model.predict(np.array([X]))[0]
            pr = pr.reshape((self.height ,self.width , self.n_classes)).argmax( axis=2 )
            pr = pr.astype(np.uint8) 
            heatmap = cv2.resize(pr,(self.output_width,self.output_height))
            ret,heatmap = cv2.threshold(heatmap,127,255,cv2.THRESH_BINARY)
            out = oldframes[i]
            #find the circle in image with 2<=radius<=7
            circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT,dp=1,minDist=1,param1=50,param2=2,minRadius=2,maxRadius=7)
            # for drawing
            PIL_image = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)   
            PIL_image = Image.fromarray(PIL_image)
            x,y=None,None
            if circles is not None:
                #if only one tennis be detected
                if len(circles) == 1:
                    x = int(circles[0][0][0])
                    y = int(circles[0][0][1])
                    q.appendleft([x,y])   
                    q.pop()    
                else:
                    q.appendleft(None)
                    q.pop()
            else:
                q.appendleft(None)
                q.pop()
            ballpoints.append([x,y])
            for i in range(0,8):
                if q[i] is not None:
                    draw_x = q[i][0]
                    draw_y = q[i][1]
                    bbox =  (draw_x - 2, draw_y - 2, draw_x + 2, draw_y + 2)
                    draw = ImageDraw.Draw(PIL_image)
                    draw.ellipse(bbox, outline ='red')
                    del draw
            
            opencvImage =  cv2.cvtColor(np.array(PIL_image), cv2.COLOR_RGB2BGR)
            output_video.write(opencvImage)

        output_video.release()
        return ballpoints

    def postprocess(self, feature_map, prev_pred, scale=2, max_dist=80):
        """
        :params
            feature_map: feature map with shape (1,360,640)
            prev_pred: [x,y] coordinates of ball prediction from previous frame
            scale: scale for conversion to original shape (720,1280)
            max_dist: maximum distance from previous ball detection to remove outliers
        :return
            x,y ball coordinates
        """
        feature_map *= 255
        feature_map = feature_map.reshape((self.height, self.width))
        feature_map = feature_map.astype(np.uint8)
        ret, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
        circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2,
                                   maxRadius=7)
        x, y = None, None
        if circles is not None:
            if prev_pred[0]:
                for i in range(len(circles[0])):
                    x_temp = circles[0][i][0]*scale
                    y_temp = circles[0][i][1]*scale
                    dist = distance.euclidean((x_temp, y_temp), prev_pred)
                    if dist < max_dist:
                        x, y = x_temp, y_temp
                        break                
            else:
                x = circles[0][0][0]*scale
                y = circles[0][0][1]*scale
        return x, y
