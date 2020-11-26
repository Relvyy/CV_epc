from src.lib.detectors.detector_factory import detector_factory
from src.lib.opts import opts
import cv2
import time
MODEL_PATH ='/home/epc/Documents/ubt/documents/Project/CenterNet-fire/exp/ctdet/default/model_120.pth'
TASK = 'ctdet' # or 'multi_pose' for human pose estimation
opt = opts().init('{} --load_model {}'.format(TASK, MODEL_PATH).split(' '))
opt.debug = 1
detector = detector_factory[opt.task](opt)


def detet_image(image_path):
    detector.run(image_path)


class camera():#摄像头读取
    def __init__(self):
        self.frame=[]
    def read_video(self,path):
        vdo = cv2.VideoCapture(path)
        # vdo=cv2.VideoCapture(0)
        while 1:
            ret, frame = vdo.read()
            # print('111')
            if ret:
                # print('read frame and update')
                self.frame=frame
                # self.frame = cv2.resize(frame, (640, 480))
            else:
                print('丢失相机')
                vdo = cv2.VideoCapture(path)


Camera1=camera()
def c1():
    global Camera1
    Camera1.read_video('rtsp://admin:epc2019.@192.168.0.64')  #启动循环
    # Camera1.read_video(0)

def main1():
    global Camera1

    # Start training
    while 1:
        if Camera1.frame==[]:
            continue
        t1=time.time()
        detector.pause=False

        ret = detector.run(Camera1.frame)['results']
        print(1/(time.time()-t1) ,'fps')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    import threading
    t1=threading.Thread(target=c1)
    t2=threading.Thread(target=main1)
    t1.start()
    t2.start()
    # detet_image('./images/5.jpg')


