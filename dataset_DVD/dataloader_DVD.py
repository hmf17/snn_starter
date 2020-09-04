#-*- coding: utf-8 -*-
# pkg
import os
import cv2
import time
# import rosbag

def Test_Python_Opencv():
    # Find version for opencv-python
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    print('major versiob = %s, minor versiob = %s, subminor versiob = %s' % (major_ver, minor_ver, subminor_ver))

def Read_Deep_Video():
    # Get the DeepVideo file list
    # TODO: change to your dataset path to convent
    RootPath = '/media/hmf_love_long' \
               '/My Passport/学习文件/深度学习/数据集' \
               '/DeepVideoDeblurring' \
               '/Cheetahs'
    DatasetPath = RootPath + '/original_high_fps_videos'
    DeepVideoList = [DatasetPath + '/' + i for i in os.listdir(DatasetPath)]  # Absolute path
    DataStorePath = RootPath + '/processed_high_fps_videos'  # TODO: change to your dataset path to store

    print(DeepVideoList)
    for VideoName in DeepVideoList:
        VideoCapture = cv2.VideoCapture(VideoName)  # 捕捉视频，未开始读取；

        if VideoCapture.isOpened():  # VideoCaputre对象是否成功打开
            print('Opened Video:', (VideoName.split('/'))[-1])
            fps = VideoCapture.get(cv2.CAP_PROP_FPS)  # 返回视频的fps--帧率
            width = VideoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)  # 返回视频的宽
            height = VideoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 返回视频的高
            print('fps:', fps, 'width:', width, 'height:', height)

            # 计算得到视频的总帧数
            VideoAllFrames = [] # 储存解析得到的照片
            FrameCount = 0
            if os.path.exists(DataStorePath + '/' + (VideoName.split('/'))[-1]):
                pass
            else:
                os.mkdir(DataStorePath + '/' + (VideoName.split('/'))[-1])
                # ret 读取了数据就返回True,没有读取数据(已到尾部)就返回False
                # frame 返回读取的视频数据--一帧数据
                while (True):
                    ret, frame = VideoCapture.read()    # 读取一帧视频
                    if ret is False:
                        break
                    FrameCount = FrameCount + 1
                    FileName = DataStorePath + '/' + (VideoName.split('/'))[-1] + '/' + str(FrameCount) + '.png'
                    cv2.imwrite(FileName, frame)
                    print('Total frame number is ', FrameCount)
                    # time.sleep(0.1)
                # break

        else:
            print('视频文件打开失败')

def Write_Deep_Video( ):
    VideoFPS = 20  # 认为其原图象为240帧
    ImgPath = '/media/hmf_love_long' \
               '/My Passport/学习文件/深度学习/数据集' \
               '/DeepVideoDeblurring' \
               '/Cheetahs' \
               '/processed_high_fps_videos' \
               '/Cheetahs on the Edge — Director\'s Cut _ National/'
    VideoPath = '/media/hmf_love_long' \
               '/My Passport/学习文件/深度学习/数据集' \
               '/DeepVideoDeblurring' \
               '/Cheetahs'
    VideoPath = VideoPath + "/Cheetah_FPS_"+str(VideoFPS)+".mp4"
    VideoSize = (720, 1280)
    VideoFourcc = cv2.VideoWriter_fourcc(*"X264")
    VideoWriter = cv2.VideoWriter('output.mp4', 0x00000021, VideoFPS, VideoSize)
    for FrameIndx, VideoFrames in enumerate(os.listdir(ImgPath)):
        if ( FrameIndx % (240 / VideoFPS) == 0 ):
            ImgSinglePath = ImgPath + VideoFrames
            img = cv2.imread(ImgSinglePath)
            # cv2.imshow('1',img)
            # cv2.waitKey(50)
            VideoWriter.write(img)
        if ( FrameIndx > 2400 ):
            break

# def bag2txt(bag_path, out_path):
#     b = rosbag.Bag(bag_path)
#     for i, (topic, msgs, t) in enumerate(b.read_messages(topics=['/cam0/events'])):
#         with open(os.path.join(out_path, 'dvs_%04d' % (i+1)), 'w') as f:
#             f.write('%d,%d\n' % (msgs.width, msgs.height))
#             for e in msgs.events:
#                 f.write('%d,%d,%d,%s\n' % (e.ts.nsecs, e.x, e.y, e.polarity))
#     b.close()

# def Get_DVS_Data():
#     sets = os.listdir(RootPath)
#     rate = 240
#     for s in sets:
#         system_bash = "%s %s %d" % (bash_path, os.path.join(RootPath, s), rate)
#         print(system_bash)
#         os.system(system_bash)
#         time.sleep(10)
#         bag2txt(tmp_path, os.path.join(root, s))

if __name__ == '__main__':
    pass
    # Read_Deep_Video()
    Write_Deep_Video()
    # Get_DVS_Data()