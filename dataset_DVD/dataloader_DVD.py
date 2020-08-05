# pkg
import os
import cv2

# Get the DeepVideo file list
DatasetPath = '/media/hmf_love_long' \
               '/My Passport/学习文件/深度学习/数据集' \
               '/DeepVideoDeblurring' \
               '/DeepVideoDeblurring_Dataset_Original_High_FPS_Videos' \
               '/original_high_fps_videos'  # TODO: change to your dataset path to convent
DeepVideoList = [ DatasetPath + '/' + i for i in os.listdir(DatasetPath)[1:-1] ] # Absolute path
DataStorePath = '/media/hmf_love_long' \
               '/My Passport/学习文件/深度学习/数据集' \
               '/DeepVideoDeblurring' \
               '/DeepVideoDeblurring_Dataset_Original_High_FPS_Videos' \
               '/processed_high_fps_videos'  # TODO: change to your dataset path to store
# print(DeepVideoList)

# Find version for opencv-python
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
print('major versiob = %s, minor versiob = %s, subminor versiob = %s' % (major_ver, minor_ver, subminor_ver))

def Read_Deep_Video():
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
            os.mkdir(DataStorePath + '/' + (VideoName.split('/'))[-1])
            # ret 读取了数据就返回True,没有读取数据(已到尾部)就返回False
            # frame 返回读取的视频数据--一帧数据
            while (True):
                ret, frame = VideoCapture.read()    # 读取一帧视频
                if ret is False:
                    break
                VideoAllFrames.append(frame)
                FrameCount = FrameCount + 1
                FileName = DataStorePath + '/' + (VideoName.split('/'))[-1] + '/' + str(FrameCount) + '.png'
                cv2.imwrite(FileName, frame)
            print('Total frame number is ', FrameCount)
        break




        else:
            print('视频文件打开失败')

if __name__ == '__main__':
    pass
    Read_Deep_Video()