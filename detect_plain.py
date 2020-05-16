import cv2
import numpy as np
import os
import scipy.io
import matplotlib.pyplot as plt
from scipy.spatial import distance

class Plain_detector:
    def __init__(self, input_dir, Height, Width):
        self.input_dir = input_dir
        self.H = Height
        self.W = Width
        self.video_name = []
        self.distance = []
        self.speed = []
        self.pixel2mm = 20
        self.stable_th = 100
        self.center_x = self.W / 2
        self.center_y = self.H / 2
        self.result = []
        if not os.path.exists(input_dir + 'result_folder'):
            os.mkdir(input_dir + 'result_folder')
    def stable_result(self, result):
        num = result.shape[0]
        stabled = result.copy()
        for i in range(4, num - 4):
            counts_x = np.bincount(result[i - 4:i + 4, 0])
            # print('result: ', stabled[i, :])
            mode_x = np.argmax(counts_x)
            counts_y = np.bincount(result[i - 4:i + 4, 1])
            mode_y = np.argmax(counts_y)
            # print('mode: ', [mode_x, mode_y])
            if mode_x == 0 or mode_y == 0:
                if stabled[i, 0] != mode_x or stabled[i, 1] != mode_y:
                    stabled[i, :] = 0
            else:
                # calculate the distance between neighbors
                d1 = distance.euclidean((result[i - 1, 0], result[i - 1, 1]), (result[i, 0], result[i, 1]))
                d2 = distance.euclidean((result[i, 0], result[i, 1]), (result[i + 1, 0], result[i + 1, 1]))
                if d1 > self.stable_th or d2 > self.stable_th:
                    stabled[i, :] = 0
        return stabled
    def fill_zeros(self, x):
        # print(x.nonzero())
        num = x.shape[0]
        y = np.sum(x, axis=1)
        z = x.copy()
        for i in range(num):
            if np.sum(np.abs(x[i, :])) == 0:
                if np.nonzero(y[:i])[0].size == 0:
                    idx = np.nonzero(y[i:])[0][0]
                    z[i, :] = x[i + idx, :]
                else:
                    idx = np.nonzero(y[:i])[0][-1]
                    z[i, :] = x[idx, :]
        return z
    def cal_dis_speed(self, location, pixel2mm, fps):
        # decision check
        d = 0
        speed = 0
        for i in range(len(location) - 1):
            d += distance.euclidean((location[i, 0], location[i, 1]), (location[i + 1, 0], location[i + 1, 1]))
            sec = (len(location) + 1) / fps
            d /= pixel2mm
            speed = d / sec

        return d, speed

    def plot_result(self, frame, result, distance, speed, video_name):

        plt.figure(num=None, figsize=(18, 12), dpi=144, facecolor='w', edgecolor='k')
        plt.imshow(frame)
        plt.scatter(result[:, 1], result[:, 0], marker='o', s = 12, color='r', label = 'Tracking curve')


        plt.plot(result[:, 1], result[:, 0], color='r')

        title_str = 'Distance = ' + str(round(distance, 4)) + ' mm, Speed = ' + str(round(speed, 4)) + ' mm/s'
        plt.title(title_str, fontsize=20)
        plt.legend(fontsize=20)
        plt.savefig(self.input_dir + 'result_folder/' + video_name + '_result.png', dpi='figure', bbox_inches="tight")
        # plt.show()
        plt.close('all')


    def process(self):
        video_count = 0
        for file in os.listdir(self.input_dir):
            if file.endswith('.avi'):
                self.detector = cv2.createBackgroundSubtractorKNN(detectShadows=False)
                # file = '/Volumes/Samsung_T5/Worm_Tracking/Plain_video/worm11_pt4.avi'
                video_count += 1
                video_path = os.path.join(self.input_dir, file)
                # parent_path = video_path.split('/')[0:-1]
                video_name = (video_path.split('/')[-1]).split('.')[0]
                print('Now processing {}'.format(video_name))
                capture = cv2.VideoCapture(cv2.samples.findFileOrKeep(video_path))
                total_frame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = int(capture.get(cv2.CAP_PROP_FPS))
                # print(fps)
                min_HW = min(self.H, self.W)
                out = cv2.VideoWriter(self.input_dir + 'result_folder/' + video_name + '_result.avi',
                                     cv2.VideoWriter_fourcc(*'MPEG'), fps, (self.W * 2, self.H))
                result = np.zeros([total_frame, 2])
                if not capture.isOpened:
                    print('Unable to open: ' + video_path)
                    exit(0)
                last_mask = np.zeros([self.H, self.W])
                count = -1
                one = 0
                while True:
                    print(count)
                    ret, frame = capture.read()
                    if frame is None:
                        break
                    count += 1
                    if one == 0:
                        firstframe = frame
                        one = 1
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    fgMask = self.detector.apply(gray_frame)
                    fgMask = cv2.GaussianBlur(fgMask, (11, 11), 0)
                    std_mask = np.std(fgMask)
                    normalized_mask = fgMask / 255.0
                    Response = np.sum(normalized_mask * last_mask)
                    last_mask = fgMask / 255.0
                    x, y = np.nonzero(fgMask)
                    if std_mask > 50:
                        self.detector = cv2.createBackgroundSubtractorKNN(detectShadows=False)
                        continue
                    print(Response)
                    if Response > 50 and Response < 1e4:
                        contours, hierarchy = cv2.findContours(fgMask,
                                                               cv2.RETR_EXTERNAL,
                                                               cv2.CHAIN_APPROX_SIMPLE)
                        if len(contours) > 1:
                            sort = sorted(contours, key=cv2.contourArea, reverse=True)
                            cmax = np.squeeze(sort[0], axis=1)
                            mean_y, mean_x = np.mean(cmax, axis=0).astype(int)
                            # mean_x correspond to the height
                            # mean_y correspond to the width
                            max_y, max_x = np.max(cmax, axis=0).astype(int)
                            min_y, min_x = np.min(cmax, axis=0).astype(int)
                            if (max_x - min_x) > min_HW / 4 or (max_y - min_y) > min_HW / 4:
                                continue
                            cv2.drawContours(frame, sort, 0, (0, 255, 0), 3)
                        else:
                            mean_x = int(np.mean(x))
                            mean_y = int(np.mean(y))
                            cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
                        result[count, :] = [mean_x, mean_y]
                        # print(con.shape)
                        cv2.circle(frame, (mean_y, mean_x), 10, (0, 0, 255), thickness=5)
                    cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
                    cv2.putText(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                    out_mask = np.transpose(np.tile(fgMask, (3, 1, 1)), (1, 2, 0))
                    out.write(np.concatenate((frame, out_mask), axis=1))
                    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('Frame', 1200, 600)
                    cv2.imshow('Frame', frame)
                    cv2.namedWindow('FG Mask', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('FG Mask', 1200, 600)
                    cv2.imshow('FG Mask', fgMask)
                    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('result', 1200, 600)
                    cv2.imshow('result', np.concatenate((frame, out_mask), axis=1))
                    ## [show]

                    keyboard = cv2.waitKey(30)
                    if keyboard == 'q' or keyboard == 27:
                        break
                out.release()
                capture.release()
                result = result.astype(int)
                # stable results for 5 times
                result_stabled = self.stable_result(result)
                for s in range(6):
                    result_stabled = self.stable_result(result_stabled)
                try:
                    result_filled = self.fill_zeros(result_stabled)
                    d, speed = self.cal_dis_speed(result_filled, self.pixel2mm, fps)
                    self.plot_result(firstframe, result_filled, d, speed, video_name)
                    self.result.append(result_filled)

                    # print(file)
                    self.video_name.append(file)

                    self.distance.append(d)
                    self.speed.append(speed)
                except:
                    print('No distance, speed information!')
                    self.result.append(result_stabled)
                    self.video_name.append(file)
                    self.distance.append([])
                    self.speed.append([])
                    pass
                cv2.destroyAllWindows()
                # print(time_count / np.sum(time_count))

            if video_count % 1 == 0:
                Output = {'video_name': self.video_name, 'location': self.result,
                          'distance': self.distance, 'speed': self.speed
                          }
                output_path = self.input_dir + 'result_folder/' + 'output.mat'
                scipy.io.savemat(output_path, Output, do_compression= True)



# input_dir = r'C:\Users\zonyul\Worm_Tracking\trp1trp2/'
# # input_dir = '/Volumes/Samsung_T5/Worm_Tracking/Experiment_downsample/'
# H = 1440
# W = 1920
# p = Tmaze_detector(input_dir, H, W)
# p.process(th1= 170, th2= 200)

# input_dir = r'C:\Users\zonyul\Worm_Tracking\trp1trp2\Control/'
input_dir = '/Volumes/Samsung_T5/Worm_Tracking/Plain_video/'
H = 1440
W = 1920
p = Plain_detector(input_dir, H, W)
p.process()

# input_dir = r'C:\Users\zonyul\Worm_Tracking\ok1605 x6/'
# # input_dir = '/Volumes/Samsung_T5/Worm_Tracking/Experiment_downsample/'
# H = 1440
# W = 1920
# p = Tmaze_detector(input_dir, H, W)
# p.process(th1= 210, th2= 170)