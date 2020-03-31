import cv2
import numpy as np
import os
import scipy.io
import matplotlib.pyplot as plt
from scipy.spatial import distance

class Tmaze_detector:
    def __init__(self, input_dir, Height, Width):
        self.input_dir = input_dir
        self.detector = cv2.createBackgroundSubtractorKNN(detectShadows= False)
        self.H = Height
        self.W = Width
        self.video_path = []
        self.first_decision = []
        self.distance = []
        self.speed = []
        self.time_ratio = []
        self.T_length = 7.4 # in mm
        self.camera_shift = 0
        self.stable_th = 100
        self.center_x = self.W / 2
        self.center_y = self.H / 2
        self.floor_wall_th = 20
        self.result = []
        self.left_right = []
        self.wall = []
        if not os.path.exists(input_dir + 'result_folder'):
            os.mkdir(input_dir + 'result_folder')
    def stable_result(self, result, left_right, wall_floor):
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
                    left_right[i] = -1
                    wall_floor[i] = -1
            else:
                # calculate the distance between neighbors
                d1 = distance.euclidean((result[i - 1, 0], result[i - 1, 1]), (result[i, 0], result[i, 1]))
                d2 = distance.euclidean((result[i, 0], result[i, 1]), (result[i + 1, 0], result[i + 1, 1]))
                if d1 > self.stable_th or d2 > self.stable_th:
                    stabled[i, :] = 0
                    left_right[i] = -1
                    wall_floor[i] = -1
        return stabled, left_right, wall_floor
    def cal_time_ratio(self, mean_x, up, down, left_right_result):
        level1 = down - (1/5) * (down - up)
        level2 = down - (3/5) * (down - up)
        assert left_right_result in {0,1,2}
        if left_right_result == 1:
            return 3
        elif left_right_result == 2:
            return 4
        else:
            if mean_x < level2:
                return 2
            elif level2 <= mean_x <= level1:
                return 1
            else:
                return 0
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
    def cal_dis_speed(self, location, left_right, T_pixel, fps):
        # decision check
        idx = np.where(left_right > 0)[0]
        d = 0
        if idx.size > 0:
            first_index = idx[0]
            assert left_right[first_index] in {1,2}
            if left_right[first_index] == 1:
                first_decision = 1
            else:
                first_decision = 2
            for i in range(first_index - 1):
                d += distance.euclidean((location[i, 0], location[i, 1]), (location[i + 1, 0], location[i + 1, 1]))
            sec = (first_index + 1) / fps
            d /= (T_pixel / self.T_length)
            speed = d / sec
        else:
            first_decision = 0
            for i in range(location.shape[0] - 1):
                d += distance.euclidean((location[i, 0], location[i, 1]), (location[i + 1, 0], location[i + 1, 1]))
            sec = location.shape[0] / fps
            d /= (T_pixel / self.T_length)
            speed = d / sec
        return d, speed, first_decision

    def plot_result(self, frame, result, distance, speed, video_name, first_decision, floor_wall_stabled):
        floor_num = np.sum((floor_wall_stabled == 0))
        wall_num = np.sum((floor_wall_stabled == 1))
        floor_ratio = floor_num / (floor_num + wall_num)
        wall_ratio = 1 - floor_ratio

        plt.figure(num=None, figsize=(18, 12), dpi=144, facecolor='w', edgecolor='k')
        plt.imshow(frame)
        plt.scatter(result[:, 1], result[:, 0], marker='o', s = 12, color='r', label = 'Tracking curve')
        plt.text(0.8 * self.W, 0.8 * self.H, 'floor ratio: ' + str(round(floor_ratio, 2)) + '\n' +
                 'wall ratio: ' + str(round(wall_ratio, 2)) + '\n',
                 fontsize=20)

        plt.plot(result[:, 1], result[:, 0], color='r')
        assert first_decision in {0, 1, 2}
        if first_decision == 0:
            decision_str = 'No decision'
        elif first_decision == 1:
            decision_str = 'Left decision'
        else:
            decision_str = 'Right decision'

        title_str = decision_str + ' with Distance = ' + str(round(distance, 4)) + ' mm, Speed = ' + str(round(speed, 4)) + ' mm/s'
        plt.title(title_str, fontsize=20)
        plt.legend(fontsize=20)
        plt.savefig(self.input_dir + 'result_folder/' + video_name + '_result.png', dpi='figure', bbox_inches="tight")
        # plt.show()
        plt.close('all')
    def segmentation(self, frame, th):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)
        kernel = np.ones((7, 7), np.uint8)
        dilate = cv2.dilate(thresh, kernel, iterations=10)
        kernel = np.ones((3, 3), np.uint8)
        erode = cv2.erode(dilate, kernel, iterations=10)
        contours, hierarchy = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            print('Contours no found!')
            return [],[],[],[],[]
        else:
            centers = np.zeros([len(contours), 2])
            max_centerness = np.iinfo(np.int16).max
            max_centerness_arg = 0
            for c in range(len(contours)):
                con = contours[c].squeeze(1)
                indices = np.argsort(con[:, 0])
                left_indices = indices[0:20]
                right_indices = indices[-20:]
                left = np.mean(con[left_indices, :], axis=0).astype(int)
                right = np.mean(con[right_indices, :], axis=0).astype(int)
                if (right[0] - left[0]) < self.W / 3:
                    continue
                else:
                    centers[c,:] = np.mean(contours[c].squeeze(1), axis=0)
                    centerness = distance.euclidean((centers[c,:][0], centers[c,:][1]), (self.W/2, self.H/2))
                    # print(centerness)
                    if centerness < max_centerness:
                        max_centerness_arg = c
                        max_centerness = centerness
        # The first dimension in contour is corresponding to the Width
        # The Second dimension in contour is corresponding to the Height
            if max_centerness < np.iinfo(np.int16).max:
                contours[max_centerness_arg] -= np.array([self.camera_shift, 0])
                con = contours[max_centerness_arg].squeeze(1)
                indices = np.argsort(con[:, 0])
                left_indices = indices[0:50]
                right_indices = indices[-50:]
                left = np.mean(con[left_indices,:], axis = 0).astype(int)
                right = np.mean(con[right_indices,:], axis = 0).astype(int)
                con_return = np.expand_dims(contours[max_centerness_arg], axis=0)
                indices = np.argsort(con[:, 1])
                up_indices = indices[0:50]
                down_indices = indices[-50:]
                up = np.mean(con[up_indices,:], axis = 0).astype(int)
                down = np.mean(con[down_indices,:], axis = 0).astype(int)
                return con_return, left, right, up, down
            else:
                print('No contours pass the sanity check!')
                return [], [], [], [], []
    def floor_wall(self, point, contour, threshold):
        contour = contour.squeeze(0)
        contour = contour.squeeze(1)
        # zero means on the floor, one means on the wall
        dis = cv2.pointPolygonTest(contour, point, measureDist = True)
        if dis < 0:
            obj = 1
            # if the worm is not inside the polygon, we believe it is on the Wall
        else:
            if dis < threshold:
                obj = 1
            else:
                obj = 0
        return obj

    def process(self):
        video_count = 0
        for file in os.listdir(self.input_dir):
            if file.endswith('.avi'):
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
                left_right = -1 * np.ones([total_frame])
                # -1 means NaN, 0 means in the middle, 1 means turn left, 2 means turn right
                floor_wall = -1 * np.ones([total_frame])
                # -1 means NaN, 0 means on the Floor, 1 means on the Wall
                time_count = np.zeros([5])
                if not capture.isOpened:
                    print('Unable to open: ' + video_path)
                    exit(0)
                last_mask = np.zeros([self.H, self.W])
                count = -1
                one = 0
                while True:
                    ret, frame = capture.read()
                    if frame is None:
                        break
                    count += 1
                    con, left, right, up, down = self.segmentation(frame, 210)
                    if con == []:
                        con, left, right, up, down = self.segmentation(frame, 170)
                    if left != [] and one == 0:
                        one = 1
                        firstframe = frame
                        T_pixel = right[0] - left[0]
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

                    if Response > 50 and Response < 10000:
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
                            if con != []:
                                cv2.drawContours(frame, con, -1, (128, 0, 128), cv2.FILLED)
                            cv2.drawContours(frame, sort, 0, (0, 255, 0), 3)
                        else:
                            mean_x = int(np.mean(x))
                            mean_y = int(np.mean(y))
                            if con != []:
                                cv2.drawContours(frame, con, -1, (128, 0, 128), cv2.FILLED)
                            cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
                        result[count, :] = [mean_x, mean_y]
                        # print(con.shape)
                        if con != []:
                            floor_wall[count] = self.floor_wall((mean_y, mean_x), con, self.floor_wall_th)
                            if floor_wall[count] == 0:
                                cv2.putText(frame, 'Floor', (int(0.5 * self.W), int(0.15 * self.H)),
                                            cv2.FONT_HERSHEY_TRIPLEX, 3, (0, 0, 0))
                            else:
                                cv2.putText(frame, 'Wall', (int(0.5 * self.W), int(0.15 * self.H)),
                                            cv2.FONT_HERSHEY_TRIPLEX, 3, (0, 0, 0))
                            left_threhold = int(left[0] + (right[0] - left[0]) * 3 / 16)
                            right_threhold = int(left[0] + (right[0] - left[0]) * 13 / 16)
                            frame = cv2.line(frame,
                                             (left_threhold, int(self.H / 6)),
                                             (left_threhold, int(self.H * 5/ 6)),
                                             (255, 0, 0), 3)
                            frame = cv2.line(frame,
                                             (right_threhold, int(self.H / 6)),
                                             (right_threhold, int(self.H * 5 / 6)),
                                             (255, 0, 0), 3)

                            if mean_y < left_threhold:
                                left_right[count] = 1
                                cv2.putText(frame, 'Left', (int(0.27 * self.W), int(0.15 * self.H)),
                                            cv2.FONT_HERSHEY_TRIPLEX, 3, (0, 0, 0))
                            elif mean_y > right_threhold:
                                left_right[count] = 2
                                cv2.putText(frame, 'Right', (int(0.27 * self.W), int(0.15 * self.H)),
                                            cv2.FONT_HERSHEY_TRIPLEX, 3, (0, 0, 0))
                            else:
                                left_right[count] = 0
                                cv2.putText(frame, 'Middle', (int(0.27 * self.W), int(0.15 * self.H)),
                                            cv2.FONT_HERSHEY_TRIPLEX, 3, (0, 0, 0))
                            index = self.cal_time_ratio(mean_x, up[1], down[1], left_right[count])
                            time_count[index] += 1
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
                result_stabled, left_right_stabled, floor_wall_stabled = self.stable_result(result, left_right, floor_wall)
                for s in range(4):
                    result_stabled, left_right_stabled, floor_wall_stabled = self.stable_result(result_stabled, left_right_stabled,
                                                                                                floor_wall_stabled)
                try:
                    result_filled = self.fill_zeros(result_stabled)
                    d, speed, first_decision = self.cal_dis_speed(result_filled, left_right_stabled, T_pixel, fps)
                    self.plot_result(firstframe, result_filled, d, speed, video_name, first_decision, floor_wall_stabled)
                    self.result.append(result_filled)
                    self.left_right.append(left_right_stabled)
                    self.wall.append(floor_wall_stabled)
                    self.video_path.append(video_path)
                    self.first_decision.append(first_decision)
                    self.distance.append(d)
                    self.speed.append(speed)
                except:
                    print('No distance, speed, decision information!')
                    self.result.append(result_stabled)
                    self.left_right.append(left_right_stabled)

                    self.wall.append(floor_wall_stabled)
                    self.video_path.append(video_path)
                    pass
                cv2.destroyAllWindows()
                self.time_ratio.append(time_count / np.sum(time_count))
            if video_count % 5 == 0:
                Output = {'video path': self.video_path, 'location': self.result,
                          'left_right': self.left_right, 'wall_floor': self.wall,
                          'first_decision': self.first_decision, 'time_ratio': self.time_ratio,
                          'distance': self.distance, 'speed': self.speed
                          }
                output_path = self.input_dir + 'result_folder/' + 'output.mat'
                scipy.io.savemat(output_path, Output, do_compression= True)

input_dir = r'C:\Users\zonyul\Worm_Tracking\ok1605 x6/'
# input_dir = '/Volumes/Samsung_T5/Worm_Tracking/Experiment_downsample/'
H = 1440
W = 1920
p = Tmaze_detector(input_dir, H, W)
p.process()

input_dir = r'C:\Users\zonyul\Worm_Tracking\trp1trp2/'
# input_dir = '/Volumes/Samsung_T5/Worm_Tracking/Experiment_downsample/'
H = 1440
W = 1920
p = Tmaze_detector(input_dir, H, W)
p.process()

input_dir = r'C:\Users\zonyul\Worm_Tracking\trp1trp2\Control/'
# input_dir = '/Volumes/Samsung_T5/Worm_Tracking/Experiment_downsample/'
H = 1440
W = 1920
p = Tmaze_detector(input_dir, H, W)
p.process()
