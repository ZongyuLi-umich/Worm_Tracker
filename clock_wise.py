import cv2
import numpy as np
import os
from scipy.spatial import distance
from scipy.optimize import curve_fit
import scipy.io
import matplotlib.pyplot as plt
class clock_detect:
    def __init__(self, input_dir, Height, Width):
        self.input_dir = input_dir
        self.detector = cv2.createBackgroundSubtractorKNN(detectShadows= False)
        self.H = Height
        self.W = Width
        self.video_path = []
        self.clock_ratio = []
        self.camera_shift_th = 20
        self.center_x = self.W / 2
        self.center_y = self.H / 2
        self.result = []
        if not os.path.exists(input_dir + 'result_folder'):
            os.mkdir(input_dir + 'result_folder')

    def fill_zeros(self, x):
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
    def clock_wise_detect(self, result):
        num = result.shape[0]
        downratio = 5
        # meansample = np.zeros([num , 2])
        # for i in range(2, num - 2):
        #     meansample[i,:] = result[i-2:i+2,:].mean()
        # meansample[0:2,:] = result[0:2,:]
        # meansample[-2:,:] = result[-2:,:]
        downsample = result[0::downratio, :]
        mask = -2 * np.ones([downsample.shape[0]], dtype= int)
        upsample = -2 * np.ones([num])
        for i in range(1, downsample.shape[0] - 1):
            v_12x = downsample[i, 0] - downsample[i-1, 0]
            v_23x = downsample[i+1, 0] - downsample[i, 0]
            v_12y = downsample[i, 1] - downsample[i-1, 1]
            v_23y = downsample[i+1, 1] - downsample[i, 1]
            outer_product = v_12x * v_23y - v_12y * v_23x
            # angle = np.arcsin(outer_product / (np.sqrt(v_12x^2 + v_12y^2)*(v_23x^2 + v_23y^2)))
            # print(outer_product)
            if outer_product > 0:
                mask[i] = 1
            elif np.abs(outer_product) < 1e-2:
                mask[i] = 0
            else:
                mask[i] = -1
        mask[0] = mask[1]
        mask[-1] = mask[-2]
        # for i in range(2, mask.shape[0] - 2):
        #     counts = np.bincount(mask[i-2:i+2])
        #     mode = np.argmax(counts)
        #     if mask[i] != mode:
        #         mask[i] = mode
        for i in range(num):
            upsample[i] = mask[min(round(i/downratio), mask.shape[0] - 1)]
        return upsample

    def plot_result(self, frame, result, video_name):
        # def func(x, a, b, c, d, e):
        #     return a*(x**5) + b*(x**4) + c*(x**3) + d*(x**2) + e
        # print(np.nonzero(result_fill))
        num = result.shape[0]
        for i in range(1, num):
            vec = [result[i, 0] - result[i-1, 0], result[i, 1] - result[i-1, 1]]
            d = np.linalg.norm(vec)
            # print(d)
            if  d > self.camera_shift_th:
                result[i:,:] -= vec
        result_mean = result.mean(axis = 0)
        shift_vec = [self.H / 2 - result_mean[0], self.W / 2 - result_mean[1]]
        result += shift_vec
        t = np.arange(0, result.shape[0]) / 20
        px = np.polyfit(t, result[:, 0], 10)
        py = np.polyfit(t, result[:, 1], 10)
        # popt_x, pcov_x = curve_fit(func, t, result[:, 0])
        # popt_y, pcov_y = curve_fit(func, t, result[:, 1])
        func_x = np.poly1d(px)
        func_y = np.poly1d(py)
        est_x = func_x(t)
        est_y = func_y(t)
        # print(est_x.shape)
        est_curve = np.zeros([est_x.shape[0], 2])
        est_curve[:, 0] = est_x
        est_curve[:, 1] = est_y
        clock_result = self.clock_wise_detect(est_curve)
        clock = result[(clock_result == 1), :]
        straight = result[(clock_result == 0), :]
        counter_clock = result[(clock_result == -1), :]
        num_clock = clock.shape[0]
        num_straight = straight.shape[0]
        num_counter_clock = counter_clock.shape[0]
        ratio = [num_clock / num, num_straight / num, num_counter_clock / num]
        plt.figure(num=None, figsize=(24, 18), dpi=144, facecolor='w', edgecolor='k')
        plt.imshow(frame)
        # plt.scatter(est_curve[:,1], est_curve[:,0], color = 'b')
        # plt.scatter(result_fill[:,1], result_fill[:,0], color = 'r')
        # plt.scatter(pcov[:,1], pcov[:,0], color = 'b')
        plt.text(result[0, 1], result[0, 0], 'start point', fontsize=20, bbox=dict(facecolor='red', alpha=0.5))
        plt.text(result[-1, 1], result[-1, 0], 'end point', fontsize=20, bbox=dict(facecolor='red', alpha=0.5))
        plt.text(0.8 * self.W, 0.8 * self.H, 'clock ratio: '+ str(round(ratio[0], 2)) + '\n' +
                 'straight ratio: '+ str(round(ratio[1], 2)) + '\n' +
                 'counter clock ratio: ' + str(round(ratio[2], 2)) + '\n',
                 fontsize=20)

        plt.scatter(clock[:, 1], clock[:, 0], color='r', label = 'counter clock', s = 12)
        plt.scatter(straight[:, 1], straight[:, 0], color = 'g', label = 'straight', s = 12)
        plt.scatter(counter_clock[:, 1], counter_clock[:, 0], color='b', label = 'clock', s = 12)
        plt.scatter(result[0, 1], result[0, 0], label='start point', s = 60)
        plt.scatter(result[-1, 1], result[-1, 0], label='end point', s = 60)
        plt.legend(fontsize = 20)
        plt.savefig(self.input_dir + 'result_folder/' + video_name + '_result.png', dpi='figure', bbox_inches="tight")
        plt.show()
        return ratio
    def process(self):
        video_count = 0
        for file in os.listdir(self.input_dir):
            if file.endswith('.avi'):
                # file = 'Testing33.avi'
                if 'Training' in file:
                    continue
                else:
                    video_count += 1
                video_path = os.path.join(self.input_dir, file)
                # parent_path = video_path.split('/')[0:-1]
                video_name = (video_path.split('/')[-1]).split('.')[0]
                print('Now processing {}'.format(video_name))
                capture = cv2.VideoCapture(cv2.samples.findFileOrKeep(video_path))
                total_frame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = int(capture.get(cv2.CAP_PROP_FPS))
                min_HW = min(self.H, self.W)
                out = cv2.VideoWriter(self.input_dir + 'result_folder/' + video_name + '_result.avi',
                                     cv2.VideoWriter_fourcc(*'MPEG'), fps, (self.W * 2, self.H))
                result = np.zeros([total_frame, 2])
                if not capture.isOpened:
                    print('Unable to open: ' + video_path)
                    exit(0)
                last_mask = np.zeros([self.H, self.W])
                count = -1
                while True:
                    ret, frame = capture.read()
                    if frame is None:
                        break
                    count += 1
                    if count == 0:
                        firstframe = frame
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
                    # print(Response)
                    if Response > 50 and Response < 10000:
                    # if Response > 10 and Response < 10000:
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
                    cv2.resizeWindow('Frame', 640, 480)
                    cv2.imshow('Frame', frame)
                    cv2.namedWindow('FG Mask', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('FG Mask', 640, 480)
                    cv2.imshow('FG Mask', fgMask)
                    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('result', 640, 480)
                    cv2.imshow('result', np.concatenate((frame, out_mask), axis=1))
                    ## [show]

                    keyboard = cv2.waitKey(30)
                    if keyboard == 'q' or keyboard == 27:
                        break
                out.release()
                capture.release()
                result_filled = self.fill_zeros(result)
                clock_ratio = self.plot_result(firstframe, result_filled, video_name)
                cv2.destroyAllWindows()
                self.result.append(result_filled)
                self.video_path.append(video_path)
                self.clock_ratio.append(clock_ratio)
            if video_count % 5 == 0:
                Output = {'video path': self.video_path, 'location': self.result,
                          'clock ratio': self.clock_ratio}
                output_path = self.input_dir + 'result_folder/' + 'output.mat'
                scipy.io.savemat(output_path, Output)
# input_dir = '/Volumes/Samsung_T5/Worm_Tracking/OpenSurfaceTesting#2/'
# H = 1440
# W = 1920
# p = clock_detect(input_dir, H, W)
# p.process()
