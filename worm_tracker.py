from detect_Tmaze import Tmaze_detector
from clock_wise import clock_detect
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, help='The input video directory',
                    default='/Volumes/Samsung_T5/Worm_Videos/ok1605 x6/')
parser.add_argument('--H', type=int, help='The height of the video', default=1440)
parser.add_argument('--W', type=int, help='The width of the video', default=1920)
parser.add_argument('--is_clock', action='store_true', help='If set to true, then process clockwise videos')
parser.add_argument("--mode", default='client')
parser.add_argument("--port", default=50093)
args = parser.parse_args()

def main():
    print(args.is_clock)
    if args.is_clock == True:
        Video_processor = clock_detect(args.input_dir,
                                       args.H,
                                       args.W)
        Video_processor.process()
    else:
        Video_processor = Tmaze_detector(args.input_dir,
                                       args.H,
                                       args.W)
        Video_processor.process()


if __name__ == '__main__':
    main()
