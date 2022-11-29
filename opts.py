import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', default='0', type=str, help='Directory path of video')
    args = parser.parse_args()

    return args