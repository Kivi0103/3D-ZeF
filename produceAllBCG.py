import os

def produceBCG():
    a = os.scandir('D:\\Download\\IDM_Download\\Download\\data_2\\data\\D1_T1\\cut_video')
    for entry in a:
        if entry.is_file():
            os.system(f'python modules/detection/ExtractBackground.py --path D:/Download/IDM_Download/Download/data_2/data/D1_T1 --video_name {entry.name}')

if __name__ == '__main__':
    produceBCG()