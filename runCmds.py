import sys
sys.path.append(".")
from common.utility import *

def runCmds(video_name):
    references = {'ch01': [2, 4], 'ch02': [1, 2, 3, 4], 'ch03': [1, 3]}
    camNO = {'ch01': 'cam3', 'ch02': 'cam1', 'ch03': 'cam2'}
    drag_name = '6PPD1ppm'
    # 从视频文件名中取出尾号
    endNO = video_name[-8:-4]
    cmds = []

    # 生成执行detect_side.py的命令
    for index in references[endNO]:
        cmd = f'python modules/yolo5/detect_side.py ' \
              f'--weights F:/3D-ZeF_Data/data/best.pt ' \
              f'--config_path F:/3D-ZeF_Data/data/D1_T1 ' \
              f'--region_name {index}_6PPD1ppm ' \
              f'--source F:/3D-ZeF_Data/data/D1_T1/cut_video/{video_name} ' \
              f'--device 0 ' \
              f'--project F:/3D-ZeF_Data/data/D1_T1/processed/{index}_6PPD1ppm ' \
              f'--img 640'
        cmds.append(cmd)

    if not os.path.exists(f'F:\\3D-ZeF_Data\\data\\D1_T1\\background\\{video_name[:-8]}{camNO[endNO]}.jpg'):
        cmds.append(f'python modules/detection/ExtractBackground.py --path F:/3D-ZeF_Data/data/D1_T1 --video_name {video_name}')

    cmds.append('python modules/detection/meanBackground.py --path F:/3D-ZeF_Data/data/D1_T1')

    # 生成执行BgDetector.py的命令
    for index in references[endNO]:
        cmd = f'python modules/detection/BgDetector.py ' \
              f'-f F:/3D-ZeF_Data/data/D1_T1 ' \
              f'--region_name {index}_6PPD1ppm ' \
              f'--video_name {video_name}'
        cmds.append(cmd)

    # 生成执行sortTracker.py的命令
    for index in references[endNO]:
        cmd = f'python modules/tracking/sortTracker.py ' \
              f'--root_path F:/3D-ZeF_Data/data/ ' \
              f'--DayTank D1_T1 ' \
              f'--RegionName {index}_6PPD1ppm ' \
              f'--detection_filename {video_name[:-4]}.csv'
        cmds.append(cmd)

    # 生成执行FinalizeTracks.py的命令
    for index in references[endNO]:
        cmd = f'python modules/tracking/FinalizeTracks.py ' \
              f'--root_path F:/3D-ZeF_Data/data/ ' \
              f'--DayTank D1_T1 ' \
              f'--tracker sortTracker ' \
              f'--RegionName {index}_6PPD1ppm ' \
              f'--track_file {video_name[:-4]}.csv'
        cmds.append(cmd)

    # 生成执行indexAnalysis.py的命令
    cmds.append(f'python modules/analysis/indexAnalysis.py -rp F:/3D-ZeF_Data/data/ -drag {drag_name} -epT 0 -trker finalTrack')

    print(cmds)
    # for cmd in cmds:
    #     try:
    #         print("命令%s开始运行%s" % (cmd, datetime.datetime.now()))
    #         os.system(cmd)
    #         print("命令%s结束运行%s" % (cmd, datetime.datetime.now()))
    #     except:
    #         print('%s\t 运行失败' % (cmd))

if __name__ == '__main__':
    runCmds('2021_10_11_21_49_59_ch01.avi')

