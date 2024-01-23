##!/bin/bash
## dos2unix filename
## 使用vi打开文本文件
## vi dos.txt
## 命令模式下输入
## :set fileformat=unix
## :w

python /home/huangjinze/code/3D-ZeF/threading_track.py --code_path /home/huangjinze/code/3D-ZeF --data_path /home/huangjinze/code/data/zef --DayTank D1_T3
python /home/huangjinze/code/3D-ZeF/threading_finalize.py --code_path /home/huangjinze/code/3D-ZeF --data_path /home/huangjinze/code/data/zef --DayTank D1_T3
python /home/huangjinze/code/3D-ZeF/threading_video.py --code_path /home/huangjinze/code/3D-ZeF --data_path /home/huangjinze/code/data/zef --DayTank D1_T3

python /home/huangjinze/code/3D-ZeF/threading_track.py --code_path /home/huangjinze/code/3D-ZeF --data_path /home/huangjinze/code/data/zef --DayTank D2_T3
python /home/huangjinze/code/3D-ZeF/threading_finalize.py --code_path /home/huangjinze/code/3D-ZeF --data_path /home/huangjinze/code/data/zef --DayTank D2_T3
python /home/huangjinze/code/3D-ZeF/threading_video.py --code_path /home/huangjinze/code/3D-ZeF --data_path /home/huangjinze/code/data/zef --DayTank D2_T3

python /home/huangjinze/code/3D-ZeF/threading_track.py --code_path /home/huangjinze/code/3D-ZeF --data_path /home/huangjinze/code/data/zef --DayTank D3_T3
python /home/huangjinze/code/3D-ZeF/threading_finalize.py --code_path /home/huangjinze/code/3D-ZeF --data_path /home/huangjinze/code/data/zef --DayTank D3_T3
python /home/huangjinze/code/3D-ZeF/threading_video.py --code_path /home/huangjinze/code/3D-ZeF --data_path /home/huangjinze/code/data/zef --DayTank D3_T3

python /home/huangjinze/code/3D-ZeF/threading_quantify.py --code_path /home/huangjinze/code/3D-ZeF --data_path /home/huangjinze/code/data/zef --drag_names 6PPD50ppb --tracker finalTrack
pause