import sys

sys.path.append("../../")
sys.path.append("../")
sys.path.append(".")
import os, sys
import argparse
import matplotlib.pyplot as plt
from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
import pyecharts.options as opts
from pyecharts.charts import Boxplot
from pyecharts.commons.utils import JsCode

sys.path.append(".")
sys.path.append('../../')
from common.utility import *


plt.rcParams['font.sans-serif'] = "Microsoft YaHei"


class BasicStatistic(object):
    def __init__(self,
                 index_data, dist_map_list=('velocity', 'dist', 'dist_cal'),
                 info_col=('exposure_time', 'DayTank', 'camNO', 'exp_time', 'filename', 'region_name')
                 ):
        self.index_data = index_data
        self.info_col = info_col
        self.column_list = index_data.columns.tolist()
        self.setting_info = {}

        # map_list = list(set(self.column_list).intersection(set(dist_map_list)))
        # if len(map_list) > 0:
        #     self.img2Word(map_list)
        print(self.index_data)

    # def img2Word(self, map_list):
    #     # 对于每一行，通过列名name访问对应的元素
    #     self.index_data.reset_index(inplace=True)
    #     for irow in self.index_data.index.tolist():
    #         DayTank = self.index_data.iloc[irow]['DayTank']
    #         region_no = self.index_data.iloc[irow]['region_name'].split("_")[0]
    #         camId = camera_id_map[self.index_data.iloc[irow]['camNO']]
    #         if f"{DayTank}_{region_no}_{str(camId)}" in self.setting_info:
    #             ratio = self.setting_info[f"{DayTank}_{region_no}_{str(camId)}"]
    #         else:
    #             cfg_path = os.path.join(root_path, DayTank)
    #             config = readConfig(cfg_path)
    #             if camId == 1:
    #                 ratio = config['Aquarium'].getfloat(f"top_ratio_{region_no}")
    #             else:
    #                 ratio = config['Aquarium'].getfloat(f"side_ratio_{region_no}")
    #             self.setting_info[f"{DayTank}_{region_no}_{str(camId)}"] = ratio
    #         scale_data = self.index_data.iloc[irow][map_list] / ratio
    #         self.index_data.loc[irow, map_list] = scale_data.copy()

    def groupbyExpData(self, stat_index=None):

        view_way = {}
        for k in self.index_data.columns.tolist():
            if k in stat_index:
                view_way[k] = stat_index[k]
        table = self.index_data.groupby(['exposure_time', 'region_name', 'DayTank']).agg(view_way)

        return table

    def formatBoxplot(self, col_data, col_name):
        col_data.reset_index(inplace=True)
        group_flag = list(set(col_data['exposure_time'].values))
        new_frame = []
        for idx, iflag in enumerate(group_flag):
            iflag_data = col_data[col_data['exposure_time'] == iflag][col_name]
            new_frame.append(iflag_data.values)

        return new_frame, group_flag

    def formatStackedBar(self, col_data, col_name):
        col_data.reset_index(inplace=True)

        group_flag = list(set(col_data['exposure_time'].values))
        mean_dict = {}
        std_dict = {}
        for idx, iflag in enumerate(group_flag):
            iflag_data = col_data[col_data['exposure_time'] == iflag][col_name]
            for angle in col_name:
                if angle not in mean_dict:
                    mean_dict[angle] = []
                if angle not in std_dict:
                    std_dict[angle] = []
                mean_dict[angle].append(iflag_data[[angle]].mean().values[0])
                std_dict[angle].append(iflag_data[[angle]].std().values[0])

        return mean_dict, std_dict, group_flag

def drawTimeBoxes(box_data, labels, axis_name, title):
    # Random test data

    fig, ax1 = plt.subplots(figsize=(9, 7), constrained_layout=True)

    # rectangular box plot
    ax1.boxplot(box_data,
                vert=True,  # vertical box alignment
                patch_artist=True,  # fill with color
                labels=labels)  # will be used to label x-ticks
    ax1.set_title(drag_name + ":" + title)
    ax1.yaxis.grid(True)
    ax1.set_xlabel('exposure time')
    ax1.set_ylabel(axis_name)
    plt.savefig(os.path.join(plt_save_path, title + ".png"), dpi=300)  # 保存图片
    # plt.show()


def drawStackedBar(labels, mean_dict, std_dict, column_names, title):
    labels = [_+1 for _ in labels]
    fig, ax = plt.subplots(figsize=(6,6), dpi=300)
    for idx, iindex in enumerate(column_names):
        imeans = []
        istd = []
        for im, iistd in zip(mean_dict[iindex], std_dict[iindex]):
            imeans.append(im * 100)
            if im - iistd >= 0:
                istd.append(iistd * 100)
            else:
                istd.append(im * 100)

        label_name = f"{iindex.split('_')[-2:][0]}°-{iindex.split('_')[-2:][1]}°".replace("91", "90")

        if idx == 0:
            ax.bar(labels, imeans, yerr=istd, label=label_name)
        else:
            last_mean_value = mean_dict[column_names[idx - 1]]
            ax.bar(
                labels, imeans, yerr=istd,
                label=label_name, bottom=last_mean_value
            )

    ax.set_xticks(labels)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Time (every 1 hour)')

    ax.set_yticks([_ * 10 for _ in range(11)])
    ax.set_yticklabels([_ * 10 for _ in range(11)])
    ax.set_ylabel(f'{title} (%)')

    ax.set_title(drag_name)
    ax.legend()
    fig.subplots_adjust(bottom=0.2)
    plt.savefig(os.path.join(plt_save_path, title + ".png"))  # 保存图片
    # plt.show()




def drawPieBar(angle_0_30, angle_30_90, angle_30_40, angle_40_50, angle_50_60, angle_60_90, expT):
    # make figure and assign axis objects
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))
    fig.subplots_adjust(wspace=0)

    # pie chart parameters
    # ratio = [0.9596, 0.02197]
    ratios = [angle_30_90, angle_0_30]
    labels = ['30°- 90°', '0°- 30°']
    explode = [0, 0.1]
    # rotate so that first wedge is split by the x-axis
    angle = -180 * ratios[0]
    ax1.pie(
        ratios, autopct='%1.3f%%',
        startangle=angle,
        labels=labels,
        labeldistance=1.1,
        explode=explode
    )

    # bar chart parameters

    xpos = 0
    bottom = 0
    # ratios = [30_40, 40_50, 50_60, 60_90]
    ratios = [angle_30_40, angle_40_50, angle_50_60, angle_60_90]
    width = .2
    colors = [[.1, .3, .3], [.1, .3, .7], [.1, .3, .5], [.1, .3, .9]]

    for j in range(len(ratios)):
        height = ratios[j]
        ax2.bar(xpos, height, width, bottom=bottom, color=colors[j])
        ypos = bottom + ax2.patches[j].get_height() / 2
        bottom += height
        ax2.text(xpos, ypos, "%1.3f%%" % (ax2.patches[j].get_height() * 100),
                 ha='center')

    ax2.set_title('Angle change ratio (%)')
    ax2.legend(('30°- 40°', '40°- 50°', '50°- 60°', 'Over 60°'))
    ax2.axis('off')
    ax2.set_xlim(- 2.5 * width, 2.5 * width)

    # use ConnectionPatch to draw lines between the two plots
    # get the wedge data
    theta1, theta2 = ax1.patches[0].theta1, ax1.patches[0].theta2
    center, r = ax1.patches[0].center, ax1.patches[0].r
    bar_height = sum([item.get_height() for item in ax2.patches])

    # draw top connecting line
    x = r * np.cos(np.pi / 180 * theta2) + center[0]
    y = r * np.sin(np.pi / 180 * theta2) + center[1]
    con = ConnectionPatch(xyA=(-width / 2, bar_height), coordsA=ax2.transData,
                          xyB=(x, y), coordsB=ax1.transData)
    con.set_color([0, 0, 0])
    con.set_linewidth(4)
    ax2.add_artist(con)

    # draw bottom connecting line
    x = r * np.cos(np.pi / 180 * theta1) + center[0]
    y = r * np.sin(np.pi / 180 * theta1) + center[1]
    con = ConnectionPatch(xyA=(-width / 2, 0), coordsA=ax2.transData,
                          xyB=(x, y), coordsB=ax1.transData)
    con.set_color([0, 0, 0])
    ax2.add_artist(con)
    con.set_linewidth(4)

    # plt.show()
    plt.savefig(os.path.join(plt_save_path, str(expT) + "_body_angle.png"), dpi=300)


def drawBar(mean_data, std_data, label, column_names, title):
    fig, ax = plt.subplots(figsize=(6,6), dpi=300)

    # Example data
    y_pos = [_*100 for _ in mean_data[column_names[0]]]
    error = [_*100 for _ in std_data[column_names[0]]]
    x_label = [_+1 for _ in label]

    ax.bar(x_label, y_pos, yerr=error, align='center')

    ax.set_xticks(x_label)
    ax.set_xticklabels(x_label)
    ax.set_yticks([_*10 for _ in range(11)])
    ax.set_yticklabels([_*10 for _ in range(11)])
    ax.set_xlabel('Time (every 1 hour)')
    ax.set_ylabel(f'{title} (%)')
    ax.set_title(drag_name)
    fig.subplots_adjust(bottom=0.2)
    plt.show()

if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()

    ap.add_argument("-rp", "--root_path", default="E:\\data\\3D_pre\\")
    ap.add_argument("-drag", "--drag_name", default="CK", type=str)
    ap.add_argument("--exposureT", default="0,1,2,3,4,5,6,7,8,9,10,11", type=str)
    # ap.add_argument("--exposureT", default="0", type=str)

    args = vars(ap.parse_args())

    root_path = args['root_path']
    drag_name = args['drag_name']
    exp_time = args['exposureT'].split(",")

    turning_angle_interval = (0, 10, 20, 30, 40, 50, 60, 70, 80, 91)
    up_down_angle_interval = (0, 15, 45, 91)

    # 单指标多时间图存储路径
    plt_save_path = os.path.join(root_path, 'drag_result', 'figure', drag_name)
    if not os.path.exists(plt_save_path):
        os.makedirs(plt_save_path)

    # 单指标多时间图存储路径
    integration_save_path = os.path.join(root_path, 'drag_result', 'multiple', drag_name)
    if not os.path.exists(integration_save_path):
        os.makedirs(integration_save_path)

    # 原始数据读入路径
    index_path = os.path.join(root_path, 'drag_result', 'single', drag_name)

    info_col = ['exposure_time', 'DayTank', 'camNO', 'exp_time', 'filename', 'region_name']

    side_col = info_col + ['top_time', 'up_down_angle']
    side_angle_col = []
    if 'up_down_angle' in side_col:
        side_col.remove('up_down_angle')
        for i in range(len(up_down_angle_interval) - 1):
            low_bound = up_down_angle_interval[i]
            up_bound = up_down_angle_interval[i + 1]
            side_angle_col.append(f"up_down_angle_{low_bound}_{up_bound}")
    side_col = side_col + side_angle_col

    top_col = info_col + ['dist', 'velocity', 'dist_cal', 'turning_angle']
    top_angle_col = []
    if 'turning_angle' in top_col:
        top_col.remove('turning_angle')
        for i in range(len(turning_angle_interval) - 1):
            low_bound = turning_angle_interval[i]
            up_bound = turning_angle_interval[i + 1]
            top_angle_col.append(f"turning_angle_{low_bound}_{up_bound}")
    top_col = top_col + top_angle_col

    all_col = side_col + top_col
    statistic_way = {}

    for icol in all_col:
        if 'turning_angle' in icol:
            statistic_way[icol] = 'mean'
        elif 'dist' in icol:
            statistic_way[icol] = 'sum'
        elif 'velocity' == icol:
            statistic_way[icol] = 'mean'
        elif 'top_time' == icol:
            statistic_way[icol] = 'mean'
        elif 'up_down_angle' in icol:
            statistic_way[icol] = 'mean'
        else:
            continue

    top_result_data = pd.DataFrame()
    side_result_data = pd.DataFrame()

    for ifile in exp_time:
        tindex_data = pd.read_csv(os.path.join(index_path, ifile + "_top.csv"))
        sindex_data = pd.read_csv(os.path.join(index_path, ifile + "_side.csv"))

        top_index_data = tindex_data.copy()[top_col]
        top_index_data.dropna(axis=0, how='any', inplace=True)
        side_index_data = sindex_data.copy()[side_col]
        side_index_data.dropna(axis=0, how='any', inplace=True)

        top_result_data = pd.concat([top_result_data, top_index_data])
        side_result_data = pd.concat([side_result_data, side_index_data])

    Tanalyzer = BasicStatistic(top_result_data)
    Sanalyzer = BasicStatistic(side_result_data)
    Tmean_data = Tanalyzer.groupbyExpData(stat_index=statistic_way)
    Smean_data = Sanalyzer.groupbyExpData(stat_index=statistic_way)

    Tmean_data.to_csv(os.path.join(integration_save_path, 'top_view.csv'), sep=',')
    Smean_data.to_csv(os.path.join(integration_save_path, 'side_view.csv'), sep=',')

    # # # 绘制速度箱线图
    # box_plot_data, label = Tanalyzer.formatBoxplot(Tmean_data[['velocity']], 'velocity')
    # drawTimeBoxes(box_plot_data, label, 'velocity', 'velocity')

    # # 绘制运动路径箱线图
    # box_plot_data, label = Tanalyzer.formatBoxplot(Tmean_data[['dist']], 'dist')
    # drawTimeBoxes(box_plot_data, label, 'dist', 'dist')

    # # 绘制运动路径箱线图
    # box_plot_data, label = Tanalyzer.formatBoxplot(Tmean_data[['dist_cal']], 'dist_cal')
    # drawTimeBoxes(box_plot_data, label, 'dist_cal', 'dist_cal')

    # ====================================================================================== #
    # # 上下浮动的角度 stack bar图
    column_names = side_angle_col
    mean_data, std_data, label = Sanalyzer.formatStackedBar(
        Smean_data[column_names], column_names
    )
    drawStackedBar(label, mean_data, std_data, column_names, 'Ratio of up down angle')

    # 绘制转向角的角度 stack bar图

    new_col_names = ['turning_angle_0_30', 'turning_angle_30_90', 'turning_angle_60_90']
    Tmean_data['turning_angle_0_30'] = Tmean_data['turning_angle_0_10'] + Tmean_data['turning_angle_10_20'] + \
                                       Tmean_data['turning_angle_20_30']
    Tmean_data['turning_angle_30_90'] = 1 - Tmean_data['turning_angle_0_30']

    Tmean_data['turning_angle_60_90'] = Tmean_data['turning_angle_30_90'] - \
                                        Tmean_data['turning_angle_30_40'] - \
                                        Tmean_data['turning_angle_40_50'] - \
                                        Tmean_data['turning_angle_50_60']

    column_names = top_angle_col + new_col_names
    for i in exp_time:
        mean_dict, _, group_flag = Tanalyzer.formatStackedBar(
            Tmean_data[column_names], column_names
        )
        angle_0_30 = mean_dict['turning_angle_0_30'][int(i)]
        angle_30_90 = mean_dict['turning_angle_30_90'][int(i)]

        angle_30_40 = mean_dict['turning_angle_30_40'][int(i)]
        angle_40_50 = mean_dict['turning_angle_40_50'][int(i)]
        angle_50_60 = mean_dict['turning_angle_50_60'][int(i)]
        angle_60_90 = mean_dict['turning_angle_60_90'][int(i)]
        drawPieBar(angle_0_30, angle_30_90, angle_30_40, angle_40_50, angle_50_60, angle_60_90, i)

    # # # 绘制柱状图
    # column_names = ['top_time']
    # mean_data, std_data, label = Sanalyzer.formatStackedBar(
    #     Smean_data[column_names], column_names
    # )
    # drawBar(mean_data, std_data, label, column_names, 'percent')
