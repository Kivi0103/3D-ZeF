import pyecharts.options as opts
from pyecharts.charts import Boxplot
from pyecharts.commons.utils import JsCode
import pandas as pd
import os, sys
import argparse
from pyecharts.charts import Bar
from pyecharts.commons.utils import JsCode
from pyecharts.globals import ThemeType
from scipy import stats
import pyecharts.options as opts
from pyecharts.faker import Faker
from pyecharts.charts import Grid, Boxplot, Scatter, Bar, Line
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(".")
sys.path.append('../')
sys.path.append('../../')
from common.utility import *


# =================== echarts boxplot  ===========================#
def formatBoxData(data, index_name, exposureT_list, category_name):
    # 种类，重复试验，对应MultipleBox中的Series
    series_names = list(data[category_name].unique())
    drag_data = {}
    for drag in series_names:
        drag_data[drag] = {
            'inliers': [],
            'outliers': [],
        }
        for iexp_t in exposureT_list:
            exp_data = data[(data['exposure_time'] == int(iexp_t)) & (data['drag_name'] == drag)][[index_name]]
            std_3 = (exp_data[index_name] - exp_data[index_name].mean()) / exp_data[index_name].std()
            inliers = exp_data[std_3.abs() < 2]
            outliers = exp_data[~(std_3.abs() < 2)]

            # q1, q3 = exp_data[index_name].quantile([0.25, 0.75])
            # iqr = q3 - q1
            # inliers = exp_data[(exp_data[index_name] < iqr * 1.5 + q3) & (exp_data[index_name] > q1 - iqr * 1.5)]
            # outliers = exp_data[(exp_data[index_name] >= iqr * 1.5 + q3) | (exp_data[index_name] <= q1 - iqr * 1.5)]

            # df_Z = exp_data[(np.abs(stats.zscore(exp_data[index_name])) < 1.2)]
            # ix_keep = df_Z.index
            # inliers = exp_data.loc[ix_keep]
            #
            # df_Z = exp_data[(np.abs(stats.zscore(exp_data[index_name])) >= 1.2)]
            # ix_keep = df_Z.index
            # outliers = exp_data.loc[ix_keep]

            drag_data[drag]['inliers'].append(inliers[index_name].tolist())
            drag_data[drag]['outliers'].append(outliers[index_name].tolist())

    return drag_data


def drawMultipleBox(exposure_T, drag_data, index_info):
    box_plot = Boxplot()

    box_plot.add_xaxis(xaxis_data=[f"{str(int(_) + 1)}" for _ in exposure_T])
    box_plot.set_global_opts(
        title_opts=opts.TitleOpts(
            pos_left="left", title='', item_gap=20
        ),
        toolbox_opts=opts.ToolboxOpts(
            feature=opts.ToolBoxFeatureOpts(
                save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(
                    type_='jpeg',
                    background_color='white'
                )
            )
        ),

        legend_opts=opts.LegendOpts(

            textstyle_opts=opts.TextStyleOpts(
                font_family='Microsoft YaHei',
                font_style='normal',
                font_size=15,
                font_weight='bold'
            ),
        ),
        tooltip_opts=opts.TooltipOpts(
            trigger="item", axis_pointer_type="shadow"
        ),
        xaxis_opts=opts.AxisOpts(
            type_="category",
            boundary_gap=True,
            splitarea_opts=opts.SplitAreaOpts(
                areastyle_opts=opts.AreaStyleOpts(opacity=1)
            ),
            name_textstyle_opts=opts.TextStyleOpts(
                font_family='Microsoft YaHei',
                font_style='normal',
                font_size=15,
                font_weight='bold'
            ),
            # axislabel_opts=opts.LabelOpts(formatter="expr {value}"),
            name='Time (every 1 hour)',
            name_location='middle',
            name_gap=45,
            splitline_opts=opts.SplitLineOpts(is_show=False),
        ),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            name=f"{index_info['name']} ({index_info['unit']})",
            name_location='middle',
            name_gap=45,
            name_rotate=90,
            name_textstyle_opts=opts.TextStyleOpts(
                font_family='Microsoft YaHei',
                font_style='normal',
                font_size=15,
                font_weight='bold'
            )
        ),

    )
    for dname, y_data in drag_data.items():
        box_plot.add_yaxis(
            series_name=dname,
            y_axis=box_plot.prepare_data(y_data['inliers']),
            itemstyle_opts=opts.ItemStyleOpts(
                border_color0='black',
                opacity=0.8,
                border_width=3,
                color=box_plot.colors[list(drag_data.keys()).index(dname)]
            )
        )
    # scatter = Scatter()
    # scatter.set_global_opts(
    #     legend_opts=opts.LegendOpts(
    #         textstyle_opts=opts.TextStyleOpts(
    #             font_family='Microsoft YaHei',
    #             font_style='normal',
    #             font_size=15,
    #             font_weight='bold'
    #         )
    #     ),
    #     xaxis_opts=opts.AxisOpts(
    #         name_textstyle_opts=opts.TextStyleOpts(font_family='Microsoft YaHei'),
    #     ),
    # )
    # scatter.add_xaxis(
    #     [f"{str(int(_)+1)}" for _ in exposure_T]
    # )
    # for dname, y_data in drag_data.items():
    #     scatter.add_yaxis(
    #         series_name=dname, y_axis=y_data['outliers'], label_opts=opts.LabelOpts(is_show=False)
    #     )

    grid = (
        Grid(init_opts=opts.InitOpts(width="1560px", height="720px"))
            .add(
            box_plot,
            grid_opts=opts.GridOpts(pos_left="10%", pos_right="10%", pos_bottom="15%"),
        )
            # .add(
            # scatter,
            # grid_opts=opts.GridOpts(
            #     pos_left="10%", pos_right="10%", pos_bottom="15%"
            # ),
            # )
            .render(os.path.join(out_path, f"{'_'.join(drag_list)}_{index_info['name']}.html"))
    )


# =================== echarts barplot  ===========================#
def formatBarData(data, index_name, exposureT_list, category_name):
    # 种类，重复试验，对应MultipleBox中的Series
    new_index_col = [
        # 'turning_angle_0_30',
        # 'turning_angle_30_60',
        # 'turning_angle_60_90'
        'turning_angle_30_90'
    ]
    # data['turning_angle_0_30'] = data['turning_angle_0_10'] + data['turning_angle_10_20'] + data['turning_angle_20_30']
    data['turning_angle_30_90'] = data['turning_angle_30_40'] + data['turning_angle_40_50'] + data[
        'turning_angle_50_60'] + data['turning_angle_60_70'] + data['turning_angle_70_80'] + data['turning_angle_80_91']
    # data['turning_angle_60_90'] = data['turning_angle_60_70'] + data['turning_angle_70_80'] + data['turning_angle_80_91']
    drag_names = list(data[category_name].unique())
    drag_data = {}
    mean_data = data.groupby(['exposure_time', 'drag_name'])[new_index_col].mean()
    mean_data.reset_index(inplace=True)
    for idrag in drag_names:
        drag_data_cond = (mean_data['drag_name'] == idrag) & (mean_data['exposure_time'].isin(exposureT_list))
        drag_data[idrag] = mean_data[drag_data_cond]
    return drag_data, new_index_col


def drawMultipleBar(exposure_T, drag_data, col_name, index_info, base_col='CK'):
    x_data = [str(int(_) + 1) for _ in exposure_T]

    bar = (
        Bar(
            init_opts=opts.InitOpts(width="1680px", height="600px"),
        ).add_xaxis(x_data)
    )

    for idrag, angle in drag_data.items():
        for col in col_name:
            bar.add_yaxis(f"{idrag}", [_ * 100 for _ in list(angle[col].values)], stack=idrag)
            if idrag == base_col:
                base_line = [_ * 100 for _ in list(angle[col].values)]

    bar.set_series_opts(
        label_opts=opts.LabelOpts(is_show=False),
        # markline_opts=opts.MarkLineOpts(
        #     data=[opts.MarkLineItem(y=_, name="CK") for _ in base_line]
        # ),
    )
    bar.set_global_opts(
        title_opts=opts.TitleOpts(
            pos_left="left", title='', item_gap=20
        ),
        toolbox_opts=opts.ToolboxOpts(
            feature=opts.ToolBoxFeatureOpts(
                save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(
                    type_='jpeg',
                    background_color='white'
                )
            )
        ),

        legend_opts=opts.LegendOpts(

            textstyle_opts=opts.TextStyleOpts(
                font_family='Microsoft YaHei',
                font_style='normal',
                font_size=15,
                font_weight='bold'
            ),
        ),
        tooltip_opts=opts.TooltipOpts(
            trigger="item", axis_pointer_type="shadow"
        ),
        xaxis_opts=opts.AxisOpts(
            type_="category",
            boundary_gap=True,
            splitarea_opts=opts.SplitAreaOpts(
                areastyle_opts=opts.AreaStyleOpts(opacity=1)
            ),
            name_textstyle_opts=opts.TextStyleOpts(
                font_family='Microsoft YaHei',
                font_style='normal',
                font_size=15,
                font_weight='bold'
            ),
            # axislabel_opts=opts.LabelOpts(formatter="expr {value}"),
            name='Time (every 1 hour)',
            name_location='middle',
            name_gap=45,
            splitline_opts=opts.SplitLineOpts(is_show=False),
        ),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            name=f"{index_info['name']} (%)",
            name_location='middle',
            name_gap=45,
            name_rotate=90,
            name_textstyle_opts=opts.TextStyleOpts(
                font_family='Microsoft YaHei',
                font_style='normal',
                font_size=15,
                font_weight='bold'
            )
        ),
    )

    bar.render(os.path.join(out_path, f"{'_'.join(drag_list)}_{index_info['name']}.html"))


# =================== matplotlib barplot  ===========================#
def drawBars(y_data, labels, drag_list, title):
    labels = labels

    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(dpi=300)

    for idx, drag_name in enumerate(drag_list):
        means_data = [_*100 for _ in y_data[drag_name]['mean']]
        stds_data = [_*100 for _ in y_data[drag_name]['std']]
        rects = ax.bar(
            x - width * len(drag_list) / 2 + width*(idx+1) - width /2,
            means_data, width,
            # yerr=stds_data,
            label=drag_name
        )
        # ax.bar_label(rects, padding=1)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Time (every 1 hour)')
    ax.set_xticks(x, labels)
    ax.set_ylabel('Ratio of water surface retention (%)')
    ax.set_yticks([_ * 10 for _ in range(11)])
    ax.set_yticklabels([_ * 10 for _ in range(11)])

    ax.legend()

    fig.tight_layout()

    plt.savefig(os.path.join(out_path, f"{title}_{'_'.join(drag_list)}.png"), dpi=300)


def calculateMeanData(data, index_col, group_col=('exposure_time', 'drag_name')):
    group_col = list(group_col)
    use_col = index_col + group_col
    mean_data = data[use_col].groupby(group_col).mean()
    std_data = data[use_col].groupby(group_col).std()
    mean_data.to_csv(os.path.join(out_path, 'mean_data.csv'))
    print("mean data done")
    return mean_data, std_data


if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()

    ap.add_argument("-rp", "--root_path", default="E:\\data\\3D_pre\\")
    ap.add_argument("--drag_names",
                    default="CK,"
                            "4Hydroxy50ppb,4Hydroxy500ppb,4Hydroxy1ppm",
                    # "6PPD50ppb,6PPD500ppb,6PPD1ppm",
                    # "6PPDQ50ppb,6PPDQ500ppb,6PPDQ1ppm",
                    # "RJ",

                    # default="4Hydroxy50ppb,6PPDQ1ppm,6PPD50ppb,6PPD500ppb,6PPD1ppm,RJ",
                    type=str)
    ap.add_argument("--exposureT", default="0,1,2,3,4,5,6,7,8,9,10,11", type=str)

    args = vars(ap.parse_args())

    root_path = args['root_path']
    exposureT = args['exposureT']
    drag_names = args['drag_names']

    drag_list = drag_names.split(",")
    exposureT_list = exposureT.split(",")

    turning_angle_interval = (0, 10, 20, 30, 40, 50, 60, 70, 80, 91)
    up_down_angle_interval = (0, 15, 45, 91)

    info_col = ['exposure_time', 'DayTank', 'region_name']
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
    out_path = os.path.join(root_path, 'drag_result', 'multiple')

    topdata_dict = pd.DataFrame()
    sidedata_dict = pd.DataFrame()
    for drag_name in drag_list:
        index_path = os.path.join(root_path, 'drag_result', 'multiple', drag_name)
        side_data = pd.read_csv(os.path.join(index_path, "side_view.csv"))
        top_data = pd.read_csv(os.path.join(index_path, "top_view.csv"))
        top_data['drag_name'] = drag_name
        side_data['drag_name'] = drag_name
        topdata_dict = pd.concat([topdata_dict, top_data])
        sidedata_dict = pd.concat([sidedata_dict, side_data])

    mean_data, std_data = calculateMeanData(topdata_dict, index_col=['velocity', 'dist_cal'])

    # ============================================================ #
    velocity_info = {
        'name': 'velocity',
        'unit': "mm/s",
    }
    drag_data = formatBoxData(
        topdata_dict, index_name='velocity', exposureT_list=exposureT_list,
        category_name='drag_name'
    )
    drawMultipleBox(exposureT_list, drag_data, velocity_info)
    #
    # dist_info = {
    #     'name': 'dist',
    #     'unit': "mm",
    # }
    # drag_data = formatBoxData(
    #     topdata_dict, index_name='dist', exposureT_list=exposureT_list,
    #     category_name='drag_name'
    # )
    # drawMultipleBox(exposureT_list, drag_data, dist_info)

    # ============================================================ #
    dist_cal_info = {
        'name': 'distance',
        'unit': "mm",
    }
    drag_data = formatBoxData(
        topdata_dict, index_name='dist_cal', exposureT_list=exposureT_list,
        category_name='drag_name'
    )
    drawMultipleBox(exposureT_list, drag_data, dist_cal_info)

    # ============================================================ #
    # angle_cal_info = {
    #     'name': 'Turning Angle',
    #     'unit': "%",
    # }
    # drag_data, new_index_col = formatBarData(
    #     topdata_dict,
    #     index_name=top_angle_col,
    #     exposureT_list=exposureT_list,
    #     category_name='drag_name'
    # )
    # drawMultipleBar(exposureT_list, drag_data, new_index_col, angle_cal_info)

    # ==========================top time================================== #
    mean_topdata, std_topdata = calculateMeanData(sidedata_dict, index_col=['top_time'])
    mean_topdata.reset_index(inplace=True)
    std_topdata.reset_index(inplace=True)
    y_data = {}
    for drag_name in drag_list:
        mean_data = mean_topdata[mean_topdata['drag_name'] == drag_name]['top_time'].tolist()
        std_data = std_topdata[std_topdata['drag_name'] == drag_name]['top_time'].tolist()
        y_data[drag_name] = {
            'mean': mean_data,
            'std': std_data,
        }
    labels = [_ + 1 for _ in list(mean_topdata['exposure_time'].unique())]
    drawBars(y_data, labels, drag_list, 'top_time')
    print("done")
