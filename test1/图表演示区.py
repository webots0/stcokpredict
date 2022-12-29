# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 03:24:41 2022

@author: webot
"""



import pyecharts
from pyecharts import options as opts
from pyecharts.charts import Geo
from pyecharts.globals import ChartType, SymbolType

c = (
    Geo()
    .add_schema(maptype="china")
    .add(
        "",
        [("广州", 55), ("北京", 66), ("杭州", 77), ("重庆", 88)],
        type_=ChartType.EFFECT_SCATTER,
        color="white",
    )
    .add(
        "geo",
        [("广州", "上海"), ("广州", "北京"), ("广州", "杭州"), ("广州", "重庆")],
        type_=ChartType.LINES,
        effect_opts=opts.EffectOpts(
            symbol=SymbolType.ARROW, symbol_size=6, color="blue"
        ),
        linestyle_opts=opts.LineStyleOpts(curve=0.2),
    )
    .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(title_opts=opts.TitleOpts(title="Geo-Lines"))
    .render("geo_lines.html")
)
#%%
import pyecharts.options as opts
from pyecharts.charts import WordCloud

"""
Gallery 使用 pyecharts 1.1.0
参考地址: https://gallery.echartsjs.com/editor.html?c=xS1jMxuOVm

目前无法实现的功能:

1、暂无
"""

data = [
    ("生活资源", "999"),
    ("供热管理", "888"),
    ("供气质量", "777"),
    ("生活用水管理", "688"),
    ("一次供水问题", "588"),
    ("交通运输", "516"),
    ("城市交通", "515"),
    ("环境保护", "483"),
    ("房地产管理", "462"),
    ("城乡建设", "449"),
    ("社会保障与福利", "429"),
    ("社会保障", "407"),
    ("文体与教育管理", "406"),
    ("公共安全", "406"),
    ("公交运输管理", "386"),
    ("出租车运营管理", "385"),
    ("供热管理", "375"),
    ("市容环卫", "355"),
    ("自然资源管理", "355"),
    ("粉尘污染", "335"),
    ("噪声污染", "324"),
    ("土地资源管理", "304"),
    ("物业服务与管理", "304"),
    ("医疗卫生", "284"),
    ("粉煤灰污染", "284"),
    ("占道", "284"),
    ("供热发展", "254"),
    ("农村土地规划管理", "254"),
    ("生活噪音", "253"),
    ("供热单位影响", "253"),
    ("城市供电", "223"),
    ("房屋质量与安全", "223"),
    ("大气污染", "223"),
    ("房屋安全", "223"),
    ("文化活动", "223"),
    ("拆迁管理", "223"),
    ("公共设施", "223"),
    ("供气质量", "223"),
    ("供电管理", "223"),
    ("燃气管理", "152"),
    ("教育管理", "152"),
    ("医疗纠纷", "152"),
    ("执法监督", "152"),
    ("设备安全", "152"),
    ("政务建设", "152"),
    ("县区、开发区", "152"),
    ("宏观经济", "152"),
    ("教育管理", "112"),
    ("社会保障", "112"),
    ("生活用水管理", "112"),
    ("物业服务与管理", "112"),
    ("分类列表", "112"),
    ("农业生产", "112"),
    ("二次供水问题", "112"),
    ("城市公共设施", "92"),
    ("拆迁政策咨询", "92"),
    ("物业服务", "92"),
    ("物业管理", "92"),
    ("社会保障保险管理", "92"),
    ("低保管理", "92"),
    ("文娱市场管理", "72"),
    ("城市交通秩序管理", "72"),
    ("执法争议", "72"),
    ("商业烟尘污染", "72"),
    ("占道堆放", "71"),
    ("地上设施", "71"),
    ("水质", "71"),
    ("无水", "71"),
    ("供热单位影响", "71"),
    ("人行道管理", "71"),
    ("主网原因", "71"),
    ("集中供热", "71"),
    ("客运管理", "71"),
    ("国有公交（大巴）管理", "71"),
    ("工业粉尘污染", "71"),
    ("治安案件", "71"),
    ("压力容器安全", "71"),
    ("身份证管理", "71"),
    ("群众健身", "41"),
    ("工业排放污染", "41"),
    ("破坏森林资源", "41"),
    ("市场收费", "41"),
    ("生产资金", "41"),
    ("生产噪声", "41"),
    ("农村低保", "41"),
    ("劳动争议", "41"),
    ("劳动合同争议", "41"),
    ("劳动报酬与福利", "41"),
    ("医疗事故", "21"),
    ("停供", "21"),
    ("基础教育", "21"),
    ("职业教育", "21"),
    ("物业资质管理", "21"),
    ("拆迁补偿", "21"),
    ("设施维护", "21"),
    ("市场外溢", "11"),
    ("占道经营", "11"),
    ("树木管理", "11"),
    ("农村基础设施", "11"),
    ("无水", "11"),
    ("供气质量", "11"),
    ("停气", "11"),
    ("市政府工作部门（含部门管理机构、直属单位）", "11"),
    ("燃气管理", "11"),
    ("市容环卫", "11"),
    ("新闻传媒", "11"),
    ("人才招聘", "11"),
    ("市场环境", "11"),
    ("行政事业收费", "11"),
    ("食品安全与卫生", "11"),
    ("城市交通", "11"),
    ("房地产开发", "11"),
    ("房屋配套问题", "11"),
    ("物业服务", "11"),
    ("物业管理", "11"),
    ("占道", "11"),
    ("园林绿化", "11"),
    ("户籍管理及身份证", "11"),
    ("公交运输管理", "11"),
    ("公路（水路）交通", "11"),
    ("房屋与图纸不符", "11"),
    ("有线电视", "11"),
    ("社会治安", "11"),
    ("林业资源", "11"),
    ("其他行政事业收费", "11"),
    ("经营性收费", "11"),
    ("食品安全与卫生", "11"),
    ("体育活动", "11"),
    ("有线电视安装及调试维护", "11"),
    ("低保管理", "11"),
    ("劳动争议", "11"),
    ("社会福利及事务", "11"),
    ("一次供水问题", "11"),
]


(
    WordCloud()
    .add(series_name="热点分析", data_pair=data, word_size_range=[6, 66])
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="热点分析", title_textstyle_opts=opts.TextStyleOpts(font_size=23)
        ),
        tooltip_opts=opts.TooltipOpts(is_show=True),
    )
    .render("basic_wordcloud.html")
)

#%%

from pyecharts import options as opts
from pyecharts.charts import Bar
from pyecharts.faker import Faker

c = (
    Bar()
    .add_xaxis(Faker.choose())
    .add_yaxis("商家A", Faker.values())
    .add_yaxis("商家B", Faker.values())
    .set_global_opts(
        title_opts=opts.TitleOpts(title="Bar-显示 ToolBox"),
        toolbox_opts=opts.ToolboxOpts(),
        legend_opts=opts.LegendOpts(is_show=False),
    )
    .render()
)



#%%
from pyecharts import options as opts
from pyecharts.charts import Graph

nodes = [
    {"name": "结点1", "symbolSize": 10},
    {"name": "结点2", "symbolSize": 20},
    {"name": "结点3", "symbolSize": 30},
    {"name": "结点4", "symbolSize": 40},
    {"name": "结点5", "symbolSize": 50},
    {"name": "结点6", "symbolSize": 40},
    {"name": "结点7", "symbolSize": 30},
    {"name": "结点8", "symbolSize": 20},
]
links = []
for i in nodes:
    for j in nodes:
        links.append({"source": i.get("name"), "target": j.get("name")})
c = (
    Graph()
    .add("", nodes, links, repulsion=8000)
    .set_global_opts(title_opts=opts.TitleOpts(title="Graph-基本示例"))
    .render()
)
#%%





import json

from pyecharts import options as opts
from pyecharts.charts import Graph

with open("weibo.json", "r", encoding="utf-8") as f:
    j = json.load(f)
    nodes, links, categories, cont, mid, userl = j
c = (
    Graph()
    .add(
        "",
        nodes,
        links,
        categories,
        repulsion=50,
        linestyle_opts=opts.LineStyleOpts(curve=0.2),
        label_opts=opts.LabelOpts(is_show=False),
    )
    .set_global_opts(
        legend_opts=opts.LegendOpts(is_show=False),
        title_opts=opts.TitleOpts(title="Graph-微博转发关系图"),
    )
    .render()
)

#%%
import math

from pyecharts import options as opts
from pyecharts.charts import Line3D
from pyecharts.faker import Faker

data = []
for t in range(0, 25000):
    _t = t / 1000
    x = (1 + 0.25 * math.cos(75 * _t)) * math.cos(_t)
    y = (1 + 0.25 * math.cos(75 * _t)) * math.sin(_t)
    z = _t + 2.0 * math.sin(75 * _t)
    data.append([x, y, z])
c = (
    Line3D()
    .add(
        "",
        data,
        xaxis3d_opts=opts.Axis3DOpts(Faker.clock, type_="value"),
        yaxis3d_opts=opts.Axis3DOpts(Faker.week_en, type_="value"),
        grid3d_opts=opts.Grid3DOpts(
            width=100, depth=100, rotate_speed=150, is_rotate=True
        ),
    )
    .set_global_opts(
        visualmap_opts=opts.VisualMapOpts(
            max_=30, min_=0, range_color=Faker.visual_color
        ),
        title_opts=opts.TitleOpts(title="Line3D-旋转的弹簧"),
    )
    .render()
)

#%%

import math

import pyecharts.options as opts
from pyecharts.charts import Line3D

week_en = "Saturday Friday Thursday Wednesday Tuesday Monday Sunday".split()
clock = (
    "12a 1a 2a 3a 4a 5a 6a 7a 8a 9a 10a 11a 12p "
    "1p 2p 3p 4p 5p 6p 7p 8p 9p 10p 11p".split()
)

data = []
for t in range(0, 25000):
    _t = t / 1000
    x = (1 + 0.25 * math.cos(75 * _t)) * math.cos(_t)
    y = (1 + 0.25 * math.cos(75 * _t)) * math.sin(_t)
    z = _t + 2.0 * math.sin(75 * _t)
    data.append([x, y, z])

(
    Line3D()
    .add(
        "",
        data,
        xaxis3d_opts=opts.Axis3DOpts(data=clock, type_="value"),
        yaxis3d_opts=opts.Axis3DOpts(data=week_en, type_="value"),
        grid3d_opts=opts.Grid3DOpts(width=100, height=100, depth=100),
    )
    .set_global_opts(
        visualmap_opts=opts.VisualMapOpts(
            dimension=2,
            max_=30,
            min_=0,
            range_color=[
                "#313695",
                "#4575b4",
                "#74add1",
                "#abd9e9",
                "#e0f3f8",
                "#ffffbf",
                "#fee090",
                "#fdae61",
                "#f46d43",
                "#d73027",
                "#a50026",
            ],
        )
    )
    .render()
)


#%%

from pyecharts import options as opts
from pyecharts.charts import Map
from pyecharts.faker import Faker

c = (
    Map()
    .add(
        "商家A",
        [list(z) for z in zip(Faker.guangdong_city, Faker.values())],
        "china-cities",
        label_opts=opts.LabelOpts(is_show=False),
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="Map-中国地图（带城市）"),
        visualmap_opts=opts.VisualMapOpts(),
    )
    .render()
)


#%%

import pyecharts.options as opts
from pyecharts.charts import MapGlobe
from pyecharts.faker import POPULATION

data = [x for _, x in POPULATION[1:]]
low, high = min(data), max(data)

c = (
    MapGlobe()
    .add_schema()
    .add(
        maptype="world",
        series_name="World Population",
        data_pair=POPULATION[1:],
        is_map_symbol_show=False,
        label_opts=opts.LabelOpts(is_show=False),
    )
    .set_global_opts(
        visualmap_opts=opts.VisualMapOpts(
            min_=low,
            max_=high,
            range_text=["max", "min"],
            is_calculable=True,
            range_color=["lightskyblue", "yellow", "orangered"],
        )
    )
    .render()
)

#%%

import random

from pyecharts import options as opts
from pyecharts.charts import GraphGL


nodes = []
for i in range(50):
    for j in range(50):
        nodes.append(opts.GraphGLNode(
            x=random.random() * 958,
            y=random.random() * 777,
            value=1,
        ))

links = []
for i in range(50):
    for j in range(50):
        if i < 50 - 1:
            links.append(opts.GraphGLLink(
                source=i + j * 50,
                target=i + 1 + j * 50,
                value=1,
            ))
        if j < 50 - 1:
            links.append(opts.GraphGLLink(
                source=i + j * 50,
                target=i + (j + 1) * 50,
                value=1,
            ))

c = (
    GraphGL(init_opts=opts.InitOpts())
    .add(
        series_name="",
        nodes=nodes,
        links=links,
        itemstyle_opts=opts.ItemStyleOpts(color="rgba(255,255,255,0.8)"),
        linestyle_opts=opts.LineStyleOpts(color="rgba(255,255,255,0.8)", width=3),
        force_atlas2_opts=opts.GraphGLForceAtlas2Opts(
            steps=5,
            edge_weight_influence=4,
        )
    )
    .set_dark_mode()
    .render()
)

