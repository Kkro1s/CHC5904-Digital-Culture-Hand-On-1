#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
西游记角色别名频率可视化
绘制孙悟空和猪八戒各别名在每章的出现频率曲线图
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from pathlib import Path
import os

# 设置matplotlib配置目录
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Hiragino Sans GB', 'STHeiti', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12


def plot_character_aliases(csv_file, character_name, output_file, aliases, exclude_most_frequent=False):
    """
    为单个角色绘制别名频率曲线图
    
    参数:
        csv_file: CSV文件路径
        character_name: 角色名称
        output_file: 输出图片文件路径
        aliases: 别名列表（不包括"总计"列）
        exclude_most_frequent: 是否排除出现频率最高的别名
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    
    # 提取章节编号（从"第X章"中提取数字）
    df['章节编号'] = df['章节'].str.extract(r'第(\d+)章').astype(int)
    
    # 如果需要排除最高频率的别名
    aliases_to_plot = aliases.copy()
    excluded_alias = None
    if exclude_most_frequent:
        # 找出总出现次数最多的别名
        alias_totals = {}
        for alias in aliases:
            if alias in df.columns:
                alias_totals[alias] = df[alias].sum()
        if alias_totals:
            most_frequent = max(alias_totals, key=alias_totals.get)
            aliases_to_plot.remove(most_frequent)
            excluded_alias = most_frequent
            print(f"  排除最高频率别名: {most_frequent} (总计 {alias_totals[most_frequent]} 次)")
    
    # 创建图表
    plt.figure(figsize=(16, 10))
    
    # 为每个别名绘制曲线
    colors = plt.cm.tab10(range(len(aliases_to_plot)))
    
    for idx, alias in enumerate(aliases_to_plot):
        if alias in df.columns:
            plt.plot(df['章节编号'], df[alias], 
                    marker='o', markersize=4, linewidth=2,
                    label=alias, color=colors[idx], alpha=0.8)
    
    # 设置图表标题和标签
    title = f'{character_name}各别名在西游记各章节的出现频率'
    if excluded_alias:
        title += f'\n(已排除最高频别名：{excluded_alias})'
    plt.title(title, fontsize=20, fontweight='bold', pad=20)
    plt.xlabel('章节', fontsize=16, fontweight='bold')
    plt.ylabel('出现次数', fontsize=16, fontweight='bold')
    
    # 设置网格
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    
    # 设置图例
    plt.legend(loc='upper right', fontsize=12, framealpha=0.9, 
              ncol=2 if len(aliases) > 4 else 1)
    
    # 设置x轴范围和刻度
    plt.xlim(0, 101)
    plt.xticks(range(0, 101, 10), fontsize=12)
    plt.yticks(fontsize=12)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  {character_name}图表已保存: {output_file}")
    
    # 关闭图表释放内存
    plt.close()


def generate_summary_stats(csv_file, character_name, aliases):
    """
    生成角色统计摘要
    
    参数:
        csv_file: CSV文件路径
        character_name: 角色名称
        aliases: 别名列表
    """
    df = pd.read_csv(csv_file)
    
    print(f"\n【{character_name}】统计摘要：")
    print("-" * 50)
    
    for alias in aliases:
        if alias in df.columns:
            total = df[alias].sum()
            avg = df[alias].mean()
            max_val = df[alias].max()
            max_chapter = df.loc[df[alias].idxmax(), '章节']
            print(f"  {alias:8s}: 总计 {total:4d} 次 | 平均 {avg:5.2f} 次/章 | "
                  f"最高 {max_val:3d} 次 ({max_chapter})")
    
    if '总计' in df.columns:
        total_all = df['总计'].sum()
        avg_all = df['总计'].mean()
        print("-" * 50)
        print(f"  {'总计':8s}: {total_all:4d} 次 | 平均 {avg_all:5.2f} 次/章")


def main():
    """主函数"""
    print("=" * 60)
    print("西游记角色别名频率可视化")
    print("=" * 60)
    
    # 设置路径
    base_dir = Path(__file__).parent
    
    # 孙悟空别名
    sun_aliases = ["孫悟空", "美猴王", "齊天大聖", "那猴", 
                   "弼馬溫", "石猴", "鬥戰勝佛", "行者"]
    
    # 猪八戒别名
    zhu_aliases = ["八戒", "天蓬元帥", "豬剛鬣", "悟能"]
    
    # 孙悟空可视化
    print("\n生成孙悟空图表...")
    sun_csv = base_dir / "sun_wukong_aliases_count.csv"
    sun_output = base_dir / "sun_wukong_frequency_plot.png"
    
    if sun_csv.exists():
        plot_character_aliases(sun_csv, "孙悟空", sun_output, sun_aliases, exclude_most_frequent=True)
        generate_summary_stats(sun_csv, "孙悟空", sun_aliases)
    else:
        print(f"  错误：找不到文件 {sun_csv}")
    
    # 猪八戒可视化
    print("\n生成猪八戒图表...")
    zhu_csv = base_dir / "zhu_bajie_aliases_count.csv"
    zhu_output = base_dir / "zhu_bajie_frequency_plot.png"
    
    if zhu_csv.exists():
        plot_character_aliases(zhu_csv, "猪八戒", zhu_output, zhu_aliases, exclude_most_frequent=True)
        generate_summary_stats(zhu_csv, "猪八戒", zhu_aliases)
    else:
        print(f"  错误：找不到文件 {zhu_csv}")
    
    # 完成
    print("\n" + "=" * 60)
    print("可视化完成！")
    print("=" * 60)
    print(f"孙悟空图表: sun_wukong_frequency_plot.png")
    print(f"猪八戒图表: zhu_bajie_frequency_plot.png")
    print("=" * 60)


if __name__ == "__main__":
    main()

