#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为每个别名创建峰值章节±10范围的频率图
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import os

os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Hiragino Sans GB', 'STHeiti', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12


def load_chapter_counts(chapters_dir, alias):
    """
    加载所有章节中某个别名的出现次数
    
    返回:
        dict: {章节号: 出现次数}
    """
    counts = {}
    for i in range(1, 101):
        chapter_file = Path(chapters_dir) / f"{i}-chapter.txt"
        if chapter_file.exists():
            with open(chapter_file, 'r', encoding='utf-8') as f:
                text = f.read()
                counts[i] = text.count(alias)
    return counts


def find_peak_chapter(counts):
    """找到峰值章节"""
    if not counts:
        return None
    return max(counts.keys(), key=lambda k: counts[k])


def create_peak_frequency_plot(counts, alias, peak_chapter, window=10, output_file=None):
    """
    创建峰值章节±window范围的频率图
    
    参数:
        counts: 章节计数字典
        alias: 别名
        peak_chapter: 峰值章节
        window: 窗口大小（默认±10章）
        output_file: 输出文件路径
    """
    # 确定显示范围
    start_ch = max(1, peak_chapter - window)
    end_ch = min(100, peak_chapter + window)
    
    # 提取范围内的数据
    chapters = list(range(start_ch, end_ch + 1))
    frequencies = [counts.get(ch, 0) for ch in chapters]
    
    # 创建图表
    plt.figure(figsize=(14, 8))
    
    # 绘制柱状图
    bars = plt.bar(chapters, frequencies, color='#4ECDC4', alpha=0.7, edgecolor='black')
    
    # 高亮峰值章节
    peak_index = chapters.index(peak_chapter)
    bars[peak_index].set_color('#FF6B6B')
    bars[peak_index].set_alpha(0.9)
    
    # 添加数值标签
    for i, (ch, freq) in enumerate(zip(chapters, frequencies)):
        if freq > 0:
            plt.text(ch, freq, str(freq), ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 设置标题和标签
    peak_count = counts.get(peak_chapter, 0)
    total_in_range = sum(frequencies)
    total_all = sum(counts.values())
    percentage_in_range = (total_in_range / total_all * 100) if total_all > 0 else 0
    
    title = f'"{alias}" 频率分布图\n'
    title += f'峰值章节: 第{peak_chapter}章 ({peak_count}次) | '
    title += f'显示范围: 第{start_ch}-{end_ch}章\n'
    title += f'范围内出现: {total_in_range}次 ({percentage_in_range:.1f}% 全书) | '
    title += f'全书总计: {total_all}次'
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('章节', fontsize=13, fontweight='bold')
    plt.ylabel('出现次数', fontsize=13, fontweight='bold')
    
    # 设置x轴刻度
    plt.xticks(chapters, [str(ch) if ch % 5 == 0 or ch == peak_chapter else '' for ch in chapters])
    
    # 添加网格
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 添加峰值章节的垂直线
    plt.axvline(x=peak_chapter, color='red', linestyle='--', linewidth=2, alpha=0.5, label=f'峰值: 第{peak_chapter}章')
    
    # 添加图例
    plt.legend(fontsize=11, loc='upper right')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  图表已保存: {output_file}")
    
    plt.close()


def main():
    """主函数"""
    print("="*80)
    print("生成峰值章节±10范围的频率图")
    print("="*80)
    
    # 设置路径
    base_dir = Path(__file__).parent
    chapters_dir = base_dir / "xiyouji_chapters"
    output_dir = base_dir / "peak_frequency_plots"
    output_dir.mkdir(exist_ok=True)
    
    # 要分析的别名及其峰值章节（从之前的分析获得）
    aliases_config = {
        '孙悟空': [
            ('弼馬溫', 4),
            ('石猴', 1),
            ('鬥戰勝佛', 100)
        ],
        '猪八戒': [
            ('豬剛鬣', 19),
            ('天蓬元帥', 38),
            ('悟能', 76)
        ]
    }
    
    for character, aliases in aliases_config.items():
        print(f"\n{'='*60}")
        print(f"处理角色: {character}")
        print(f"{'='*60}")
        
        for alias, expected_peak in aliases:
            print(f"\n别名: {alias}")
            
            # 加载章节计数
            counts = load_chapter_counts(chapters_dir, alias)
            
            # 验证峰值章节
            actual_peak = find_peak_chapter(counts)
            if actual_peak != expected_peak:
                print(f"  警告: 实际峰值章节({actual_peak})与预期({expected_peak})不同")
                peak_to_use = actual_peak
            else:
                peak_to_use = expected_peak
            
            print(f"  峰值章节: 第{peak_to_use}章 ({counts[peak_to_use]}次)")
            print(f"  全书总计: {sum(counts.values())}次")
            
            # 创建图表
            output_file = output_dir / f"{alias}_peak_frequency.png"
            create_peak_frequency_plot(counts, alias, peak_to_use, window=10, output_file=output_file)
    
    print("\n" + "="*80)
    print("所有图表生成完成！")
    print(f"输出目录: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()

