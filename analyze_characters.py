#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
西游记角色别名统计分析
统计孙悟空和猪八戒的别名在各章节中的出现次数
"""

import os
import csv
from pathlib import Path

# 定义角色别名
SUN_WUKONG_ALIASES = [
    "孫悟空", "美猴王", "齊天大聖", "那猴", "弼馬溫", "石猴", "鬥戰勝佛", "行者"
]

ZHU_BAJIE_ALIASES = [
    "八戒", "天蓬元帥", "豬剛鬣", "悟能"
]


def count_aliases_in_chapter(chapter_file, aliases):
    """
    统计一个章节文件中各别名的出现次数
    
    参数:
        chapter_file: 章节文件路径
        aliases: 别名列表
    
    返回:
        字典，键为别名，值为出现次数
    """
    try:
        with open(chapter_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        counts = {}
        for alias in aliases:
            counts[alias] = content.count(alias)
        return counts
    except Exception as e:
        print(f"读取文件 {chapter_file} 时出错: {e}")
        return {alias: 0 for alias in aliases}


def generate_character_csv(chapters_dir, output_file, character_name, aliases):
    """
    为单个角色生成CSV统计文件
    
    参数:
        chapters_dir: 章节目录路径
        output_file: 输出CSV文件路径
        character_name: 角色名称
        aliases: 角色别名列表
    """
    # 收集所有章节文件
    chapter_files = []
    for i in range(1, 101):
        chapter_file = chapters_dir / f"{i}-chapter.txt"
        if chapter_file.exists():
            chapter_files.append((i, chapter_file))
    
    chapter_files.sort(key=lambda x: x[0])
    
    # 创建CSV文件
    with open(output_file, 'w', encoding='utf-8-sig', newline='') as csvfile:
        # CSV头部：章节，所有别名，总计
        headers = ['章节'] + aliases + ['总计']
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        
        # 遍历每一章
        for chapter_num, chapter_file in chapter_files:
            print(f"  正在处理第 {chapter_num} 章...")
            
            # 统计别名出现次数
            alias_counts = count_aliases_in_chapter(chapter_file, aliases)
            
            # 计算总计
            total_count = sum(alias_counts.values())
            
            # 构建行数据
            row = {'章节': f"第{chapter_num}章"}
            row.update(alias_counts)
            row['总计'] = total_count
            
            writer.writerow(row)
    
    print(f"  {character_name}统计完成！结果已保存到: {output_file}")
    return len(chapter_files)


def main():
    """主函数"""
    print("=" * 60)
    print("西游记角色别名统计分析")
    print("=" * 60)
    
    # 设置路径
    base_dir = Path(__file__).parent
    chapters_dir = base_dir / "xiyouji_chapters"
    
    # 检查章节目录是否存在
    if not chapters_dir.exists():
        print(f"错误：章节目录不存在 {chapters_dir}")
        return
    
    # 统计孙悟空
    print("\n【孙悟空】别名统计中...")
    sun_output = base_dir / "sun_wukong_aliases_count.csv"
    sun_chapters = generate_character_csv(
        chapters_dir, 
        sun_output, 
        "孙悟空", 
        SUN_WUKONG_ALIASES
    )
    
    # 统计猪八戒
    print("\n【猪八戒】别名统计中...")
    zhu_output = base_dir / "zhu_bajie_aliases_count.csv"
    zhu_chapters = generate_character_csv(
        chapters_dir, 
        zhu_output, 
        "猪八戒", 
        ZHU_BAJIE_ALIASES
    )
    
    # 输出统计摘要
    print("\n" + "=" * 60)
    print("统计完成！")
    print("=" * 60)
    print(f"共处理章节: {sun_chapters} 章")
    print(f"\n孙悟空别名统计文件: {sun_output.name}")
    print(f"  包含别名: {', '.join(SUN_WUKONG_ALIASES)}")
    print(f"\n猪八戒别名统计文件: {zhu_output.name}")
    print(f"  包含别名: {', '.join(ZHU_BAJIE_ALIASES)}")
    print("=" * 60)


if __name__ == "__main__":
    main()

