#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
西游记别名高级分析 - PCA降维和对比分析
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# 设置matplotlib配置
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Hiragino Sans GB', 'STHeiti', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

# 分析的别名
SUN_WUKONG_ALIASES = ["弼馬溫", "石猴", "鬥戰勝佛"]
ZHU_BAJIE_ALIASES = ["天蓬元帥", "豬剛鬣", "悟能"]


class AdvancedAliasAnalyzer:
    """高级别名分析器"""
    
    def __init__(self, chapters_dir):
        self.chapters_dir = Path(chapters_dir)
        self.chapters_data = {}
        self.load_chapters()
    
    def load_chapters(self):
        """加载所有章节"""
        for i in range(1, 101):
            chapter_file = self.chapters_dir / f"{i}-chapter.txt"
            if chapter_file.exists():
                with open(chapter_file, 'r', encoding='utf-8') as f:
                    self.chapters_data[i] = f.read()
    
    def build_feature_matrix(self, aliases):
        """
        为别名构建特征矩阵用于PCA分析
        
        特征包括:
        - 每10章的出现频率
        - 首次出现位置
        - 最后出现位置
        - 总频率
        - 分布广度
        """
        features = []
        feature_names = []
        
        for alias in aliases:
            # 按10章为单位统计频率
            freq_by_decade = []
            for decade in range(10):
                start_ch = decade * 10 + 1
                end_ch = min((decade + 1) * 10, 100)
                count = sum(self.chapters_data.get(ch, '').count(alias) 
                           for ch in range(start_ch, end_ch + 1))
                freq_by_decade.append(count)
            
            # 其他统计特征
            all_counts = [self.chapters_data[ch].count(alias) 
                         for ch in range(1, 101)]
            
            total_count = sum(all_counts)
            if total_count > 0:
                chapters_with_alias = [i+1 for i, c in enumerate(all_counts) if c > 0]
                first_chapter = min(chapters_with_alias)
                last_chapter = max(chapters_with_alias)
                span = last_chapter - first_chapter + 1
                concentration = max(all_counts) / total_count if total_count > 0 else 0
            else:
                first_chapter = last_chapter = span = concentration = 0
            
            # 组合特征向量
            feature_vector = freq_by_decade + [
                total_count,
                first_chapter,
                last_chapter,
                span,
                concentration,
                len([c for c in all_counts if c > 0])  # 出现的章节数
            ]
            
            features.append(feature_vector)
            feature_names.append(alias)
        
        return np.array(features), feature_names
    
    def perform_pca_analysis(self, all_aliases, character_labels, output_dir):
        """
        执行PCA降维分析
        
        参数:
            all_aliases: 所有别名列表
            character_labels: 每个别名对应的角色
            output_dir: 输出目录
        """
        # 构建特征矩阵
        features, feature_names = self.build_feature_matrix(all_aliases)
        
        # 标准化
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # PCA降维到2D
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features_scaled)
        
        # 可视化
        plt.figure(figsize=(14, 10))
        
        # 定义颜色
        colors = {'孙悟空': '#FF6B6B', '猪八戒': '#4ECDC4'}
        
        for i, (alias, label) in enumerate(zip(feature_names, character_labels)):
            plt.scatter(features_pca[i, 0], features_pca[i, 1],
                       c=colors[label], s=500, alpha=0.6, edgecolors='black', linewidth=2)
            plt.annotate(alias, (features_pca[i, 0], features_pca[i, 1]),
                        fontsize=16, fontweight='bold',
                        ha='center', va='center')
        
        plt.xlabel(f'第一主成分 (解释方差: {pca.explained_variance_ratio_[0]:.1%})',
                  fontsize=14, fontweight='bold')
        plt.ylabel(f'第二主成分 (解释方差: {pca.explained_variance_ratio_[1]:.1%})',
                  fontsize=14, fontweight='bold')
        plt.title('别名使用模式的PCA降维分析\n(基于时间分布、频率和其他特征)',
                 fontsize=18, fontweight='bold', pad=20)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors['孙悟空'], label='孙悟空'),
                          Patch(facecolor=colors['猪八戒'], label='猪八戒')]
        plt.legend(handles=legend_elements, loc='best', fontsize=12)
        
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(output_dir / 'PCA_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nPCA分析完成！")
        print(f"第一主成分解释方差: {pca.explained_variance_ratio_[0]:.2%}")
        print(f"第二主成分解释方差: {pca.explained_variance_ratio_[1]:.2%}")
        print(f"累计解释方差: {sum(pca.explained_variance_ratio_):.2%}")
        
        return features_pca, pca
    
    def create_temporal_heatmap(self, aliases, character_name, output_file):
        """
        创建时间热力图
        
        参数:
            aliases: 别名列表
            character_name: 角色名称
            output_file: 输出文件
        """
        # 构建数据矩阵 (别名 x 章节)
        data = []
        for alias in aliases:
            counts = [self.chapters_data[ch].count(alias) for ch in range(1, 101)]
            data.append(counts)
        
        data = np.array(data)
        
        # 创建热力图
        plt.figure(figsize=(20, 6))
        sns.heatmap(data, cmap='YlOrRd', cbar_kws={'label': '出现次数'},
                   yticklabels=aliases, xticklabels=[f'{i}' if i % 10 == 0 else '' for i in range(1, 101)])
        
        plt.title(f'{character_name}各别名在100章中的时间分布热力图',
                 fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('章节', fontsize=14, fontweight='bold')
        plt.ylabel('别名', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_comparison_charts(self, aliases, labels, output_dir):
        """
        创建对比图表
        
        参数:
            aliases: 所有别名列表
            labels: 每个别名的角色标签
            output_dir: 输出目录
        """
        # 1. 时间线对比图
        fig, axes = plt.subplots(2, 1, figsize=(18, 12))
        
        sun_aliases = [a for a, l in zip(aliases, labels) if l == '孙悟空']
        zhu_aliases = [a for a, l in zip(aliases, labels) if l == '猪八戒']
        
        # 孙悟空
        for alias in sun_aliases:
            counts = [self.chapters_data[ch].count(alias) for ch in range(1, 101)]
            axes[0].plot(range(1, 101), counts, marker='o', markersize=3,
                        linewidth=2, label=alias, alpha=0.8)
        
        axes[0].set_title('孙悟空三个别名的时间分布对比', fontsize=16, fontweight='bold')
        axes[0].set_xlabel('章节', fontsize=12)
        axes[0].set_ylabel('出现次数', fontsize=12)
        axes[0].legend(fontsize=12, loc='best')
        axes[0].grid(True, alpha=0.3)
        
        # 猪八戒
        for alias in zhu_aliases:
            counts = [self.chapters_data[ch].count(alias) for ch in range(1, 101)]
            axes[1].plot(range(1, 101), counts, marker='o', markersize=3,
                        linewidth=2, label=alias, alpha=0.8)
        
        axes[1].set_title('猪八戒三个别名的时间分布对比', fontsize=16, fontweight='bold')
        axes[1].set_xlabel('章节', fontsize=12)
        axes[1].set_ylabel('出现次数', fontsize=12)
        axes[1].legend(fontsize=12, loc='best')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'temporal_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 统计对比条形图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 总出现次数对比
        totals = [sum(self.chapters_data[ch].count(alias) for ch in range(1, 101)) 
                 for alias in aliases]
        colors = ['#FF6B6B' if l == '孙悟空' else '#4ECDC4' for l in labels]
        
        axes[0, 0].bar(aliases, totals, color=colors, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('总出现次数对比', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('次数', fontsize=12)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 分布章节数对比
        chapter_counts = [len([ch for ch in range(1, 101) 
                              if self.chapters_data[ch].count(alias) > 0]) 
                         for alias in aliases]
        axes[0, 1].bar(aliases, chapter_counts, color=colors, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('出现的章节数对比', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('章节数', fontsize=12)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 首次出现章节
        first_appearances = []
        for alias in aliases:
            for ch in range(1, 101):
                if self.chapters_data[ch].count(alias) > 0:
                    first_appearances.append(ch)
                    break
            else:
                first_appearances.append(0)
        
        axes[1, 0].bar(aliases, first_appearances, color=colors, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('首次出现章节', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('章节号', fontsize=12)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 平均每章出现次数
        avg_per_chapter = [t / max(c, 1) for t, c in zip(totals, chapter_counts)]
        axes[1, 1].bar(aliases, avg_per_chapter, color=colors, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('平均每章出现次数（在出现的章节中）', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('平均次数', fontsize=12)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'statistics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """主函数"""
    print("="*80)
    print("西游记别名高级分析 - PCA和对比分析")
    print("="*80)
    
    # 设置路径
    base_dir = Path(__file__).parent
    chapters_dir = base_dir / "xiyouji_chapters"
    output_dir = base_dir / "analysis_results"
    output_dir.mkdir(exist_ok=True)
    
    # 初始化分析器
    print("\n加载数据...")
    analyzer = AdvancedAliasAnalyzer(chapters_dir)
    
    # 准备数据
    all_aliases = SUN_WUKONG_ALIASES + ZHU_BAJIE_ALIASES
    character_labels = ['孙悟空'] * len(SUN_WUKONG_ALIASES) + \
                      ['猪八戒'] * len(ZHU_BAJIE_ALIASES)
    
    # 1. PCA分析
    print("\n执行PCA降维分析...")
    features_pca, pca = analyzer.perform_pca_analysis(
        all_aliases, character_labels, output_dir
    )
    
    # 2. 时间热力图
    print("\n生成时间分布热力图...")
    analyzer.create_temporal_heatmap(
        SUN_WUKONG_ALIASES, '孙悟空',
        output_dir / '孙悟空_temporal_heatmap.png'
    )
    analyzer.create_temporal_heatmap(
        ZHU_BAJIE_ALIASES, '猪八戒',
        output_dir / '猪八戒_temporal_heatmap.png'
    )
    
    # 3. 对比图表
    print("\n生成对比图表...")
    analyzer.create_comparison_charts(all_aliases, character_labels, output_dir)
    
    print("\n" + "="*80)
    print("高级分析完成！所有结果已保存到 analysis_results 目录")
    print("="*80)
    print("\n生成的文件:")
    print("  - PCA_analysis.png: PCA降维可视化")
    print("  - 孙悟空_temporal_heatmap.png: 孙悟空时间热力图")
    print("  - 猪八戒_temporal_heatmap.png: 猪八戒时间热力图")
    print("  - temporal_comparison.png: 时间分布对比")
    print("  - statistics_comparison.png: 统计数据对比")
    print("="*80)


if __name__ == "__main__":
    main()

