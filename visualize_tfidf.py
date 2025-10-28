#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化TF-IDF结果
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

# 已知人物名称
CHARACTERS = {'行者', '八戒', '三藏', '聖僧', '老孫', '那怪', '師父', '悟空', '悟能'}

def visualize_tfidf(csv_file, alias):
    """可视化TF-IDF"""
    df = pd.read_csv(csv_file)
    
    # 取前20个
    df_top = df.head(20)
    
    # 标记人物
    colors = ['#FF6B6B' if word in CHARACTERS else '#4ECDC4' 
             for word in df_top['word']]
    
    plt.figure(figsize=(14, 10))
    plt.barh(range(len(df_top)), df_top['tfidf_score'], color=colors, alpha=0.7, edgecolor='black')
    plt.yticks(range(len(df_top)), df_top['word'], fontsize=13)
    plt.xlabel('TF-IDF 分数', fontsize=14, fontweight='bold')
    plt.title(f'"{alias}" 上下文的TF-IDF分析 (Top 20)\n红色=人物名称, 蓝色=其他词汇',
             fontsize=16, fontweight='bold', pad=20)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    output_file = Path(csv_file).parent / f"{alias}_tfidf_visualization.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"可视化已保存: {output_file}")

def create_character_network(csv_file, alias):
    """创建简化的人物关系图"""
    import networkx as nx
    
    df = pd.read_csv(csv_file)
    
    # 筛选人物
    characters = df[df['word'].isin(CHARACTERS)]
    
    G = nx.Graph()
    G.add_node(alias, node_type='center')
    
    for _, row in characters.iterrows():
        word = row['word']
        score = row['tfidf_score']
        G.add_node(word, node_type='character')
        G.add_edge(alias, word, weight=score)
    
    # 可视化
    plt.figure(figsize=(14, 10))
    
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    node_sizes = []
    node_colors = []
    for node in G.nodes():
        if node == alias:
            node_sizes.append(4000)
            node_colors.append('#FF6B6B')
        else:
            node_sizes.append(2000)
            node_colors.append('#4ECDC4')
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                          alpha=0.8, edgecolors='black', linewidths=3)
    
    edge_widths = [G[u][v]['weight'] * 10 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, edge_color='gray')
    
    nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold',
                           font_family='Arial Unicode MS')
    
    # 边标签
    edge_labels = {(u, v): f"{G[u][v]['weight']:.3f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10)
    
    plt.title(f'"{alias}" 人物关系网络图 (基于TF-IDF相关度)\n边上的数字为TF-IDF分数',
             fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    
    output_file = Path(csv_file).parent / f"{alias}_relationship_network.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"关系网络图已保存: {output_file}")

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    csv_file = base_dir / "relationship_analysis" / "豬剛鬣_tfidf_top30.csv"
    alias = "豬剛鬣"
    
    print("生成可视化...")
    visualize_tfidf(csv_file, alias)
    
    print("\n生成关系网络图...")
    create_character_network(csv_file, alias)
    
    print("\n完成！")

