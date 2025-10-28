#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
西游记人物关系分析
基于TF-IDF提取别名相关的人物和关系
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import os
import jieba
import jieba.analyse
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx

# 设置matplotlib配置
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Hiragino Sans GB', 'STHeiti', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

# 停用词
STOP_WORDS = set([
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一個',
    '上', '也', '很', '到', '說', '要', '去', '你', '會', '著', '沒有', '看', '好',
    '自己', '這', '那', '與', '之', '其', '中', '又', '為', '以', '將', '個', '道',
    '只', '見', '把', '他', '時', '來', '兩', '便', '向', '卻', '裡', '邊', '下',
    '麼', '多', '然', '可', '些', '等', '此', '而', '若', '乃', '方', '即', '使',
    '說道', '怎麼', '如何', '不知', '原來', '起來', '出來', '過來', '回來',
    '什麼', '還是', '只見', '卻說', '只是', '如此', '因此', '於是'
])

# 常见人物名称（扩展列表）
COMMON_CHARACTERS = {
    # 取经团队
    '三藏', '唐僧', '唐三藏', '聖僧', '師父', '和尚', '禪師', '法師',
    '悟空', '行者', '孫行者', '大聖', '猴王', '美猴王', '齊天大聖', '石猴', '弼馬溫',
    '八戒', '悟能', '豬八戒', '豬悟能', '天蓬元帥', '豬剛鬣', '獃子', '呆子',
    '悟淨', '沙僧', '沙和尚', '沙悟淨', '捲簾大將',
    '白龍馬', '龍馬', '小白龍', '敖閏', '玉龍',
    
    # 神仙
    '觀音', '觀音菩薩', '菩薩', '南海觀音', '大士',
    '如來', '如來佛', '佛祖', '釋迦',
    '玉帝', '玉皇大帝', '天帝',
    '太上老君', '老君', '太上',
    '文殊', '文殊菩薩', '普賢', '普賢菩薩',
    '太白金星', '金星',
    '托塔天王', '李天王', '哪吒', '三太子',
    '二郎神', '楊戩',
    
    # 妖怪
    '妖精', '妖怪', '怪物', '魔王', '大王', '魔頭',
    '白骨精', '白骨夫人',
    '牛魔王', '鐵扇公主', '紅孩兒',
    '黃袍怪', '金角', '銀角',
    '蜘蛛精', '蠍子精',
    '黃眉怪', '青牛精',
    
    # 其他
    '龍王', '東海龍王',
    '閻王', '判官',
    '土地', '山神',
    '高老莊', '高太公', '高小姐', '高翠蘭',
    '嫦娥', '天蓬'
}


class CharacterRelationshipAnalyzer:
    """人物关系分析器"""
    
    def __init__(self, chapters_dir):
        self.chapters_dir = Path(chapters_dir)
        self.chapters_data = {}
        self.load_chapters()
    
    def load_chapters(self):
        """加载所有章节"""
        print("加载章节数据...")
        for i in range(1, 101):
            chapter_file = self.chapters_dir / f"{i}-chapter.txt"
            if chapter_file.exists():
                with open(chapter_file, 'r', encoding='utf-8') as f:
                    self.chapters_data[i] = f.read()
        print(f"已加载 {len(self.chapters_data)} 个章节")
    
    def extract_contexts(self, alias, window_size=100):
        """提取别名的所有上下文"""
        contexts = []
        for chapter_num, text in self.chapters_data.items():
            start = 0
            while True:
                pos = text.find(alias, start)
                if pos == -1:
                    break
                
                before = text[max(0, pos-window_size):pos]
                after = text[pos+len(alias):min(len(text), pos+len(alias)+window_size)]
                full_context = before + alias + after
                
                contexts.append({
                    'chapter': chapter_num,
                    'context': full_context
                })
                
                start = pos + 1
        
        return contexts
    
    def tfidf_analysis(self, alias, top_n=50):
        """
        对别名的上下文进行TF-IDF分析
        
        参数:
            alias: 别名
            top_n: 返回前N个高TF-IDF词
        
        返回:
            DataFrame: 词汇及其TF-IDF分数
        """
        print(f"\n分析别名: {alias}")
        
        # 提取上下文
        contexts = self.extract_contexts(alias, window_size=100)
        print(f"找到 {len(contexts)} 个出现位置")
        
        if len(contexts) == 0:
            print(f"未找到别名 '{alias}'")
            return None
        
        # 合并所有上下文
        alias_corpus = ' '.join([ctx['context'] for ctx in contexts])
        
        # 构建对比语料库（所有章节）
        all_chapters_text = ' '.join(self.chapters_data.values())
        
        # 分词
        alias_words = ' '.join(jieba.cut(alias_corpus))
        all_words = ' '.join(jieba.cut(all_chapters_text))
        
        # TF-IDF分析
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=list(STOP_WORDS),
            token_pattern=r'(?u)\b\w+\b'
        )
        
        # 文档：[别名上下文, 全文]
        tfidf_matrix = vectorizer.fit_transform([alias_words, all_words])
        feature_names = vectorizer.get_feature_names_out()
        
        # 获取别名上下文的TF-IDF分数
        alias_tfidf = tfidf_matrix[0].toarray().flatten()
        
        # 创建DataFrame
        tfidf_df = pd.DataFrame({
            'word': feature_names,
            'tfidf_score': alias_tfidf
        }).sort_values('tfidf_score', ascending=False)
        
        # 过滤掉别名本身
        tfidf_df = tfidf_df[tfidf_df['word'] != alias]
        
        return tfidf_df.head(top_n)
    
    def extract_characters_from_tfidf(self, tfidf_df, threshold=0.01):
        """
        从TF-IDF结果中提取人物名称
        
        参数:
            tfidf_df: TF-IDF DataFrame
            threshold: TF-IDF阈值
        
        返回:
            list: 识别出的人物名称
        """
        if tfidf_df is None:
            return []
        
        identified_characters = []
        
        for _, row in tfidf_df.iterrows():
            word = row['word']
            score = row['tfidf_score']
            
            if score < threshold:
                break
            
            # 检查是否在已知人物列表中
            if word in COMMON_CHARACTERS:
                identified_characters.append({
                    'name': word,
                    'score': score,
                    'type': 'known'
                })
            # 或者是2-4字的可能人名
            elif 2 <= len(word) <= 4 and not any(c in word for c in ['個', '們', '著', '過']):
                identified_characters.append({
                    'name': word,
                    'score': score,
                    'type': 'potential'
                })
        
        return identified_characters
    
    def analyze_co_occurrence(self, alias, characters, window_size=100):
        """
        分析别名与其他人物的共现关系
        
        参数:
            alias: 主要别名
            characters: 要分析的人物列表
            window_size: 上下文窗口大小
        
        返回:
            dict: 共现统计
        """
        co_occurrence = defaultdict(int)
        contexts = self.extract_contexts(alias, window_size)
        
        for ctx in contexts:
            context_text = ctx['context']
            for char_info in characters:
                char_name = char_info['name']
                if char_name in context_text:
                    co_occurrence[char_name] += 1
        
        return dict(co_occurrence)
    
    def visualize_tfidf(self, tfidf_df, alias, output_file, top_n=20):
        """可视化TF-IDF结果"""
        if tfidf_df is None or len(tfidf_df) == 0:
            print(f"没有TF-IDF数据可视化")
            return
        
        plt.figure(figsize=(14, 10))
        
        data = tfidf_df.head(top_n)
        
        # 标记已知人物
        colors = ['#FF6B6B' if word in COMMON_CHARACTERS else '#4ECDC4' 
                 for word in data['word']]
        
        plt.barh(range(len(data)), data['tfidf_score'], color=colors, alpha=0.7)
        plt.yticks(range(len(data)), data['word'], fontsize=12)
        plt.xlabel('TF-IDF 分数', fontsize=14, fontweight='bold')
        plt.title(f'"{alias}" 上下文的TF-IDF分析 (Top {top_n})\n红色=已知人物, 蓝色=其他高频词',
                 fontsize=16, fontweight='bold', pad=20)
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"TF-IDF可视化已保存: {output_file}")
    
    def build_relationship_network(self, alias, characters, co_occurrence, output_file):
        """
        构建人物关系网络图
        
        参数:
            alias: 中心别名
            characters: 人物列表
            co_occurrence: 共现统计
            output_file: 输出文件
        """
        G = nx.Graph()
        
        # 添加中心节点
        G.add_node(alias, node_type='center')
        
        # 添加相关人物节点和边
        for char_name, count in co_occurrence.items():
            if count > 0:
                G.add_node(char_name, node_type='character')
                G.add_edge(alias, char_name, weight=count)
        
        # 可视化
        plt.figure(figsize=(16, 12))
        
        # 布局
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # 节点大小和颜色
        node_sizes = []
        node_colors = []
        for node in G.nodes():
            if node == alias:
                node_sizes.append(3000)
                node_colors.append('#FF6B6B')
            else:
                # 根据共现次数调整大小
                count = co_occurrence.get(node, 1)
                node_sizes.append(500 + count * 100)
                node_colors.append('#4ECDC4')
        
        # 边的宽度
        edge_widths = [G[u][v]['weight'] * 0.5 for u, v in G.edges()]
        
        # 绘制
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                              alpha=0.7, edgecolors='black', linewidths=2)
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, edge_color='gray')
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold',
                               font_family='Arial Unicode MS')
        
        # 边标签（共现次数）
        edge_labels = {(u, v): f"{G[u][v]['weight']}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10)
        
        plt.title(f'"{alias}" 人物关系网络图\n(数字表示共现次数)',
                 fontsize=18, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"关系网络图已保存: {output_file}")


def main():
    """主函数"""
    print("="*80)
    print("西游记人物关系分析 - 基于TF-IDF")
    print("="*80)
    
    # 设置路径
    base_dir = Path(__file__).parent
    chapters_dir = base_dir / "xiyouji_chapters"
    output_dir = base_dir / "relationship_analysis"
    output_dir.mkdir(exist_ok=True)
    
    # 初始化分析器
    analyzer = CharacterRelationshipAnalyzer(chapters_dir)
    
    # 分析的别名
    aliases_to_analyze = {
        '孙悟空': ['弼馬溫', '石猴', '鬥戰勝佛'],
        '猪八戒': ['豬剛鬣', '天蓬元帥', '悟能']
    }
    
    all_results = {}
    
    for character, aliases in aliases_to_analyze.items():
        print(f"\n{'='*60}")
        print(f"分析角色: {character}")
        print(f"{'='*60}")
        
        character_results = {}
        
        for alias in aliases:
            # 1. TF-IDF分析
            tfidf_df = analyzer.tfidf_analysis(alias, top_n=50)
            
            if tfidf_df is not None and len(tfidf_df) > 0:
                # 保存TF-IDF结果
                tfidf_file = output_dir / f"{alias}_tfidf.csv"
                tfidf_df.to_csv(tfidf_file, index=False, encoding='utf-8-sig')
                print(f"TF-IDF结果已保存: {tfidf_file}")
                
                # 可视化TF-IDF
                tfidf_viz_file = output_dir / f"{alias}_tfidf_visualization.png"
                analyzer.visualize_tfidf(tfidf_df, alias, tfidf_viz_file, top_n=20)
                
                # 2. 提取人物
                characters = analyzer.extract_characters_from_tfidf(tfidf_df, threshold=0.01)
                print(f"\n识别出 {len(characters)} 个相关人物:")
                for char in characters[:10]:
                    print(f"  - {char['name']}: {char['score']:.4f} ({char['type']})")
                
                # 3. 共现分析
                co_occurrence = analyzer.analyze_co_occurrence(alias, characters, window_size=100)
                print(f"\n共现统计 (前10):")
                sorted_co = sorted(co_occurrence.items(), key=lambda x: x[1], reverse=True)
                for name, count in sorted_co[:10]:
                    print(f"  - {name}: {count}次")
                
                # 4. 构建关系网络
                if len(co_occurrence) > 0:
                    network_file = output_dir / f"{alias}_relationship_network.png"
                    analyzer.build_relationship_network(alias, characters, co_occurrence, network_file)
                
                character_results[alias] = {
                    'tfidf': tfidf_df,
                    'characters': characters,
                    'co_occurrence': co_occurrence
                }
        
        all_results[character] = character_results
    
    print("\n" + "="*80)
    print("分析完成！所有结果已保存到 relationship_analysis 目录")
    print("="*80)


if __name__ == "__main__":
    main()

