#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于峰值章节的TF-IDF人物关系分析
策略：找到别名出现最频繁的章节（peak），然后在peak±1的范围内分析TF-IDF
"""

import pandas as pd
from pathlib import Path
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Hiragino Sans GB', 'STHeiti', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 停用词
STOP_WORDS = set([
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一個',
    '上', '也', '很', '到', '說', '要', '去', '你', '會', '著', '沒有', '看', '好',
    '自己', '這', '那', '與', '之', '其', '中', '又', '為', '以', '將', '個', '道',
    '只', '見', '把', '他', '時', '來', '兩', '便', '向', '卻', '裡', '邊', '下',
    '麼', '多', '然', '可', '些', '等', '此', '而', '若', '乃', '方', '即', '使',
    '說道', '怎麼', '如何', '不知', '原來', '起來', '出來', '過來', '回來',
    '什麼', '還是', '只見', '卻說', '只是', '如此', '因此', '於是', '那裡'
])

# 已知人物名称
CHARACTERS = {
    '三藏', '唐僧', '聖僧', '師父', '禪師', '法師',
    '悟空', '行者', '孫行者', '大聖', '猴王', '美猴王', '石猴', '弼馬溫',
    '八戒', '悟能', '豬八戒', '天蓬元帥', '豬剛鬣', '獃子', '呆子',
    '悟淨', '沙僧', '沙和尚', '捲簾大將',
    '觀音', '菩薩', '如來', '佛祖', '玉帝', '老君',
    '妖精', '妖怪', '那怪', '大王', '魔王',
    '龍王', '土地', '山神', '高老莊', '高太公', '嫦娥',
    '老孫', '師兄', '那廝'
}


class PeakBasedTFIDFAnalyzer:
    """基于峰值的TF-IDF分析器"""
    
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
    
    def find_peak_chapter(self, alias):
        """
        找到别名出现最频繁的章节（峰值）
        
        返回:
            dict: 包含峰值信息的字典
        """
        counts = {}
        for chapter_num, text in self.chapters_data.items():
            count = text.count(alias)
            if count > 0:
                counts[chapter_num] = count
        
        if not counts:
            return None
        
        peak_chapter = max(counts.keys(), key=lambda k: counts[k])
        peak_count = counts[peak_chapter]
        
        return {
            'peak_chapter': peak_chapter,
            'peak_count': peak_count,
            'all_counts': counts,
            'total_occurrences': sum(counts.values()),
            'chapter_span': len(counts)
        }
    
    def get_peak_range_text(self, alias, peak_info, window=1):
        """
        获取峰值章节±window范围内的文本
        
        参数:
            alias: 别名
            peak_info: 峰值信息
            window: 窗口大小（默认±1章）
        
        返回:
            str: 合并后的文本
        """
        peak_chapter = peak_info['peak_chapter']
        start_chapter = max(1, peak_chapter - window)
        end_chapter = min(100, peak_chapter + window)
        
        texts = []
        for ch in range(start_chapter, end_chapter + 1):
            if ch in self.chapters_data:
                texts.append(self.chapters_data[ch])
        
        return ' '.join(texts), list(range(start_chapter, end_chapter + 1))
    
    def peak_based_tfidf(self, alias, top_n=30, window=1):
        """
        基于峰值章节的TF-IDF分析
        
        参数:
            alias: 别名
            top_n: 返回前N个词
            window: 峰值章节±window范围
        
        返回:
            tuple: (tfidf_df, peak_info, analyzed_chapters)
        """
        print(f"\n{'='*60}")
        print(f"分析别名: {alias}")
        print(f"{'='*60}")
        
        # 1. 找峰值
        peak_info = self.find_peak_chapter(alias)
        
        if peak_info is None:
            print(f"未找到别名 '{alias}'")
            return None, None, None
        
        print(f"\n峰值信息:")
        print(f"  峰值章节: 第{peak_info['peak_chapter']}章")
        print(f"  峰值频率: {peak_info['peak_count']}次")
        print(f"  总出现次数: {peak_info['total_occurrences']}次")
        print(f"  分布章节数: {peak_info['chapter_span']}章")
        
        # 2. 获取峰值范围文本
        peak_range_text, analyzed_chapters = self.get_peak_range_text(alias, peak_info, window)
        print(f"  分析范围: 第{analyzed_chapters[0]}-{analyzed_chapters[-1]}章")
        
        # 3. 对比语料：全书
        all_text = ' '.join(self.chapters_data.values())
        
        # 4. 分词
        print(f"\n分词中...")
        peak_words = ' '.join(jieba.cut(peak_range_text))
        all_words = ' '.join(jieba.cut(all_text))
        
        # 5. TF-IDF
        print(f"计算TF-IDF...")
        vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words=list(STOP_WORDS),
            token_pattern=r'(?u)\b\w+\b'
        )
        
        tfidf_matrix = vectorizer.fit_transform([peak_words, all_words])
        feature_names = vectorizer.get_feature_names_out()
        peak_tfidf = tfidf_matrix[0].toarray().flatten()
        
        # 6. 结果
        tfidf_df = pd.DataFrame({
            'word': feature_names,
            'tfidf_score': peak_tfidf
        }).sort_values('tfidf_score', ascending=False)
        
        # 过滤别名本身
        tfidf_df = tfidf_df[tfidf_df['word'] != alias]
        
        return tfidf_df.head(top_n), peak_info, analyzed_chapters
    
    def visualize_tfidf(self, tfidf_df, alias, peak_info, analyzed_chapters, output_file, top_n=20):
        """可视化TF-IDF结果"""
        if tfidf_df is None or len(tfidf_df) == 0:
            print(f"没有数据可视化")
            return
        
        plt.figure(figsize=(14, 10))
        
        data = tfidf_df.head(top_n)
        
        # 标记人物
        colors = ['#FF6B6B' if word in CHARACTERS else '#4ECDC4' 
                 for word in data['word']]
        
        plt.barh(range(len(data)), data['tfidf_score'], color=colors, alpha=0.7, edgecolor='black')
        plt.yticks(range(len(data)), data['word'], fontsize=13)
        plt.xlabel('TF-IDF 分数', fontsize=14, fontweight='bold')
        
        title = f'"{alias}" 上下文的TF-IDF分析 (Top {top_n})\n'
        title += f'峰值章节: 第{peak_info["peak_chapter"]}章 ({peak_info["peak_count"]}次) | '
        title += f'分析范围: 第{analyzed_chapters[0]}-{analyzed_chapters[-1]}章\n'
        title += '红色=人物名称, 蓝色=其他词汇'
        
        plt.title(title, fontsize=15, fontweight='bold', pad=20)
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"TF-IDF可视化已保存: {output_file}")
    
    def create_relationship_network(self, tfidf_df, alias, peak_info, output_file):
        """创建人物关系网络图"""
        import networkx as nx
        
        if tfidf_df is None or len(tfidf_df) == 0:
            print("没有数据生成网络图")
            return
        
        # 筛选人物
        characters = tfidf_df[tfidf_df['word'].isin(CHARACTERS)]
        
        if len(characters) == 0:
            print("未识别出足够的人物名称")
            return
        
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
        
        title = f'"{alias}" 人物关系网络图\n'
        title += f'基于峰值章节（第{peak_info["peak_chapter"]}章）附近的TF-IDF分析\n'
        title += '边上的数字为TF-IDF分数'
        
        plt.title(title, fontsize=15, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"关系网络图已保存: {output_file}")


def analyze_all_aliases():
    """分析所有6个别名"""
    
    base_dir = Path(__file__).parent
    chapters_dir = base_dir / "xiyouji_chapters"
    output_dir = base_dir / "peak_tfidf_analysis"
    output_dir.mkdir(exist_ok=True)
    
    # 初始化分析器
    print("加载数据...")
    analyzer = PeakBasedTFIDFAnalyzer(chapters_dir)
    
    # 要分析的别名
    aliases_config = {
        '孙悟空': ['弼馬溫', '石猴', '鬥戰勝佛'],
        '猪八戒': ['豬剛鬣', '天蓬元帥', '悟能']
    }
    
    all_results = {}
    
    print("\n" + "="*80)
    print("基于峰值章节的TF-IDF人物关系分析")
    print("="*80)
    
    for character, aliases in aliases_config.items():
        print(f"\n\n{'#'*80}")
        print(f"# 角色: {character}")
        print(f"{'#'*80}")
        
        character_results = {}
        
        for alias in aliases:
            # TF-IDF分析
            tfidf_df, peak_info, analyzed_chapters = analyzer.peak_based_tfidf(
                alias, top_n=30, window=1
            )
            
            if tfidf_df is not None:
                # 保存CSV
                csv_file = output_dir / f"{alias}_peak_tfidf.csv"
                tfidf_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
                print(f"\nTF-IDF结果已保存: {csv_file}")
                
                # 打印Top 10
                print(f"\nTop 10 TF-IDF词汇:")
                for idx, row in tfidf_df.head(10).iterrows():
                    char_mark = "★" if row['word'] in CHARACTERS else " "
                    print(f"  {char_mark} {row['word']:8s}: {row['tfidf_score']:.4f}")
                
                # 可视化
                viz_file = output_dir / f"{alias}_peak_tfidf_viz.png"
                analyzer.visualize_tfidf(tfidf_df, alias, peak_info, 
                                        analyzed_chapters, viz_file, top_n=20)
                
                # 关系网络
                network_file = output_dir / f"{alias}_peak_network.png"
                analyzer.create_relationship_network(tfidf_df, alias, peak_info, network_file)
                
                character_results[alias] = {
                    'tfidf': tfidf_df,
                    'peak_info': peak_info,
                    'analyzed_chapters': analyzed_chapters
                }
        
        all_results[character] = character_results
    
    # 生成总结报告
    generate_summary_report(all_results, output_dir)
    
    print("\n" + "="*80)
    print("所有分析完成！结果保存在 peak_tfidf_analysis 目录")
    print("="*80)
    
    return all_results


def generate_summary_report(all_results, output_dir):
    """生成总结报告"""
    report_file = output_dir / "分析总结报告.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("基于峰值章节的TF-IDF人物关系分析报告\n")
        f.write("="*80 + "\n\n")
        
        for character, results in all_results.items():
            f.write(f"\n{'='*60}\n")
            f.write(f"{character}\n")
            f.write(f"{'='*60}\n\n")
            
            for alias, data in results.items():
                peak_info = data['peak_info']
                analyzed_chapters = data['analyzed_chapters']
                tfidf_df = data['tfidf']
                
                f.write(f"别名: {alias}\n")
                f.write(f"{'-'*40}\n")
                f.write(f"峰值章节: 第{peak_info['peak_chapter']}章\n")
                f.write(f"峰值频率: {peak_info['peak_count']}次\n")
                f.write(f"总出现次数: {peak_info['total_occurrences']}次\n")
                f.write(f"分析范围: 第{analyzed_chapters[0]}-{analyzed_chapters[-1]}章\n\n")
                
                f.write(f"Top 10 相关词汇:\n")
                for idx, row in tfidf_df.head(10).iterrows():
                    char_mark = "★[人物]" if row['word'] in CHARACTERS else ""
                    f.write(f"  {idx+1}. {row['word']:10s}: {row['tfidf_score']:.4f} {char_mark}\n")
                
                f.write("\n")
        
    print(f"\n总结报告已保存: {report_file}")


if __name__ == "__main__":
    analyze_all_aliases()

