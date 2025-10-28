#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
西游记角色别名深度分析
分析不同别名的使用场景、上下文和语义特征
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import os
import re
from collections import Counter, defaultdict
import jieba
import jieba.analyse
from wordcloud import WordCloud

# 设置matplotlib配置
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Hiragino Sans GB', 'STHeiti', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

# 分析的别名
SUN_WUKONG_ALIASES = ["弼馬溫", "石猴", "鬥戰勝佛"]
ZHU_BAJIE_ALIASES = ["天蓬元帥", "豬剛鬣", "悟能"]

# 停用词列表（用于过滤无意义词汇）
STOP_WORDS = set([
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
    '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
    '自己', '这', '那', '与', '之', '其', '中', '又', '为', '以', '将', '个', '道',
    '只', '见', '把', '他', '时', '来', '两', '便', '向', '却', '里', '边', '下',
    '么', '多', '然', '可', '些', '等', '此', '而', '若', '乃', '方', '即', '使'
])


class AliasAnalyzer:
    """别名分析器"""
    
    def __init__(self, chapters_dir):
        self.chapters_dir = Path(chapters_dir)
        self.chapters_data = {}
        self.load_chapters()
    
    def load_chapters(self):
        """加载所有章节文本"""
        print("加载章节数据...")
        for i in range(1, 101):
            chapter_file = self.chapters_dir / f"{i}-chapter.txt"
            if chapter_file.exists():
                with open(chapter_file, 'r', encoding='utf-8') as f:
                    self.chapters_data[i] = f.read()
        print(f"已加载 {len(self.chapters_data)} 个章节")
    
    def extract_context(self, alias, window_size=50):
        """
        提取别名出现的上下文
        
        参数:
            alias: 别名
            window_size: 前后文字数
        
        返回:
            list of dict: 包含章节号、位置、前文、别名、后文
        """
        contexts = []
        for chapter_num, text in self.chapters_data.items():
            # 找到所有出现位置
            start = 0
            while True:
                pos = text.find(alias, start)
                if pos == -1:
                    break
                
                # 提取前后文
                before = text[max(0, pos-window_size):pos]
                after = text[pos+len(alias):min(len(text), pos+len(alias)+window_size)]
                
                contexts.append({
                    'chapter': chapter_num,
                    'position': pos,
                    'before': before,
                    'alias': alias,
                    'after': after,
                    'full_context': before + alias + after
                })
                
                start = pos + 1
        
        return contexts
    
    def extract_ngrams(self, contexts, n=2):
        """
        提取别名周围的n-gram
        
        参数:
            contexts: 上下文列表
            n: n-gram的n值（2=bigram, 3=trigram）
        
        返回:
            Counter: n-gram频率统计
        """
        ngrams = []
        for ctx in contexts:
            # 分词
            words_before = list(jieba.cut(ctx['before']))
            words_after = list(jieba.cut(ctx['after']))
            
            # 提取前面的n-gram
            if len(words_before) >= n-1:
                ngram = ' '.join(words_before[-(n-1):]) + ' ' + ctx['alias']
                ngrams.append(ngram)
            
            # 提取后面的n-gram
            if len(words_after) >= n-1:
                ngram = ctx['alias'] + ' ' + ' '.join(words_after[:n-1])
                ngrams.append(ngram)
        
        return Counter(ngrams)
    
    def get_context_words(self, contexts):
        """
        获取上下文中的所有词汇（用于词云）
        
        参数:
            contexts: 上下文列表
        
        返回:
            list: 词汇列表
        """
        words = []
        for ctx in contexts:
            # 分词并过滤停用词
            text = ctx['before'] + ctx['after']
            words_list = jieba.cut(text)
            words.extend([w for w in words_list 
                         if len(w.strip()) > 1 and w not in STOP_WORDS])
        return words
    
    def analyze_temporal_pattern(self, alias):
        """
        分析别名的时间分布模式
        
        返回:
            dict: 包含各种统计信息
        """
        counts = {}
        for chapter_num, text in self.chapters_data.items():
            counts[chapter_num] = text.count(alias)
        
        # 统计信息
        total = sum(counts.values())
        chapters_with_alias = [c for c, cnt in counts.items() if cnt > 0]
        
        if chapters_with_alias:
            first_chapter = min(chapters_with_alias)
            last_chapter = max(chapters_with_alias)
            peak_chapter = max(counts.keys(), key=lambda k: counts[k])
            peak_count = counts[peak_chapter]
        else:
            first_chapter = last_chapter = peak_chapter = peak_count = 0
        
        return {
            'alias': alias,
            'total_count': total,
            'chapter_span': len(chapters_with_alias),
            'first_appearance': first_chapter,
            'last_appearance': last_chapter,
            'peak_chapter': peak_chapter,
            'peak_count': peak_count,
            'distribution': counts
        }


def create_wordcloud(words, alias, output_file):
    """
    创建词云
    
    参数:
        words: 词汇列表
        alias: 别名（用于标题）
        output_file: 输出文件路径
    """
    # 统计词频
    word_freq = Counter(words)
    
    # 创建词云
    wc = WordCloud(
        font_path='/System/Library/Fonts/STHeiti Medium.ttc',
        width=1600,
        height=800,
        background_color='white',
        max_words=100,
        relative_scaling=0.5,
        colormap='viridis'
    ).generate_from_frequencies(word_freq)
    
    # 绘制
    plt.figure(figsize=(16, 8))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'"{alias}" 上下文词云', fontsize=24, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def analyze_character_aliases(analyzer, aliases, character_name, output_dir):
    """
    分析一组别名
    
    参数:
        analyzer: AliasAnalyzer实例
        aliases: 别名列表
        character_name: 角色名称
        output_dir: 输出目录
    """
    print(f"\n{'='*60}")
    print(f"分析 {character_name} 的别名")
    print(f"{'='*60}")
    
    results = []
    
    for alias in aliases:
        print(f"\n分析别名: {alias}")
        
        # 1. 提取上下文
        contexts = analyzer.extract_context(alias, window_size=50)
        print(f"  找到 {len(contexts)} 个出现位置")
        
        if len(contexts) == 0:
            print(f"  别名 '{alias}' 未找到，跳过")
            continue
        
        # 2. 时间分布分析
        temporal = analyzer.analyze_temporal_pattern(alias)
        print(f"  首次出现: 第{temporal['first_appearance']}章")
        print(f"  最后出现: 第{temporal['last_appearance']}章")
        print(f"  出现峰值: 第{temporal['peak_chapter']}章 ({temporal['peak_count']}次)")
        
        # 3. N-gram分析
        bigrams = analyzer.extract_ngrams(contexts, n=2)
        trigrams = analyzer.extract_ngrams(contexts, n=3)
        
        print(f"  最常见的bigram组合:")
        for ngram, count in bigrams.most_common(5):
            print(f"    {ngram}: {count}次")
        
        # 4. 词云生成
        words = analyzer.get_context_words(contexts)
        wordcloud_file = output_dir / f"{character_name}_{alias}_wordcloud.png"
        create_wordcloud(words, alias, wordcloud_file)
        print(f"  词云已保存: {wordcloud_file.name}")
        
        # 保存结果
        results.append({
            'alias': alias,
            'contexts': contexts,
            'temporal': temporal,
            'bigrams': bigrams,
            'trigrams': trigrams,
            'context_words': words
        })
    
    return results


def generate_summary_report(results, character_name, output_file):
    """
    生成分析摘要报告
    
    参数:
        results: 分析结果列表
        character_name: 角色名称
        output_file: 输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"{'='*80}\n")
        f.write(f"{character_name} 别名使用分析报告\n")
        f.write(f"{'='*80}\n\n")
        
        for result in results:
            alias = result['alias']
            temporal = result['temporal']
            
            f.write(f"\n{'='*60}\n")
            f.write(f"别名: {alias}\n")
            f.write(f"{'='*60}\n\n")
            
            # 基本统计
            f.write(f"1. 基本统计\n")
            f.write(f"   总出现次数: {temporal['total_count']}\n")
            f.write(f"   分布章节数: {temporal['chapter_span']}\n")
            f.write(f"   首次出现: 第{temporal['first_appearance']}章\n")
            f.write(f"   最后出现: 第{temporal['last_appearance']}章\n")
            f.write(f"   出现峰值: 第{temporal['peak_chapter']}章 ({temporal['peak_count']}次)\n\n")
            
            # 常见搭配
            f.write(f"2. 最常见的Bigram搭配 (前10)\n")
            for i, (ngram, count) in enumerate(result['bigrams'].most_common(10), 1):
                f.write(f"   {i}. {ngram}: {count}次\n")
            
            f.write(f"\n3. 最常见的Trigram搭配 (前10)\n")
            for i, (ngram, count) in enumerate(result['trigrams'].most_common(10), 1):
                f.write(f"   {i}. {ngram}: {count}次\n")
            
            # 上下文高频词
            f.write(f"\n4. 上下文高频词 (前20)\n")
            word_freq = Counter(result['context_words'])
            for i, (word, count) in enumerate(word_freq.most_common(20), 1):
                f.write(f"   {i}. {word}: {count}次\n")
            
            f.write("\n")


def main():
    """主函数"""
    print("="*80)
    print("西游记角色别名深度分析")
    print("="*80)
    
    # 设置路径
    base_dir = Path(__file__).parent
    chapters_dir = base_dir / "xiyouji_chapters"
    output_dir = base_dir / "analysis_results"
    output_dir.mkdir(exist_ok=True)
    
    # 初始化分析器
    analyzer = AliasAnalyzer(chapters_dir)
    
    # 分析孙悟空
    sun_results = analyze_character_aliases(
        analyzer, 
        SUN_WUKONG_ALIASES, 
        "孙悟空", 
        output_dir
    )
    
    # 生成孙悟空报告
    sun_report = output_dir / "孙悟空_别名分析报告.txt"
    generate_summary_report(sun_results, "孙悟空", sun_report)
    print(f"\n孙悟空分析报告已保存: {sun_report}")
    
    # 分析猪八戒
    zhu_results = analyze_character_aliases(
        analyzer,
        ZHU_BAJIE_ALIASES,
        "猪八戒",
        output_dir
    )
    
    # 生成猪八戒报告
    zhu_report = output_dir / "猪八戒_别名分析报告.txt"
    generate_summary_report(zhu_results, "猪八戒", zhu_report)
    print(f"\n猪八戒分析报告已保存: {zhu_report}")
    
    print("\n" + "="*80)
    print("分析完成！所有结果已保存到 analysis_results 目录")
    print("="*80)


if __name__ == "__main__":
    main()

