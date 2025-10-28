#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速TF-IDF分析 - 针对单个别名
"""

import pandas as pd
from pathlib import Path
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

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

def load_chapters(chapters_dir):
    """加载章节"""
    chapters_data = {}
    for i in range(1, 101):
        chapter_file = Path(chapters_dir) / f"{i}-chapter.txt"
        if chapter_file.exists():
            with open(chapter_file, 'r', encoding='utf-8') as f:
                chapters_data[i] = f.read()
    return chapters_data

def extract_contexts(chapters_data, alias, window_size=100):
    """提取上下文"""
    contexts = []
    for chapter_num, text in chapters_data.items():
        start = 0
        while True:
            pos = text.find(alias, start)
            if pos == -1:
                break
            before = text[max(0, pos-window_size):pos]
            after = text[pos+len(alias):min(len(text), pos+len(alias)+window_size)]
            contexts.append(before + alias + after)
            start = pos + 1
    return contexts

def tfidf_analysis(alias, chapters_dir, top_n=30):
    """TF-IDF分析"""
    print(f"加载数据...")
    chapters_data = load_chapters(chapters_dir)
    
    print(f"提取 '{alias}' 的上下文...")
    contexts = extract_contexts(chapters_data, alias)
    print(f"找到 {len(contexts)} 个出现位置")
    
    if len(contexts) == 0:
        print(f"未找到别名 '{alias}'")
        return None
    
    # 合并上下文
    alias_corpus = ' '.join(contexts)
    all_text = ' '.join(chapters_data.values())
    
    # 分词
    print("分词中...")
    alias_words = ' '.join(jieba.cut(alias_corpus))
    all_words = ' '.join(jieba.cut(all_text))
    
    # TF-IDF
    print("计算TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words=list(STOP_WORDS),
        token_pattern=r'(?u)\b\w+\b'
    )
    
    tfidf_matrix = vectorizer.fit_transform([alias_words, all_words])
    feature_names = vectorizer.get_feature_names_out()
    alias_tfidf = tfidf_matrix[0].toarray().flatten()
    
    # 结果
    tfidf_df = pd.DataFrame({
        'word': feature_names,
        'tfidf_score': alias_tfidf
    }).sort_values('tfidf_score', ascending=False)
    
    # 过滤别名本身
    tfidf_df = tfidf_df[tfidf_df['word'] != alias]
    
    return tfidf_df.head(top_n)

def main():
    # 分析豬剛鬣
    alias = '豬剛鬣'
    chapters_dir = Path(__file__).parent / "xiyouji_chapters"
    output_dir = Path(__file__).parent / "relationship_analysis"
    output_dir.mkdir(exist_ok=True)
    
    print(f"="*60)
    print(f"TF-IDF 分析: {alias}")
    print(f"="*60)
    
    result = tfidf_analysis(alias, chapters_dir, top_n=30)
    
    if result is not None:
        # 保存结果
        output_file = output_dir / f"{alias}_tfidf_top30.csv"
        result.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"\n=== Top 30 TF-IDF 词汇 ===")
        print(result.to_string(index=False))
        print(f"\n结果已保存: {output_file}")
        
        # 识别可能的人物
        print(f"\n=== 可能的人物名称 (TF-IDF > 0.01) ===")
        characters = result[result['tfidf_score'] > 0.01]
        for _, row in characters.iterrows():
            word = row['word']
            score = row['tfidf_score']
            if 2 <= len(word) <= 4:
                print(f"  {word}: {score:.4f}")

if __name__ == "__main__":
    main()

