#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"城市区域表征论文综述统计脚本（v2）
功能更新：
 - 增强字段清洗与规范化（方法/模态/任务标准化映射）
 - 更科研/报告风格的可视化（matplotlib 配置、viridis colormap、网格、图注）
 - 生成交叉热力图、条形图、饼图，并输出 PDF 报告
 - 支持输出中间标准化表（standardized_fields.csv）以便复查
\"\"\"

import os, re, argparse
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import Counter
from pathlib import Path

# ---------- Normalization maps ----------
METHOD_MAP = {
    # graph families
    'gcn':'GNN', 'gat':'GNN', 'graph attention':'GNN', 'graph neural network':'GNN', 'graphnet':'GNN',
    'graph convolution':'GNN', 'g.nn':'GNN', 'gcn/gat':'GNN', 'gcn/gat/graph':'GNN',
    # embedding / random-walk families
    'deepwalk':'RandomWalk/Word2Vec', 'node2vec':'RandomWalk/Word2Vec', 'word2vec':'Word2Vec', 'skip-gram':'Word2Vec',
    # contrastive / self-supervised
    'contrastive':'Contrastive Learning', 'self-supervised':'Self-Supervised', 'simclr':'Contrastive Learning',
    # transformer / attention
    'transformer':'Transformer', 'attention':'Transformer',
    # other
    'autoencoder':'AutoEncoder', 'ae':'AutoEncoder', 'clustering':'Clustering', 'rl':'Reinforcement Learning'
}

MODALITY_MAP = {
    'poi':'POI', 'point of interest':'POI', 'points of interest':'POI',
    'trajectory':'Trajectory', 'mobility':'Trajectory', 'human mobility':'Trajectory', 'flow':'Trajectory',
    'building':'Building Geometry', 'building footprint':'Building Geometry', 'osm':'OSM',
    'remote sensing':'Remote Sensing', 'satellite':'Remote Sensing', 'imagery':'Remote Sensing',
    'social':'Social Media', 'check-in':'Check-in', 'checkin':'Check-in',
    'census':'Census/Demographics', 'demographic':'Census/Demographics'
}

TASK_MAP = {
    'land use':'Land Use Classification', 'land-use':'Land Use Classification',
    'crime':'Crime Prediction', 'crime prediction':'Crime Prediction',
    'population':'Population Estimation', 'house price':'House Price Prediction', 'price':'House Price Prediction',
    'clustering':'Clustering', 'region similarity':'Similarity Search', 'recommend':'Recommendation'
}

# ---------- Helpers ----------
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def safe_split_multi(s):
    if pd.isna(s) or s is None:
        return []
    if not isinstance(s, str):
        s = str(s)
    parts = re.split(r'[,\;/\|\n，；、]+', s)
    return [p.strip() for p in parts if p.strip()!='']

def normalize_token(token, mapping):
    t = token.strip().lower()
    for k,v in mapping.items():
        if k in t:
            return v
    return token.strip()

def normalize_list_field(items, mapping):
    out = []
    for it in items:
        norm = normalize_token(it, mapping)
        out.append(norm)
    return sorted(list(dict.fromkeys(out)))  # unique, stable order

# ---------- Counting & cooccurrence ----------
def count_field_multi(df, col, mapping=None):
    cnt = Counter()
    for v in df[col].fillna(''):
        for item in safe_split_multi(v):
            if mapping:
                item = normalize_token(item, mapping)
            cnt[item] += 1
    return cnt

def build_cooccurrence(df, left_col, right_col, left_map=None, right_map=None):
    rows, cols, pairs = set(), set(), []
    for _, r in df[[left_col, right_col]].fillna('').iterrows():
        lefts = safe_split_multi(r[left_col])
        rights = safe_split_multi(r[right_col])
        for L in lefts:
            Lnorm = normalize_token(L, left_map) if left_map else L
            if Lnorm=='': continue
            rows.add(Lnorm)
            for R in rights:
                Rnorm = normalize_token(R, right_map) if right_map else R
                if Rnorm=='': continue
                cols.add(Rnorm)
                pairs.append((Lnorm, Rnorm))
    rows = sorted(rows); cols = sorted(cols)
    import pandas as pd
    mat = pd.DataFrame(0, index=rows, columns=cols)
    for L,R in pairs:
        mat.loc[L,R] += 1
    return mat

# ---------- Plotting (scientific style) ----------
def style_for_science():
    plt.rcParams.update({
        'font.size': 10,
        'figure.dpi': 150,
        'font.family': 'sans-serif',
        'axes.grid': True,
        'grid.color': '#EAEAF2',
        'grid.linestyle': '--',
        'axes.edgecolor': '#333333',
        'axes.linewidth': 0.8,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
    })

def save_bar(counter, title, outpath, topk=20):
    items = counter.most_common(topk)
    if not items:
        return None
    labels, values = zip(*items)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35*len(labels))))
    ax.barh(range(len(values))[::-1], values[::-1])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Count')
    plt.tight_layout()
    fig.savefig(outpath, bbox_inches='tight')
    plt.close(fig)
    return outpath

def save_pie(counter, title, outpath, topk=8):
    items = counter.most_common(topk)
    if not items:
        return None
    labels, values = zip(*items)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(outpath, bbox_inches='tight')
    plt.close(fig)
    return outpath

def save_heatmap(dfmat, title, outpath, cmap='viridis'):
    if dfmat.size == 0:
        return None
    fig, ax = plt.subplots(figsize=(10, max(4, 0.25*dfmat.shape[0])))
    im = ax.imshow(dfmat.values, aspect='auto', cmap=cmap)
    ax.set_yticks(range(dfmat.shape[0])); ax.set_yticklabels(dfmat.index)
    ax.set_xticks(range(dfmat.shape[1])); ax.set_xticklabels(dfmat.columns, rotation=45, ha='right')
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(outpath, bbox_inches='tight')
    plt.close(fig)
    return outpath

# ---------- Main workflow ----------
def analyze(input_excel, outdir, topk=20):
    ensure_dir(outdir)
    df = pd.read_excel(input_excel, engine='openpyxl')
    style_for_science()
    # Standardize and export standardized fields for review
    standardized = df.copy()
    # normalize methods/modalities/tasks to lists
    standardized['__methods_norm'] = standardized.apply(lambda r: ', '.join(normalize_list_field(safe_split_multi(r.get('表征学习方法（聚类/Word2Vec/GNN/Transformer/对比学习/RL等）','')), METHOD_MAP)), axis=1)
    standardized['__modalities_norm'] = standardized.apply(lambda r: ', '.join(normalize_list_field(safe_split_multi(r.get('数据模态（POI/轨迹/遥感/社交/建筑/交通等）','')), MODALITY_MAP)), axis=1)
    standardized['__tasks_norm'] = standardized.apply(lambda r: ', '.join(normalize_list_field(safe_split_multi(r.get('下游任务（功能识别/预测/推荐等）','')), TASK_MAP)), axis=1)
    standardized.to_csv(os.path.join(outdir, 'standardized_fields.csv'), index=False)
    # Counters
    methods_cnt = count_field_multi(standardized, '__methods_norm') if '__methods_norm' in standardized else None
    modalities_cnt = count_field_multi(standardized, '__modalities_norm') if '__modalities_norm' in standardized else None
    tasks_cnt = count_field_multi(standardized, '__tasks_norm') if '__tasks_norm' in standardized else None
    # Generate plots
    images = []
    if modalities_cnt:
        p = save_bar(modalities_cnt, '数据模态分布（Top）', os.path.join(outdir, 'modalities_bar.png'), topk=topk); images.append(p)
        p = save_pie(modalities_cnt, '数据模态占比（Top）', os.path.join(outdir, 'modalities_pie.png')); images.append(p)
    if methods_cnt:
        p = save_bar(methods_cnt, '方法类别分布（Top）', os.path.join(outdir, 'methods_bar.png'), topk=topk); images.append(p)
        p = save_pie(methods_cnt, '方法类别占比（Top）', os.path.join(outdir, 'methods_pie.png')); images.append(p)
    if tasks_cnt:
        p = save_bar(tasks_cnt, '下游任务分布（Top）', os.path.join(outdir, 'tasks_bar.png'), topk=topk); images.append(p)
        p = save_pie(tasks_cnt, '下游任务占比（Top）', os.path.join(outdir, 'tasks_pie.png')); images.append(p)
    # Heatmaps (methods x modalities, methods x tasks)
    mm = build_cooccurrence(standardized, '__methods_norm', '__modalities_norm')
    mt = build_cooccurrence(standardized, '__methods_norm', '__tasks_norm')
    if not mm.empty:
        p = save_heatmap(mm, '方法 × 模态 热力图', os.path.join(outdir, 'methods_by_modalities_heatmap.png')); images.append(p)
    if not mt.empty:
        p = save_heatmap(mt, '方法 × 任务 热力图', os.path.join(outdir, 'methods_by_tasks_heatmap.png')); images.append(p)
    # PDF report
    pdf_path = os.path.join(outdir, 'analysis_report_v2.pdf')
    with PdfPages(pdf_path) as pdf:
        for img in images:
            if img and os.path.exists(img):
                fig = plt.figure()
                plt.axis('off')
                im = plt.imread(img)
                plt.imshow(im)
                pdf.savefig(fig)
                plt.close(fig)
    print('报告生成：', pdf_path)
    return pdf_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--outdir', '-o', default='./analysis_output_v2')
    parser.add_argument('--topk', type=int, default=20)
    args = parser.parse_args()
    analyze(args.input, args.outdir, topk=args.topk)
