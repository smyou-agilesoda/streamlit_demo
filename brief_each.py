import ast
import logging
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
import yaml
# from PIL import Image


# 로깅 설정
logger = logging.getLogger("AutoRAG")

def convert_str(x):
    x = "".join(["score: ", str(round(x, 5)), "\n\n"])
    return x

def dict_to_markdown(d, level=1):
    """
    Convert a dictionary to a Markdown formatted string.

    :param d: Dictionary to convert
    :param level: Current level of heading (used for nested dictionaries)
    :return: Markdown formatted string
    """
    markdown = ""
    for key, value in d.items():
        if isinstance(value, dict):
            markdown += f"{'#' * level} {key}\n"
            markdown += dict_to_markdown(value, level + 1)
        elif isinstance(value, list):
            markdown += f"{'#' * level} {key}\n"
            for item in value:
                if isinstance(item, dict):
                    markdown += dict_to_markdown(item, level + 1)
                else:
                    markdown += f"- {item}\n"
        else:
            markdown += f"{'#' * level} {key}\n{value}\n"
    return markdown

def dict_to_markdown_table(data, key_column_name: str, value_column_name: str):
    # Check if the input is a dictionary
    if not isinstance(data, dict):
        raise ValueError("Input must be a dictionary")

    # Create the header of the table
    header = f"| {key_column_name} | {value_column_name} |\n| :---: | :-----: |\n"

    # Create the rows of the table
    rows = ""
    for key, value in data.items():
        rows += f"| {key} | {value} |\n"

    # Combine header and rows
    markdown_table = header + rows
    return markdown_table

def find_node_dir(trial_dir: str) -> List[str]:
    trial_summary_df = pd.read_csv(os.path.join(trial_dir, 'summary.csv'))
    result_paths = []
    for idx, row in trial_summary_df.iterrows():
        node_line_name = row['node_line_name']
        node_type = row['node_type']
        result_paths.append(os.path.join(trial_dir, node_line_name, node_type))
    return result_paths


def get_metric_values(node_summary_df: pd.DataFrame) -> Dict:
    non_metric_column_names = ['filename', 'module_name', 'module_params', 'execution_time', 'average_output_token',
                               'is_best']
    best_row = node_summary_df.loc[node_summary_df['is_best']].drop(columns=non_metric_column_names, errors='ignore')
    assert len(best_row) == 1, "The best module must be only one."
    return best_row.iloc[0].to_dict()

def make_trial_summary_md(trial_dir, trial):
    markdown_text = f"""# Trial Result Summary
- Trial : {trial}

"""
    node_dirs = find_node_dir(trial_dir)
    for node_dir in node_dirs:
        node_summary_filepath = os.path.join(node_dir, 'summary.csv')
        node_type = os.path.basename(node_dir)
        node_summary_df = pd.read_csv(node_summary_filepath)
        best_row = node_summary_df.loc[node_summary_df['is_best']].iloc[0]
        metric_dict = get_metric_values(node_summary_df)
        markdown_text += f"""---

## {node_type} best module

### Module Name

{best_row['module_name']}

### Module Params

{dict_to_markdown(ast.literal_eval(best_row['module_params']), level=3)}

### Metric Values

{dict_to_markdown_table(metric_dict, key_column_name='metric_name', value_column_name='metric_value')}

"""

    return markdown_text

def yaml_to_markdown(yaml_filepath):
    markdown_content = ""
    with open(yaml_filepath, 'r', encoding='utf-8') as file:
        try:
            content = yaml.safe_load(file)
            markdown_content += f"## {os.path.basename(yaml_filepath)}\n```yaml\n{yaml.dump(content, allow_unicode=True)}\n```\n\n"
        except yaml.YAMLError as exc:
            print(f"Error in {yaml_filepath}: {exc}")
    return markdown_content

@st.cache_data
def summary_plot_df(summary_df):
    non_metric_column_names = ['filename', 'module_name', 'module_params', 'execution_time', 'average_output_token', 'is_best']
    metric_df = summary_df.drop(columns=non_metric_column_names, errors='ignore')
    metric_df_melted = metric_df.reset_index().melt(id_vars='index', var_name='metric', value_name='value')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
    sns.stripplot(data=metric_df_melted, x='metric', y='value', hue='index', ax=ax1, palette='nipy_spectral')
    ax1.set_title("Strip Plot")
    sns.boxplot(data=metric_df_melted, x='metric', y='value', ax=ax2)
    ax2.set_title("Box Plot")
    st.pyplot(fig)

# # generator metric 상관관계 확인용
# @st.cache_data
# def corr_plot_df(result_df):
#     non_metric_column_names = ['generated_texts', 'generated_tokens', 'generated_log_probs']
#     metric_df = result_df.drop(columns=non_metric_column_names, errors='ignore')
#     fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
#     sns.heatmap(data = metric_df.corr(), annot=True, fmt = '.2f', linewidths=.5, cmap='Blues')
#     st.pyplot(fig)
    

@st.cache_data
def merge_split_df(df, filepath):
    module_df = pd.read_parquet(filepath, engine='pyarrow')
    df = pd.concat([df[["contents", "query", "generation_gt"]], module_df], axis=1)
    return df

@st.cache_data
def split_retrieval_df(df):
    df.pop("retrieved_ids")
    contents = df.pop("retrieved_contents")
    scores = df.pop("retrieve_scores")
    scores = scores.apply(lambda lst: list(map(convert_str, lst)))
    contents = scores + contents
    df = pd.concat([pd.DataFrame(contents.tolist()), df], axis=1)
    return df

@st.cache_data
def del_generated_token_df(df):
    df.pop("generated_tokens")
    df.pop("generated_log_probs")
    return df

doc_name = "[KISTEP 브리프] 감염병 백신치료"

# sidebar 데이터 목록 로드
base_path = os.path.abspath(".")
trial_path = os.path.join(base_path, "benchmark", doc_name)
trial_list = os.listdir(trial_path)
trial_list = [x for x in trial_list if os.path.isdir(os.path.join(trial_path, x))]
trial_dict = {"_".join(x.split("_")[:-1]):x for x in trial_list}

_trial = st.sidebar.radio('선택하세요', trial_dict.keys())

# selected data path
trial = os.path.join(trial_path, trial_dict.get(_trial))
trial_dir = os.path.join(trial, "0")
data_dir = os.path.join(trial, "data")

# resource 
corpus_df = pd.read_parquet(os.path.join(data_dir, "corpus.parquet"), engine='pyarrow')
qa_df = pd.read_parquet(os.path.join(data_dir, "qa.parquet"), engine='pyarrow')
qa_df["retrieval_gt"] = qa_df.retrieval_gt.apply(lambda x: x[0][0])
qa_df["generation_gt"] = qa_df.generation_gt.apply(lambda x: x[0])
merged_df = pd.merge(qa_df, corpus_df, left_on='retrieval_gt', right_on='doc_id')
merged_df = merged_df[["contents", "query", "generation_gt"]]

if not trial_dir:
    st.stop()

# 탭 생성
tabs = st.tabs(["Summary"] + [os.path.basename(node_dir) for node_dir in find_node_dir(trial_dir)] + ["Used YAML file"])

# Summary 탭
with tabs[0]:
    trial_summary_md = make_trial_summary_md(trial_dir=trial_dir, trial=_trial)
    st.markdown(trial_summary_md)

# 각 노드 탭
for i, node_dir in enumerate(find_node_dir(trial_dir), start=1):
    with tabs[i]:
        node = os.path.basename(node_dir)
        st.header(f"Node: {node}")
        
        # Summary DataFrame
        summary_df = pd.read_csv(os.path.join(node_dir, 'summary.csv'))
        best_idx = summary_df[summary_df["is_best"]].index[0]
        columns = summary_df.columns.tolist()
        columns = columns[-1:] + columns[:-1]
        summary_df = summary_df[columns]

        # Summary Distribution Plots
        st.subheader("Summary Distribution Plots")
        try:
            summary_plot_df(summary_df)
        except Exception as e:
            logger.error(f'Skipping make boxplot and stripplot with error {e}')
            st.error("Error creating plots")

        st.subheader("Summary DataFrame")
        st.dataframe(summary_df, hide_index=True)

        # Module Result DataFrame
        st.subheader("Module Result DataFrame")
        #TODO) 기본값 설정 필요
        module_selector = st.selectbox(
            "Select a module", 
            summary_df['filename'], 
            key=f"module_selector_{i}",
            index=int(best_idx)
        )
        if module_selector:
            filepath = os.path.join(node_dir, module_selector)
            module_df = pd.read_parquet(filepath, engine='pyarrow')
            if node == "retrieval" or node == "passage_reranker":
                module_df = pd.concat([merged_df, split_retrieval_df(module_df)], axis=1)
            elif node == "generator":
                module_df = pd.concat([merged_df, del_generated_token_df(module_df)], axis=1)
            st.dataframe(module_df)

# YAML 파일 탭
with tabs[-1]:
    yaml_file_markdown = yaml_to_markdown(os.path.join(trial_dir, "config.yaml"))
    st.markdown(yaml_file_markdown)
