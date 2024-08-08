import os
import ast
from typing import List, Dict

import yaml
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# benchmark = "benchmark_temp_non_gen"
# benchmark = "benchmark_less_metric"
# benchmark = "benchmark_rk3"
benchmark = "benchmark"

def update_selection(node):
    st.session_state.selected_indices = st.session_state[f"{node}_selection"]["selection"]["rows"]
    st.write(st.session_state)
    st.rerun()


def get_trial_path(doc_name):
    base_path = os.path.abspath(".")
    trial_path = os.path.join(base_path, benchmark, doc_name)
    return trial_path


def get_trial_info(trial_path):
    trial_list = os.listdir(trial_path)
    trial_list = [x for x in trial_list if os.path.isdir(os.path.join(trial_path, x))]
    trial_list = [x for x in trial_list if not x.startswith("_")]
    # trial_dict = {"_".join(x.split("_")[:-1]):x for x in trial_list}
    return trial_list


def get_best_row(file_path, trial):
    summary_df = pd.read_csv(file_path)
    best_row = summary_df.loc[summary_df['is_best']].iloc[0]
    best_row.name = "_".join(trial.split("_")[:-1])
    best_row.drop(["filename", "is_best"], inplace=True)
    best_row = best_row.to_frame().transpose()
    return best_row


def select_rows(selection, df):
    non_metric_column_names = ['filename', 'module_name', 'module_params', 'execution_time', 'is_best']
    
    idx_list = selection["selection"]["rows"]
    if idx_list == []: idx_list = list(range(len(df)))
    
    df_melted = df.iloc[idx_list]
    df_melted = df_melted.drop(columns=non_metric_column_names, errors='ignore')
    df_melted = df_melted.reset_index().melt(id_vars='index', var_name='metric', value_name='score')
    return df_melted


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


def make_trial_summary_md(trial_dir, markdown_text):
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
    metric_df_melted = metric_df.reset_index().melt(id_vars='index', var_name='metric', value_name='score')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
    sns.stripplot(data=metric_df_melted, x='metric', y='score', hue='index', ax=ax1, palette='nipy_spectral')
    ax1.set_title("Strip Plot")
    sns.boxplot(data=metric_df_melted, x='metric', y='score', ax=ax2)
    ax2.set_title("Box Plot")
    st.pyplot(fig)



@st.cache_data
def split_retrieval_df(df):
    def convert_str(x):
        return "".join(["score: ", str(round(x, 5)), "\n\n"])
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


@st.cache_data
def get_corpus(file_path):
    corpus_df = pd.read_parquet(os.path.join(file_path, "corpus.parquet"), engine='pyarrow')
    qa_df = pd.read_parquet(os.path.join(file_path, "qa.parquet"), engine='pyarrow')
    qa_df["retrieval_gt"] = qa_df.retrieval_gt.apply(lambda x: x[0][0])
    qa_df["generation_gt"] = qa_df.generation_gt.apply(lambda x: x[0])
    merged_df = pd.merge(qa_df, corpus_df, left_on='retrieval_gt', right_on='doc_id')
    return merged_df[["contents", "query", "generation_gt"]]


@st.cache_data
def get_summary_df(node_dir):
    summary_df = pd.read_csv(os.path.join(node_dir, 'summary.csv'))
    best_idx = summary_df[summary_df["is_best"]].index[0]
    columns = summary_df.columns.tolist()
    columns = columns[-1:] + columns[:-1]
    summary_df = summary_df[columns]
    return best_idx, summary_df