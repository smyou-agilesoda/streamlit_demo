import os
from typing import List
from functools import partial

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def find_node_dir(trial_dir: str) -> List[str]:
    trial_summary_df = pd.read_csv(os.path.join(trial_dir, 'summary.csv'))
    result_paths = []
    for idx, row in trial_summary_df.iterrows():
        node_line_name = row['node_line_name']
        node_type = row['node_type']
        result_paths.append(os.path.join(trial_dir, node_line_name, node_type))
    return result_paths

# 데이터 로드
@st.cache_data
def load_data():
    data = pd.read_csv('your_data.csv')
    return data

@st.cache_data
def select_rows(selection, df):
    idx_list = selection["selection"]["rows"]
    
    if idx_list == []:
        idx_list = list(range(len(df)))
    
    df_melted = df.iloc[idx_list]
        
    non_metric_column_names = ['filename', 'module_name', 'module_params', 'execution_time', 'is_best']
    df_melted = df_melted.drop(columns=non_metric_column_names, errors='ignore')
    df_melted = df_melted.reset_index().melt(id_vars='index', var_name='metric', value_name='value')
    
    return df_melted

def _update_selection(node):
    st.session_state.selected_indices = st.session_state[f"{node}_selection"]["selection"]["rows"]
    st.write(st.session_state)
    st.rerun()

st.title("Total Dashboard")

doc_name = "(S&T GPS)AI+휴머노이드+로봇+동향+및+시사점"

base_path = os.path.abspath(".")
trial_path = os.path.join(base_path, "benchmark", doc_name)
trial_list = os.listdir(trial_path)
trial_list = [x for x in trial_list if os.path.isdir(os.path.join(trial_path, x))]
trial_list = [x for x in trial_list if not x.startswith("_")]
trial_dict = {"_".join(x.split("_")[:-1]):x for x in trial_list}

node_names = ["retrieval", "passage_reranker"]

tabs = st.tabs(node_names)
for tab, node in zip(tabs, node_names):
    summary_concat_df = pd.DataFrame()
    update_selection = partial(_update_selection, node)
    with tab:
        for _trial in trial_dict.keys():
            trial = os.path.join(trial_path, trial_dict.get(_trial))
            result = os.path.join(trial, "0")
            node_summary_filepath = os.path.join(result, "retrieve_node_line", node, 'summary.csv')
            node_summary_df = pd.read_csv(node_summary_filepath)
            node_summary_df = node_summary_df.rename(index={0:_trial})
            best_row = node_summary_df.loc[node_summary_df['is_best']].iloc[0]
            best_row.name = _trial
            best_row = best_row.to_frame().transpose()
            summary_concat_df = pd.concat([summary_concat_df, best_row])        
        
        plot_space = st.empty()
        selection = st.dataframe(
            summary_concat_df,
            on_select="rerun",
        )
        selected_df = select_rows(selection, summary_concat_df)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.stripplot(data=selected_df, x='metric', y='value', hue='index', ax=ax, palette='nipy_spectral')
        plot_space.pyplot(fig)
        
        # selection = st.dataframe(
        #     summary_concat_df,
        #     on_select=update_selection,
        #     key=f"selection_{node}"
        # )
        # st.write(type(selection))
        # idx_list = selection["selection"]["rows"]
        # selected_df = summary_concat_df.iloc[idx_list]
        # non_metric_column_names = ['filename', 'module_name', 'module_params', 'execution_time', 'is_best']
        # selected_df = selected_df.drop(columns=non_metric_column_names, errors='ignore')
        # selected_df = selected_df.reset_index().melt(id_vars='index', var_name='metric', value_name='value')
        # if not selected_df.empty:
        #     fig, ax = plt.subplots(figsize=(12, 8))
        #     sns.stripplot(data=selected_df, x='metric', y='value', hue='index', ax=ax, palette='nipy_spectral')
        #     plot_space.pyplot(fig)
        # else:
        #     plot_space.info("데이터를 선택해주세요.")
        
        
        
        
        # selection = st.dataframe(
        #     summary_concat_df, 
        #     hide_index=True,
        #     on_select="rerun", 
        # )
        # idx = selection["selection"]["rows"]
        # st.write(idx)
        # st.dataframe(node_summary_df.iloc[idx])