import os
from functools import partial

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    select_rows, 
    update_selection, 
    get_trial_info,
    get_trial_path,
    get_best_row,
)

st.title("Total Dashboard")

doc_name = "(조사평가)최종보고서_AI 기반 국가연구개발사업 평가지원체계 구축방안 연구"
node_names = ["retrieval", "passage_reranker"]

trial_path = get_trial_path(doc_name)
trial_dict = get_trial_info(trial_path)

tabs = st.tabs(node_names)
for tab, node in zip(tabs, node_names):
    node_summary_df = pd.DataFrame()
    # update_selection_node = partial(update_selection, node)
    with tab:
        for _trial in trial_dict.keys():
            node_summary_filepath = os.path.join(
                trial_path, 
                trial_dict.get(_trial), 
                "0",
                "retrieve_node_line",
                node,
                "summary.csv"
            )
            best_row = get_best_row(node_summary_filepath, _trial)
            node_summary_df = pd.concat([node_summary_df, best_row])
        
        plot_space = st.empty()
        selection = st.dataframe(node_summary_df, on_select="rerun")
        selected_df = select_rows(selection, node_summary_df)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.stripplot(data=selected_df, x='metric', y='score', hue='index', ax=ax, palette='nipy_spectral')
        plot_space.pyplot(fig)
        
        # selection = st.dataframe(
        #     node_summary_df,
        #     on_select=update_selection_node,
        #     key=f"selection_{node}"
        # )
        # st.write(type(selection))
        # idx_list = selection["selection"]["rows"]
        # selected_df = node_summary_df.iloc[idx_list]
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
        #     node_summary_df, 
        #     hide_index=True,
        #     on_select="rerun", 
        # )
        # idx = selection["selection"]["rows"]
        # st.write(idx)
        # st.dataframe(node_summary_df.iloc[idx])