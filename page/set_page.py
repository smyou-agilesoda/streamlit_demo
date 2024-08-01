import os
import logging

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    select_rows, 
    get_trial_info,
    get_trial_path,
    get_best_row,
    get_corpus,
    make_trial_summary_md,
    find_node_dir,
    summary_plot_df,
    split_retrieval_df,
    del_generated_token_df,
    yaml_to_markdown,
    get_summary_df,
)

def set_total(doc_name):
    trial_path = get_trial_path(doc_name)
    trial_dict = get_trial_info(trial_path)
    if not os.listdir(trial_path): st.stop()

    node_names = ["retrieval", "passage_reranker"]
    tabs = st.tabs(node_names)
    for tab, node in zip(tabs, node_names):
        node_summary_df = pd.DataFrame()
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


def set_trial(doc_name):
    # 로깅 설정
    logger = logging.getLogger("AutoRAG")

    # sidebar 데이터 목록 로드
    trial_path = get_trial_path(doc_name)
    trial_dict = get_trial_info(trial_path)
    if not os.listdir(trial_path): st.stop()

    _trial = st.sidebar.radio('선택하세요', trial_dict.keys())

    # trial data path
    trial_dir = os.path.join(trial_path, trial_dict.get(_trial), "0")
    data_dir = os.path.join(trial_path, trial_dict.get(_trial), "data")

    # resource 
    merged_df = get_corpus(data_dir)

    if not trial_dir: st.stop()

    # 탭 생성
    tabs = ["Summary"] + [os.path.basename(node_dir) for node_dir in find_node_dir(trial_dir)] + ["Used YAML file"]
    tabs = [tab for tab in tabs if tab != "prompt_maker"]
    tabs = st.tabs(tabs)

    # Summary 탭
    with tabs[0]:
        markdown_text = f"""# Trial Result Summary
- Trial : {_trial}

"""
        trial_summary_md = make_trial_summary_md(trial_dir, markdown_text)
        st.markdown(trial_summary_md)

    # 각 노드 탭
    for i, node_dir in enumerate(find_node_dir(trial_dir), start=1):
        with tabs[i]:
            node = os.path.basename(node_dir)
            st.header(f"Node: {node}")
            
            best_idx, summary_df = get_summary_df(node_dir)

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
            module_selector = st.selectbox(
                "Select a module", 
                summary_df['filename'], 
                key=f"module_selector_{i}",
                index=int(best_idx)
            )
            if module_selector:
                module_df = pd.read_parquet(os.path.join(node_dir, module_selector), engine='pyarrow')
                if node == "retrieval" or node == "passage_reranker":
                    module_df = pd.concat([merged_df, split_retrieval_df(module_df)], axis=1)
                elif node == "generator":
                    module_df = pd.concat([merged_df, del_generated_token_df(module_df)], axis=1)
                st.dataframe(module_df)

    # YAML 파일 탭
    with tabs[-1]:
        yaml_file_markdown = yaml_to_markdown(os.path.join(trial_dir, "config.yaml"))
        st.markdown(yaml_file_markdown)
