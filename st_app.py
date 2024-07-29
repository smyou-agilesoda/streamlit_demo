import os
from io import BytesIO

import pandas as pd
import streamlit as st


def convert_str(x):
    x = "".join(["score: ", str(round(x, 5)), "\n\n"])
    return x

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")

@st.cache_data
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Sheet1')
    return output.getvalue()


base_path = os.path.abspath(".")
trial_path = os.path.join(base_path, "benchmark")
trial_list = os.listdir(trial_path)
trial_list = [x for x in trial_list if os.path.isdir(os.path.join(trial_path, x))]

_trial = st.sidebar.radio('선택하세요', trial_list)
trial = os.path.join(trial_path, _trial)
result = os.path.join(trial, "0")
data = os.path.join(trial, "data")
corpus_df = pd.read_parquet(os.path.join(data, "corpus.parquet"), engine='pyarrow')
qa_df = pd.read_parquet(os.path.join(data, "qa.parquet"), engine='pyarrow')
qa_df["retrieval_gt"] = qa_df.retrieval_gt.apply(lambda x: x[0][0])
qa_df["generation_gt"] = qa_df.generation_gt.apply(lambda x: x[0])

total = pd.read_csv(os.path.join(result, "summary.csv"))
st.write("### total:")
st.dataframe(total)

retrieve_node_line_path = os.path.join(result, "retrieve_node_line")
retrieval_total = pd.read_csv(os.path.join(retrieve_node_line_path, "summary.csv"))
st.write("### retrieval_total:")
st.dataframe(retrieval_total)

retrieval_path = os.path.join(retrieve_node_line_path, "retrieval")
retrieval_only = pd.read_csv(os.path.join(retrieval_path, "summary.csv"))
is_best_col = retrieval_only.pop("is_best")
retrieval_only.insert(0,"is_best", is_best_col)

# df.insert(0, "check", False)
st.write("### retrieval_only:")

selection = st.dataframe(
    retrieval_only, 
    hide_index=True,
    on_select="rerun", 
    selection_mode="single-row"
)
select_rows = selection["selection"]["rows"]
if select_rows:
    select_idx = select_rows[0]
    select_parquet = retrieval_only.loc[select_idx,"filename"]
    selected_df = pd.read_parquet(os.path.join(retrieval_path, select_parquet), engine='pyarrow')
    selected_df.pop("retrieved_ids")
    contents = selected_df.pop("retrieved_contents")
    scores = selected_df.pop("retrieve_scores")
    scores = scores.apply(lambda lst: list(map(convert_str, lst)))
    contents = scores + contents
    selected_df = pd.concat([pd.DataFrame(contents.tolist()), selected_df], axis=1)
    merged_df = pd.merge(qa_df, corpus_df, 
                         left_on='retrieval_gt', 
                         right_on='doc_id', 
                         how='inner')
    selected_df = pd.concat([merged_df[["contents", "query", "generation_gt"]], selected_df], axis=1)
    st.header(select_parquet)
    st.download_button(label="Download", data=to_excel(selected_df), file_name=f"{_trial}_temp.xlsx")
    st.dataframe(
        selected_df,
        column_config={
            "0": st.column_config.TextColumn(
                "0",
                # width="medium",
                help="긴 텍스트 열입니다.",
                max_chars=50
            )
        },
    )
    # st.dataframe(
    #     selected_df.style.set_properties(**{'white-space': 'pre-wrap'})
    # )



# # 페이지 제목 설정
# st.title('데이터프레임 표시 예시')