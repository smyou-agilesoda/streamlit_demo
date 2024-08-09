
from langchain.schema import ChatMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI
import streamlit as st

# TODO)
# 추천질문 집어넣기
# 파싱 데이터 
# 요약문
# pdf뷰어

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

cfg = RunnableConfig()
cfg["configurable"] = {"session_id": "any"}

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "한글로 간결하게 답변하세요."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

@st.dialog("Ask your question", width="large")
def ask_question():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # 채팅 기록 히스토리
    for msg in st.session_state.messages:
        with st.chat_message(msg.role):
            st.markdown(msg.content)
        # st.chat_message(msg.role).write(msg.content)

    # 질문 입력 전까지 placeholder 숨기기
    if "chat_before" not in st.session_state: 
        st.session_state["chat_before"] = True
    if st.session_state["chat_before"]:
        st.chat_message("user").write("질문자는 빨간색 사람으로 표시됩니다.")
        st.chat_message("assistant").write("답변자는 노란색 로봇으로 표시됩니다.")
        question_placeholder = st.empty()
        answer_placeholder = st.empty()
    else:
        question_placeholder = st.chat_message("user")
        answer_placeholder = st.chat_message("assistant")
    # 질문 입력되면 숨기기 해제
    st.session_state["chat_before"] = False
    
    if user_input := st.chat_input():
        st.session_state.messages.append(ChatMessage(role="user", content=user_input))
        with question_placeholder:
            st.markdown(user_input)
        # with answer_placeholder:
        #     response = user_input
        #     st.markdown(user_input)
        #     st.session_state.messages.append(
        #         ChatMessage(role="assistant", content=response+response)
        #     )
        with answer_placeholder:
            # stream_handler = StreamHandler(st.empty())
            llm = ChatOpenAI()#streaming=True)#, callbacks=[stream_handler])
            chain = prompt | llm
            chain_with_history = RunnableWithMessageHistory(
                chain,
                lambda session_id: msgs,
                input_messages_key="question",
                history_messages_key="history",
            )
            response = chain_with_history.stream({"question": user_input}, cfg)
            response = st.write_stream(response)
            st.session_state.messages.append(
                ChatMessage(role="assistant", content=response)
            )


left_col, right_col = st.columns([1, 1])
with left_col:
    st.subheader("pdf 요약")
    file_uploader_placeholder = st.empty()
    uploaded_file = file_uploader_placeholder.file_uploader("파일 업로드", type=["pdf"])
            
    if uploaded_file is not None:
        # 업로드 파일 임시 저장
        file_path = f"./tmp/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.write("업로드된 파일:", uploaded_file.name)
        
        # parsing
        
        
        st.write("요약 blahblah")
        # 파일 업로드 UI 박스 제거
        file_uploader_placeholder.empty()
        
    if uploaded_file:
        if st.button("Ask a question"):
            ask_question()

with right_col:
    st.subheader("pdf 미리보기")
    # 파일 업로드 전 버튼 비활성화? 숨기기?
    if st.button("modal test"):
        ask_question()