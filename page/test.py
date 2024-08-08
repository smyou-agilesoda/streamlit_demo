
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
        st.session_state["messages"] = [
            ChatMessage(role="assistant", content="답변자는 노란색 로봇으로 표시됩니다")
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg.role):
            st.markdown(msg.content)
        # st.chat_message(msg.role).write(msg.content)

    if "chat_init" not in st.session_state: st.session_state["chat_init"] = True
    question = st.empty() if st.session_state["chat_init"] else st.chat_message("user")
    answer = st.empty() if st.session_state["chat_init"] else st.chat_message("assistant")
    st.session_state["chat_init"] = False
    
    if user_input := st.chat_input():
        st.session_state.messages.append(ChatMessage(role="user", content=user_input))
        with question:
            st.markdown(user_input)
        # with answer:
        #     response = user_input
        #     st.markdown(user_input)
        #     st.session_state.messages.append(
        #         ChatMessage(role="assistant", content=response)
        #     )
        with answer:
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


if "question_history" not in st.session_state:
    st.write("Do you have a question?")
    if st.button("Ask a question"):
        ask_question()