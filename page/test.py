import streamlit as st

@st.dialog("Ask your question")
def ask_question():
    if "question_history" not in st.session_state:
        st.session_state.question_history = []

    question = st.text_input("What's your question?")
    if st.button("Submit"):
        st.session_state.question_history.append(question)
        st.stop()

    if st.session_state.question_history:
        st.write("Question History:")
        for q in st.session_state.question_history:
            st.write(f"- {q}")

if "question_history" not in st.session_state:
    st.write("Do you have a question?")
    if st.button("Ask a question"):
        ask_question()
else:
    st.write("Question History:")
    for q in st.session_state.question_history:
        st.write(f"- {q}")
    if st.button("Ask another question"):
        ask_question()