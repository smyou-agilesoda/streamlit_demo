import os

import streamlit as st
from st_pages import add_page_title, get_nav_from_toml

st.set_page_config(page_title="AutoRAG Dashboard", layout="wide")

os.environ["OPENAI_API_KEY"] = st.secrets.OPENAI_API_KEY
nav = get_nav_from_toml()
pg = st.navigation(nav)
add_page_title(pg)

pg.run()