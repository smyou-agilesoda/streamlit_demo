import streamlit as st
from st_pages import add_page_title, get_nav_from_toml
from dotenv import load_dotenv

st.set_page_config(page_title="AutoRAG Dashboard", layout="wide")
load_dotenv()

nav = get_nav_from_toml()
pg = st.navigation(nav)
add_page_title(pg)

pg.run()