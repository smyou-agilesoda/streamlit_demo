import os
from io import BytesIO

import pandas as pd
import streamlit as st
from st_pages import add_page_title, get_nav_from_toml

st.set_page_config(page_title="Total Dashboard", layout="wide")

nav = get_nav_from_toml()

pg = st.navigation(nav)

add_page_title(pg)

pg.run()