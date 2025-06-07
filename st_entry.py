from pathlib import Path

import pandas as pd
import streamlit as st


# Answer keys reader
from utility.read_keys import load_dataframe, dataframe_to_answer_keys
uploaded_file = st.file_uploader("Upload Answer Keys", type=['csv', 'xlsx', 'xls'])

# Store parsed data in session state
if uploaded_file:
    df, load_error = load_dataframe(uploaded_file)
    if load_error:
        st.error(load_error)
    else:
        answer_keys, process_error = dataframe_to_answer_keys(df)
        if process_error:
            st.error(process_error)
            st.session_state.answer_keys = None
        else:
            st.session_state.answer_keys = answer_keys
            st.success(f"Loaded {len(answer_keys)} answer keys")
            st.json(answer_keys[:1])  # Preview top row

# Later operations
if st.button("Print Keys"):
    if st.session_state.get('answer_keys') is None:
        st.warning("Please upload a valid answer key file")
    else:
        st.write(st.session_state.answer_keys)


st.title("Answer Sheet Generator & Extractor")

# Generator section
st.header("1. Generate Answer Sheet")
col1, col2, col3 = st.columns(3)
a = col1.number_input("Questions", min_value=1)
b = col2.number_input("Choices", min_value=2)
c = col3.number_input("Groupings", min_value=1)
keys_file = st.file_uploader("Answer Keys", type=['xlsx, xls, csv'])

if st.button("Generate"):
    # Your generation logic
    st.download_button("Download Answer Sheet", data=sheet_data)

# Extractor section
st.header("2. Extract Results")
template = st.file_uploader("Template Image", type=['png', 'jpg'])
student_sheet = st.file_uploader("Student Sheet", type=['png', 'jpg'])

if st.button("Extract"):
    # Your extraction logic
    st.download_button("Download Results", data=results)
