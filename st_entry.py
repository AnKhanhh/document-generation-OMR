from pathlib import Path

import pandas as pd
import streamlit as st


def load_file(file_path):
    try:
        with open(file_path, "rb") as f:
            return f.read()
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return None


# Main Streamlit app
st.title("Answer Sheet Generator & Extractor")
if st.button("Refresh session"):
    st.rerun()

tab1, tab2 = st.tabs(["Generator", "Extractor"])

# Generator tab
with tab1:
    # Step 1: input choices and groupings
    st.header("1. First, define answer sheet parameters")
    col2, col3 = st.columns(2)

    with col2:
        b = st.number_input("Choices", min_value=2, max_value=26, value=4,
                            help="Number of choices per question")

    with col3:
        c = st.number_input("Groupings", min_value=1, value=5,
                            help="Number of questions per group")

    from document_generation.layout_1_limits import validate_a, validate_b, validate_c

    b_valid = validate_b(b)
    c_valid = validate_c(c)
    col2.write("✓" if b_valid else "✗ Must be < 100")
    col3.write("✓" if c_valid else "✗ Must be < 100")
    b_c_valid = b_valid and c_valid

    # Step 2: input number of questions
    a = st.number_input(
        "Questions",
        min_value=1,
        disabled=not b_c_valid,
        help="Total number of questions"
    )

    a_valid = b_c_valid and validate_a(a, b, c)
    st.write(f"✓" if a_valid else f"✗ Must be ≤ {b * c}")
    if a_valid:
        st.success(f"All input parameters validated")
    else:
        st.info(f"Page dimension not enough for {a} questions!")

    # Step 3: input answer key
    st.divider()
    st.header("2. Then, either upload answer key")
    st.caption("Answer key file must have 3 columns: | question | answer | score |")
    from utility.read_keys import load_dataframe, dataframe_to_answer_keys

    key_file = st.file_uploader("Answer Keys", type=['csv', 'xlsx', 'xls'],
                                disabled=not a_valid)
    if key_file:
        df, load_error = load_dataframe(key_file)
        if load_error:
            st.error(load_error)
        else:
            answer_keys, process_error = dataframe_to_answer_keys(df)
            if process_error:
                st.error(process_error)
                st.session_state.answer_keys = None
            elif len(answer_keys) != a:
                st.error(f"There are {len(answer_keys)} answer keys, but answer sheet have {a} questions!")
                st.session_state.answer_keys = None
            else:
                st.session_state.answer_keys = answer_keys
                st.success(f"Loaded {len(answer_keys)} rows of answer keys.")
                # st.json(answer_keys[:1])

    # Generation output
    if st.button("Generate"):
        if st.session_state.get('answer_keys') is None:
            st.warning("Please upload a valid answer key file")
        else:
            st.session_state['template_img'] = load_file("out/image/pristine_1749185425000.png")
            st.session_state['sheet_pdf'] = load_file("out/pdf/pristine_1749185425000.pdf")
            st.success("Answer sheet generated")

    # Show download buttons if files exist in session state
    if 'sheet_pdf' in st.session_state and 'template_img' in st.session_state:
        col_pdf, col_img = st.columns(2)
        with col_pdf:
            st.download_button("Download document 📄",
                               data=st.session_state['sheet_pdf'],
                               file_name="answer_sheet.pdf")
        with col_img:
            st.download_button("Download template image 📸",
                               data=st.session_state['template_img'],
                               file_name="template.png")
        if st.button("Clear"):
            del st.session_state['sheet_pdf']
            del st.session_state['template_img']
            st.rerun()

    st.divider()
    st.header("3. Or run an integration test")
    st.text("The integration test will:")
    st.text("- Generate random answer keys")
    st.text("- Generate random student sheets")
    st.text("- Extract data from student sheet, grade and export result")
    if st.checkbox("Customize the distortion module behavior:"):
        pass
    if st.button("Run test on program pipeline"):
        pass

# Extractor tab
with tab2:
    st.header("1. First, upload the template and student answer sheet:")
    template = st.file_uploader("Template Image", type=['png', 'jpg', 'jpeg'])
    student_sheet = st.file_uploader("Student Sheet", type=['png', 'jpg', 'jpeg'])
    st.header("2. Then, customize the grading behavior:")
    partial = st.checkbox("Give point to partially correct questions")
    if st.checkbox("Deduct point from wrong questions"):
        if st.checkbox("by a set amount"):
            pass
        st.caption("or")
        if st.checkbox("based on the question's score"):
            pass
    if st.button("Grade answer sheet"):
        summary_pdf = load_file("out/pdf/summary.pdf")
        st.success("Grading complete, result ready for download!")
        st.download_button("Download report 📄", data=summary_pdf, file_name="summary.pdf")
