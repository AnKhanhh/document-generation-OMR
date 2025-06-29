import io

import cv2
import streamlit as st

from DB_bridging.database_bridge import DatabaseBridge
from DB_bridging.models import AnswerKeys
from document_extraction.extraction_entry import extract
from document_generation.layout_1_limits import AnswerSheetLayoutValidator, LayoutConstraints
from document_generation.layout_one import AnswerSheetGenerator
from utility.grading import GradingConfig
from utility.id_gen import IDGenerator
from utility.misc import generate_answer_keys
from utility.read_template import file_to_grayscale
from utility.ui_utils import pdf_stream_2_img, process_distortion, pdf_stream_2_cv2_gray


def render_config(show: bool) -> dict:
    """Render config controls or return defaults"""
    params = {}
    for key, (label, default, min_val, max_val, step) in DISTORTION_CONFIG.items():
        if show:
            params[key] = st.slider(label, min_val, max_val, default, step)
        else:
            params[key] = default
    return params


DISTORTION_CONFIG = {
    'severity': ('Perspective Severity', 0.6, 0.0, 1.0, 0.01),
    'angle': ('Rotation Angle', 50, 0, 360, 1),
    'max_shadow': ('Shadow Intensity', 0.5, 0.0, 1.0, 0.01),
    'amount': ('Noise Amount', 0.1, 0.0, 2.0, 0.01)  # 2.0 allows reasonable noise range
}

# Initialize global persistent variables
if 'validator' not in st.session_state:
    st.session_state['validator'] = AnswerSheetLayoutValidator(LayoutConstraints())
if 'id_gen' not in st.session_state:
    st.session_state['id_gen'] = IDGenerator()
if 'db_init_result' not in st.session_state:
    st.session_state['db_init_result'] = DatabaseBridge.initialize()

# Main Streamlit app
st.title("Answer Sheet Generator & Extractor")
if st.button("Refresh session"):
    st.rerun()

tab1, tab2 = st.tabs(["Generator", "Extractor"])

# Generator tab
with tab1:
    # ==========================================================================================
    # Step 1: input choices and groupings
    q_valid, c_valid, g_valid = False, False, False
    st.header("1. First, define answer sheet parameters")
    col_choice, col_group = st.columns(2)

    with col_choice:
        val_choice = st.number_input("Choices", min_value=2, max_value=26, value=4,
                                     help="Number of choices per question")

    with col_group:
        val_group = st.number_input("Groupings", min_value=1, value=5,
                                    help="Number of questions per group")

    c_valid, c_limit = st.session_state['validator'].validate_choices_per_question(val_choice)
    g_valid, g_limit = st.session_state['validator'].validate_questions_per_group(val_group)
    col_choice.write("✓" if c_valid else f"Invalid: question can hold {c_limit} choices maximum")
    col_group.write("✓" if g_valid else f"Invalid: group can hold {g_limit} questions maximum")
    b_c_valid = c_valid and g_valid

    # Step 2: input number of questions
    val_question = st.number_input(
        "Questions",
        min_value=1,
        disabled=not b_c_valid,
        help="Total number of questions"
    )

    if b_c_valid:
        q_valid, a_limit = st.session_state['validator'].validate_questions_num(val_question, val_choice, val_group)
        st.write(f"✓" if q_valid else f"Invalid: page dimensions allow {a_limit} questions maximum")
        if q_valid:
            if val_group > val_question:
                st.warning(f"Group size exceed number of questions, reduced to {val_question}")
                val_group = val_question
            st.success(f"✓ All input parameters validated")

    # Step 3: input answer key
    st.divider()
    st.header("2. Then, either upload answer key")
    st.caption("Answer key file must meet the specified format. Click on question mark button for detail.")
    from utility.read_keys import load_dataframe, dataframe_to_answer_keys

    key_file = st.file_uploader("Answer Keys", type=['csv', 'xlsx', 'xls'],
                                disabled=not q_valid,
                                help="A valid answer key file must have 3 following columns: | question | answer | score |."
                                     " Cells under 'score' can be left empty, in which case default_score=1."
                                     " Answers must be letters, and comma-separated.")
    if key_file:
        df, load_error = load_dataframe(key_file)
        if load_error:
            st.error(load_error)
        else:
            answer_keys, process_error = dataframe_to_answer_keys(df)
            if process_error:
                st.error(process_error)
            elif len(answer_keys) != val_question:
                st.error(f"Length mismatch: {len(answer_keys)} answer key entries, expected {val_question}!")
            elif not all(d['question'] <= val_question for d in answer_keys):
                st.error(f"Invalid entry: out-of-bound question number")
            else:
                st.session_state["answer_keys"] = answer_keys
                st.success(f"Loaded {len(answer_keys)} key entries.")
                # st.json(answer_keys[:1])

    # ==========================================================================================
    # Generation output
    if st.button("Generate"):
        if st.session_state.get('answer_keys') is None:
            st.warning("Please upload a valid answer key file")
        else:
            # Generate sheet
            buffer_gen_pdf = io.BytesIO()
            template_sheet = AnswerSheetGenerator()
            template_sheet.generate_answer_sheet(
                num_questions=val_question, choices_per_question=val_choice, questions_per_group=val_group,
                buffer=buffer_gen_pdf,
                sheet_id=st.session_state['id_gen'].generate()
            )
            # Save data into DB
            key_model = AnswerKeys()
            key_model.set_answers(answer_keys)
            db_metrics_log = DatabaseBridge.create_complete_sheet(template_sheet.static_metrics,
                                                                  key_model,
                                                                  template_sheet.dynamic_metrics)
            # Serve generated sheet
            st.session_state['sheet_pdf'] = buffer_gen_pdf.getvalue()
            st.session_state['template_img'] = pdf_stream_2_img(buffer_gen_pdf.getvalue())
            st.success(f"Answer sheet ready for download! Layout metrics saved to BD with uuid {db_metrics_log['instance_id']}")

    # Show download buttons if files exist in session state
    if 'sheet_pdf' in st.session_state and 'template_img' in st.session_state:
        col_pdf, col_img = st.columns(2)
        with col_pdf:
            st.download_button("Download document 📄",
                               data=st.session_state['sheet_pdf'],
                               file_name="answer_sheet.pdf",
                               mime="application/pdf")
        with col_img:
            st.download_button("Download template image 📸",
                               data=st.session_state['template_img'],
                               file_name="template.png")
        if st.button("Clear"):
            del st.session_state["answer_keys"]
            del st.session_state['sheet_pdf']
            del st.session_state['template_img']
            st.rerun()

    # ==========================================================================================
    # Synth data test
    # Configuration mapping: param_name -> (label, default, min, max, step)

    st.divider()
    st.header("3. Or run an integration test")
    st.text("The integration test will:")
    st.text("- Generate random answer keys")
    st.text("- Generate random student sheets")
    st.text("- Extract data from student sheet, grade and export result")
    config_distortion = st.checkbox("Configure parameters for synthetic document distortion", value=False)
    params = render_config(config_distortion)
    if st.button("Run test on program pipeline"):
        # Generate synthetic input - template
        with st.spinner("Generating synthetic input..."):
            buffer_gen_pdf = io.BytesIO()
            template_sheet = AnswerSheetGenerator()
            shared_sheet_id = st.session_state['id_gen'].generate()
            template_sheet.generate_answer_sheet(
                num_questions=val_question, choices_per_question=val_choice, questions_per_group=val_group,
                buffer=buffer_gen_pdf,
                sheet_id=shared_sheet_id
            )
            key_model = AnswerKeys()
            key_model.set_answers(generate_answer_keys(num_questions=val_question, choices_per_question=val_choice))
            db_metrics_log = DatabaseBridge.create_complete_sheet(template_sheet.static_metrics,
                                                                  key_model,
                                                                  template_sheet.dynamic_metrics)
            print(f"Layout metrics saved to BD with uuid {db_metrics_log['instance_id']}")
            st.session_state['template_pdf'] = buffer_gen_pdf.getvalue()
            template_img = pdf_stream_2_cv2_gray(buffer_gen_pdf.getvalue())
            # Generate synthetic input - student sheet
            buffer_gen_synth = io.BytesIO()
            synthetic_sheet = AnswerSheetGenerator(fill_in=True)
            synthetic_sheet.generate_answer_sheet(
                num_questions=val_question, choices_per_question=val_choice, questions_per_group=val_group,
                buffer=buffer_gen_synth,
                sheet_id=shared_sheet_id
            )
            distorted_filled = process_distortion(buffer_gen_synth.getvalue(), **params)
            st.session_state['distorted_filled'] = cv2.cvtColor(distorted_filled, cv2.COLOR_BGR2GRAY)
        # Extract and grade on the spot
        with st.spinner("Performing extraction on synthesized input..."):
            buffer_sum_pdf = io.BytesIO()
            corrected_photo, viz = extract(st.session_state['distorted_filled'], template_img,
                                           visualize=False,
                                           summary_buffer=buffer_sum_pdf)
            st.session_state['summary_synth'] = buffer_sum_pdf.getvalue()
        st.success("Successfully generated and extracted synthetic data")

    if ('distorted_filled' in st.session_state) and ('template_pdf' in st.session_state) and ('summary_synth' in st.session_state):
        st.image(st.session_state['distorted_filled'], caption="Synthetic input answer sheet")
        col_pdf_synth, col_img_synth = st.columns(2)
        with col_pdf_synth:
            st.download_button("Template answer sheet 📄", data=st.session_state['template_pdf'], file_name="template.pdf")
        with col_img_synth:
            st.download_button("Extraction result 📄", data=st.session_state['summary_synth'], file_name="summary.pdf")

        if st.button('Clear'):
            del st.session_state['distorted_filled']
            del st.session_state['template_pdf']
            del st.session_state['summary_synth']
            st.rerun()

# Extractor tab
with tab2:
    # Upload images
    st.header("1. First, upload the template and student answer sheet:")
    template = st.file_uploader("Template Image", type=['png', 'jpg', 'jpeg', 'pdf'])
    if template:
        gray_template = file_to_grayscale(template)
        # st.image(gray_template, caption="Grayscale Template")
    student_sheet = st.file_uploader("Student Sheet", type=['png', 'jpg', 'jpeg'])
    if student_sheet:
        gray_photo = file_to_grayscale(student_sheet)

    # Config grading
    st.header("2. Then, customize the grading behavior:")
    partial = st.checkbox("Give points to partially correct answers")
    deduct = st.checkbox("Deduct points for wrong answers")
    if deduct:
        option = st.radio("choose how points are deducted for each wrong answer:", ["Static", "Proportional"], index=0)

        col_static_lbl, col_static_input = st.columns([3, 1])
        with col_static_lbl:
            st.write("By set amount of points")
        with col_static_input:
            val_static = st.number_input(label="lorem ipsum", value=1, min_value=1,
                                         disabled=option != "Static",
                                         label_visibility="collapsed")

        col_dynamic_lbl, col_dynamic_input = st.columns([3, 1])
        with col_dynamic_lbl:
            st.write("Proportional to the question's score")
        with col_dynamic_input:
            val_dynamic = st.number_input(label="lorem ipsum", min_value=0.01, max_value=1.0, value=0.5, step=0.01,
                                          disabled=option != "Proportional",
                                          label_visibility="collapsed")

    # Init extraction
    if st.button("Grade answer sheet"):
        grading_config = GradingConfig(allow_partial_credit=partial)

        if deduct:
            grading_config.enable_deduction = True
            if option == "Proportional":
                grading_config.deduction_is_relative = True
                grading_config.deduction_amount = val_dynamic
            else:
                grading_config.deduction_is_relative = False
                grading_config.deduction_amount = val_static

        buffer_sum_pdf = io.BytesIO()
        with st.spinner("Extracting... Please wait"):
            corrected_photo, viz = extract(gray_photo, gray_template,
                                           visualize=True,
                                           summary_buffer=buffer_sum_pdf,
                                           grading_config=grading_config)
            if viz is not None:
                for k, v in viz.items():
                    cv2.imwrite(f"out/vis_detection/{k}.png", v)
            st.session_state['summary_pdf'] = buffer_sum_pdf.getvalue()
            st.success("Grading complete, result ready for download!")

    if 'summary_pdf' in st.session_state:
        st.download_button("Download report 📄", data=st.session_state['summary_pdf'], file_name="summary.pdf")
        if st.button("Clear"):
            del st.session_state["summary_pdf"]
            st.rerun()
