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


def find_optimal_grouping(num_questions: int, choices_per_question: int, validator) -> tuple[int, str, int]:
    """
    Find optimal grouping for given questions and choices.
    Returns (grouping, reason_message, max_possible_questions)
    """
    best_grouping = None
    reason = ""

    # Track the grouping that allows maximum questions (even if less than requested)
    max_capacity_grouping = None
    max_capacity = 0

    # Phase 1: Try ideal range (4-6)
    for g in range(4, 7):
        if g > num_questions:
            continue
        valid, max_q = validator.validate_questions_num(num_questions, choices_per_question, g)

        # Track maximum capacity
        if max_q > max_capacity:
            max_capacity = max_q
            max_capacity_grouping = g

        if valid:
            # Check aesthetic: does it create complete rows in 3-column layout?
            groups_per_row = validator.max_group_on_row(choices_per_question)
            total_rows = (num_questions + g - 1) // g  # Ceiling division
            is_aesthetic = total_rows % groups_per_row == 0

            if is_aesthetic:
                print(f"found optimal grouping={g} for questions={num_questions} and choices={choices_per_question}")
                return g, f"All input valid, ready to generate answer sheet", num_questions
            if best_grouping is None:
                best_grouping = g
                reason = "All input valid, ready to generate answer sheet"

    # Phase 2: Expand search if nothing in ideal range works
    if best_grouping is None:
        _, max_val_g = validator.validate_questions_per_group(4)
        max_val_g = min(max_val_g, num_questions)
        for distance in range(1, max_val_g):
            for g in [4 - distance, 6 + distance]:
                if g < 1 or g > max_val_g:
                    continue
                valid, max_q = validator.validate_questions_num(num_questions, choices_per_question, g)

                # Track maximum capacity
                if max_q > max_capacity:
                    max_capacity = max_q
                    max_capacity_grouping = g

                if valid:
                    print(f"found grouping={g} for questions={num_questions} and choices={choices_per_question}")
                    return g, "All input valid, ready to generate answer sheet", num_questions

    # Return best from ideal range if found
    if best_grouping:
        return best_grouping, reason, num_questions

    # No valid grouping found - return the one with highest capacity
    return None, f"maximum total {max_capacity} questions allowed", max_capacity


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
    # Step 1: Upload answer key first
    st.header("1. Upload answer keys file to generate document")
    st.caption("Answer key file must meet the specified format. Hover on question mark for details.")
    from utility.read_keys import load_dataframe, dataframe_to_answer_keys

    key_file = st.file_uploader("Answer Keys", type=['csv', 'xlsx', 'xls'],
                                help="A valid answer key file must have 3 following columns: | question | answer | score |."
                                     " Cells under 'score' can be left empty, in which case default_score=1."
                                     " Answers must be comma-separated letters.")

    k_valid = False
    val_question = 0

    if key_file:
        df, load_error = load_dataframe(key_file)
        if load_error:
            st.error(load_error)
        else:
            answer_keys, process_error = dataframe_to_answer_keys(df)
            if process_error:
                st.error(process_error)
            else:
                val_question = len(answer_keys)
                if not all(d['question'] <= val_question for d in answer_keys):
                    st.error(f"Invalid entry: out-of-bound question number")
                else:
                    st.session_state["answer_keys"] = answer_keys
                    st.success(f"✓ Loaded {val_question} questions from answer key file")
                    k_valid = True

    # Step 2: Define answer sheet parameters
    st.subheader("Specify the number of choices for each question")
    val_choice = st.number_input("Choices per question", min_value=2, max_value=26, value=4,
                                 help="Number of choices per question",
                                 disabled=not k_valid)

    c_valid, c_limit = st.session_state['validator'].validate_choices_per_question(val_choice)
    if not c_valid:
        st.error(f"Invalid: question can hold {c_limit} choices maximum")

    # Auto-calculate optimal grouping if we have valid inputs
    val_group = None
    q_valid = False

    if k_valid and c_valid and val_question > 0:
        # Find optimal grouping
        val_group, reason, max_possible = find_optimal_grouping(val_question, val_choice, st.session_state['validator'])

        if val_group:
            # If we found a valid grouping, questions definitely fit
            q_valid = True
            st.success(f"✓ {reason}")
        else:
            # No valid grouping found - show maximum possible
            st.error(f"Cannot fit {val_question} questions with {val_choice} choices. {reason}")
    else:
        if k_valid and not c_valid:
            st.warning("Please select valid value for number of choices per question")

    # ==========================================================================================
    # Generation output
    no_generate = not (k_valid and c_valid and q_valid and val_group is not None)
    if st.button("Generate", disabled=no_generate):
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
            key_model.set_answers(st.session_state["answer_keys"])
            db_metrics_log = DatabaseBridge.create_complete_sheet(template_sheet.static_metrics,
                                                                  key_model,
                                                                  template_sheet.dynamic_metrics)
            # Serve generated sheet
            st.session_state['sheet_pdf'] = buffer_gen_pdf.getvalue()
            st.session_state['template_img'] = pdf_stream_2_img(buffer_gen_pdf.getvalue())
            print(f"Document metadata saved with uuid {db_metrics_log['instance_id']}")
            st.success("Answer sheet ready for download!")

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
            # del st.session_state["answer_keys"]
            del st.session_state['sheet_pdf']
            del st.session_state['template_img']
            st.rerun()

    # ==========================================================================================
    # Synth data test
    st.divider()
    st.header("2. Or run an integration test")
    st.text("The integration test will:")
    st.text("- Generate random answer keys")
    st.text("- Generate random student sheets")
    st.text("- Extract data from student sheet, grade and export result")

    # Integration test needs its own inputs now
    col_test_q, col_test_c = st.columns(2)
    with col_test_q:
        test_val_question = st.number_input("Test questions", min_value=1, value=20,
                                            help="Number of questions for test")
    with col_test_c:
        test_val_choice = st.number_input("Test choices", min_value=2, max_value=26, value=4,
                                          help="Choices per question for test")

    # Validate test inputs and find grouping
    test_c_valid, _ = st.session_state['validator'].validate_choices_per_question(test_val_choice)
    test_val_group = None
    test_q_valid = False

    if test_c_valid:
        test_val_group, test_reason, test_max = find_optimal_grouping(test_val_question, test_val_choice, st.session_state['validator'])
        if test_val_group:
            test_q_valid = True
            # st.info(f"Test will use groups of {test_val_group} ({test_reason})")
        else:
            st.error(f"Invalid test parameters: {test_reason}")
    config_distortion = st.checkbox("Configure parameters for synthetic document distortion", value=False)
    params = render_config(config_distortion)

    if st.button("Run test on program pipeline", disabled=not (test_q_valid and test_val_group)):
        # Generate synthetic input - template
        with st.spinner("Generating synthetic input..."):
            buffer_gen_pdf = io.BytesIO()
            template_sheet = AnswerSheetGenerator()
            shared_sheet_id = st.session_state['id_gen'].generate()
            template_sheet.generate_answer_sheet(
                num_questions=test_val_question, choices_per_question=test_val_choice, questions_per_group=test_val_group,
                buffer=buffer_gen_pdf,
                sheet_id=shared_sheet_id
            )
            key_model = AnswerKeys()
            key_model.set_answers(generate_answer_keys(num_questions=test_val_question, choices_per_question=test_val_choice))
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
                num_questions=test_val_question, choices_per_question=test_val_choice, questions_per_group=test_val_group,
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

        if st.button('Clear', key='clear_synth'):
            del st.session_state['distorted_filled']
            del st.session_state['template_pdf']
            del st.session_state['summary_synth']
            st.rerun()

# Extractor tab
with tab2:
    # Upload images
    st.header("1. First, upload the template and student answer sheet")
    template = st.file_uploader("Template Image", type=['png', 'jpg', 'jpeg', 'pdf'])
    if template:
        gray_template = file_to_grayscale(template)
        # st.image(gray_template, caption="Grayscale Template")
    student_sheet = st.file_uploader("Student Sheet", type=['png', 'jpg', 'jpeg'])
    if student_sheet:
        gray_photo = file_to_grayscale(student_sheet)

    # Config grading
    st.header("2. Then, customize the grading behavior")
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
        if st.button("Clear", key='clear_extract'):
            del st.session_state["summary_pdf"]
            st.rerun()
