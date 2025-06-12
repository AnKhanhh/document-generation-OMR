import pandas as pd
import streamlit as st
from typing import List, Dict, Tuple, Optional
from pathlib import Path


def load_dataframe(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Load uploaded file to DataFrame. Returns (df, error_message)."""
    if not uploaded_file:
        return None, "No file uploaded"

    try:
        extension = Path(uploaded_file.name).suffix.lower()
        if extension == '.csv':
            df = pd.read_csv(uploaded_file)
        elif extension in ['.xlsx', '.xls']:
            df = pd.read_excel(uploaded_file)
        else:
            return None, f"Unsupported file type: {extension}. Please use CSV or Excel files."

        return df, None
    except Exception as e:
        return None, f"Failed to read file: {str(e)}"


def dataframe_to_answer_keys(df: pd.DataFrame) -> Tuple[Optional[List[Dict]], Optional[str]]:
    """Convert DataFrame to answer keys. Returns (answer_keys, error_message)."""
    # Validate columns
    error_msg = _validate_columns(df)
    if error_msg:
        return None, error_msg
    # validate rows
    answer_keys = []
    for _, row in df.iterrows():
        parsed_row, parse_error = _parse_row(row)
        if parse_error:
            return None, parse_error

        validation_error = _validate_row(parsed_row, row.name + 1)
        if validation_error:
            return None, validation_error

        answer_keys.append(parsed_row)

    return answer_keys, None


def _validate_columns(df: pd.DataFrame) -> Optional[str]:
    """Validate required columns exist."""
    required = {'question', 'answer', 'score'}
    missing = required - set(df.columns)
    if missing:
        return f"Missing required columns: {missing}. Expected columns: question, answer, score"
    return None


def _parse_row(row: pd.Series) -> Tuple[Optional[Dict], Optional[str]]:
    """Parse a single row to answer key format."""
    try:
        answer_str = str(row['answer']).strip()
        answers = [a.strip().upper() for a in answer_str.split(',') if a.strip()]

        # Handle empty score values by defaulting to 1.0
        score_val = row['score']
        score = 1.0 if pd.isna(score_val) or str(score_val).strip() == '' else float(score_val)

        return {
            'question': int(row['question']),
            'answer': answers,
            'score': score
        }, None
    except Exception as e:
        return None, f"Row {row.name + 1}: Failed to parse data - {str(e)}"


def _validate_row(parsed_row: Dict, row_num: int) -> Optional[str]:
    """Validate individual row data."""
    q_num, answers, score = parsed_row.values()

    if q_num <= 0:
        return f"Row {row_num}: Invalid question number {q_num}. Must be positive."

    # Validate answers are single letters
    invalid_answers = [a for a in answers if not (len(a) == 1 and a.isalpha())]
    if invalid_answers:
        return f"Row {row_num}: Invalid answer choices {invalid_answers}."

    if score < 0:
        return f"Row {row_num}: Invalid score {score}."

    if len(answers) != len(set(answers)):
        return f"Row {row_num}: Duplicate answers {answers}."

    return None
