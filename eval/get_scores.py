import ast
import os
import re
import pandas as pd
import numpy as np
from rouge import Rouge
from sklearn.metrics import f1_score as sklearn_f1_score
import ast

from utils import extract_answer_and_evidence


def get_short_EM_score(file_path):
    df = pd.read_csv(file_path)

    pred_column = df['pred'].fillna('').astype(str)
    pred_list = pred_column.tolist()
    long_answers = df['long_answers'].tolist()
    short_answers_list = df['short_answers'].fillna('').astype(str).tolist()
    document_texts = df['document_text'].tolist()

    answers = []
    evidence_num_list = []
    for index, pred in pred_column.items():
        answer, evidence_num = extract_answer_and_evidence(pred)
        if not answer or answer == 'num:':
            if isinstance(pred, str):
                remaining_text_match = re.search(r'Evidence\s*\d+\s*(.*)', pred, re.IGNORECASE)
                if remaining_text_match:
                    answer = remaining_text_match.group(1).strip()
                else:
                    answer = ""

        answers.append(answer)
        evidence_num_list.append(evidence_num)

    em_score_answers_list = []
    for pred, short_answers in zip(pred_list, short_answers_list):
        short_answers_to_list = ast.literal_eval(short_answers)
        if len(short_answers_to_list) >= 2:
            if any(short_answer in pred for short_answer in short_answers_to_list):
                em_score_answers_list.append(1)
            else:
                em_score_answers_list.append(0)
        else:
            if short_answers_to_list[0] in pred:
                em_score_answers_list.append(1)
            else:
                em_score_answers_list.append(0)
    em_score_answers = sum(em_score_answers_list) / 100

    # Check if the evidence is the same
    evidence_positions = [next((i for i, doc in enumerate(ast.literal_eval(document_text)) if long_answer in doc), -1)
                          for document_text, long_answer in zip(document_texts, long_answers)]
    em_score_evidence = sum([1 if (predicted_evidence_num == evidence_position) else 0 for
        predicted_evidence_num, evidence_position in zip(evidence_num_list, evidence_positions)]) / 100

    non_empty_evidence_proportion = sum([0 if num is not None else 1 for num in evidence_num_list]) / len(evidence_num_list)

    return {
        "Evidence EM Score": em_score_evidence,
        "No Evidence Proportion": non_empty_evidence_proportion,
        "Short answers EM Score": em_score_answers
    }

def get_long_EM_score(file_path):
    df = pd.read_csv(file_path)

    pred_column = df['pred'].fillna('').astype(str)
    long_answers = df['long_answers'].tolist()
    long_answers_score_len = 0
    count = 0

    for index, pred in pred_column.items():
        answer, evidence_num = extract_answer_and_evidence(pred)
        if not answer or answer == 'num:':
            if isinstance(pred, str):
                remaining_text_match = re.search(r'Evidence\s*\d+\s*(.*)', pred, re.IGNORECASE)
                if remaining_text_match:
                    answer = remaining_text_match.group(1).strip()
                else:
                    answer = ""

        predicted_answers = answer.split()
        long_answer_splitted = long_answers[index].split()
        long_answer_len = len(long_answer_splitted)
        long_answers_score_len += long_answer_len

        for long_answer_element in long_answer_splitted:
            if long_answer_element in predicted_answers:
                count +=1
    return (count / long_answers_score_len)

def get_em_evidence(file_path):
    df = pd.read_csv(file_path)

    pred_column = df['pred'].fillna('').astype(str)
    pred_list = pred_column.tolist()
    long_answers = df['long_answers'].tolist()
    short_answers_list = df['short_answers'].fillna('').astype(str).tolist()
    document_texts = df['document_text'].tolist()


    answers = []
    evidence_num_list = []
    for index, pred in pred_column.items():
        answer, evidence_num = extract_answer_and_evidence(pred)
        if not answer or answer == 'num:':
            if isinstance(pred, str):
                remaining_text_match = re.search(r'Evidence\s*\d+\s*(.*)', pred, re.IGNORECASE)
                if remaining_text_match:
                    answer = remaining_text_match.group(1).strip()
                else:
                    answer = ""
        # if not evidence_num
        answers.append(answer)
        evidence_num_list.append(evidence_num)

    # short_answer EM score: Check if short_answer is included in the pred sentence
    em_score_answers_list = []
    for pred, short_answers in zip(pred_list, short_answers_list):
        short_answers_to_list = ast.literal_eval(short_answers)
        if len(short_answers_to_list) >= 2:
            if any(short_answer in pred for short_answer in short_answers_to_list):
                em_score_answers_list.append(1)
            else:
                em_score_answers_list.append(0)
        else:
            if short_answers_to_list[0] in pred:
                em_score_answers_list.append(1)
            else:
                em_score_answers_list.append(0)

    # Check if the evidence is the same
    evidence_positions = [next((i for i, doc in enumerate(ast.literal_eval(document_text)) if long_answer in doc), -1)
                          for document_text, long_answer in zip(document_texts, long_answers)]
    em_score_evidence = [1 if (predicted_evidence_num == evidence_position) else None if predicted_evidence_num is None else 0 for predicted_evidence_num, evidence_position in zip(evidence_num_list, evidence_positions)]


    if (len(em_score_evidence) == len(em_score_answers_list)):
        correct_em_zero = 0
        correct_em_plus = 0
        correct_em_minus = 0
        wrong_em_zero = 0
        wrong_em_plus = 0
        wrong_em_minus = 0

        for i in range(len(em_score_evidence)):
            # Correct
            if em_score_answers_list[i] == 1:
                # Correct, EM0
                if em_score_evidence[i] == None:
                    correct_em_zero += 1
                # Correct, EM+
                elif em_score_evidence[i] == 1:
                    correct_em_plus += 1
                # Correct, EM-
                else:
                    correct_em_minus += 1
            # Wrong
            else:
                # Wrong, EM0
                if em_score_evidence[i] == None:
                    wrong_em_zero += 1
                # Wrong, EM+
                elif em_score_evidence[i] == 1:
                    wrong_em_plus += 1
                # Wrong, EM-
                else:
                    wrong_em_minus += 1
        
        return {
            "Correct, EM+": correct_em_plus / 100,
            "Correct, EM-": correct_em_minus / 100,
            "Correct, EM0": correct_em_zero / 100,
            "Wrong, EM+": wrong_em_plus / 100,
            "Wrong, EM-": wrong_em_minus / 100,
            "Wrong, EM0": wrong_em_zero / 100
        }



rouge = Rouge()
def get_rouge_score(file_path):
    df = pd.read_csv(file_path)

    model_output = df['pred'].fillna('').astype(str).tolist()
    reference = df['long_answers'].fillna('').astype(str).tolist()
    short_answers = df['short_answers'].fillna('').astype(str).tolist()
    short_answer_list = []

    for mo, sa in zip(model_output, short_answers):
        reference_list = ast.literal_eval(sa)

        best_score = 0
        best_score_answer = ''
        for ref in reference_list:
            if mo.strip() and ref.strip():
                best_score_answer = ref
                scores = rouge.get_scores(mo, ref)
                rouge_l_f1 = scores[0]['rouge-l']['f']
                if rouge_l_f1 > best_score:
                    best_score = rouge_l_f1
                    best_score_answer = ref
            else:
                best_score_answer = ref
        short_answer_list.append(best_score_answer)
    model_output, reference, short_answer_list = zip(*[(m, r, s) for m, r, s in zip(model_output, reference, short_answer_list) if m.strip()])

    short_scores = rouge.get_scores(list(model_output), list(short_answer_list), avg=True)
    long_scores = rouge.get_scores(list(model_output), list(reference), avg=True)

    short_rouge_l_f1 = short_scores['rouge-l']['f']
    long_rouge_l_f1 = long_scores['rouge-l']['f']

    return {
        "short_rouge_l_f1": short_rouge_l_f1,
        "long_rouge_l_f1": long_rouge_l_f1
    }

def get_f1_score(file_path):
    df = pd.read_csv(file_path)

    model_output = df['pred'].fillna('').astype(str).tolist()
    reference = df['long_answers'].fillna('').astype(str).tolist()
    short_answers = df['short_answers'].fillna('').astype(str).tolist()
    short_answer_list = []

    for mo, sa in zip(model_output, short_answers):
        reference_list = ast.literal_eval(sa)

        # To prevent multiple short_answers: Select the one with the highest rouge-l score for f1 score evaluation
        best_score = 0
        best_score_answer = ''
        for ref in reference_list:
            if mo.strip() and ref.strip():
                best_score_answer = ref
                scores = rouge.get_scores(mo, ref)
                rouge_l_f1 = scores[0]['rouge-l']['f']
                if rouge_l_f1 > best_score:
                    best_score = rouge_l_f1
                    best_score_answer = ref
            else:
                best_score_answer = ref
        short_answer_list.append(best_score_answer)

    f1_scores_long = []
    f1_scores_short = []
    model_output, reference, short_answer_list = zip(*[(m, r, s) for m, r, s in zip(model_output, reference, short_answer_list) if m.strip()])

    for m, r, s in zip(list(model_output), list(reference), list(short_answer_list)):
        m_tokens = m.split()
        r_tokens = r.split()
        s_tokens = s.split()

        all_tokens_long = list(set(m_tokens + r_tokens))
        all_tokens_short = list(set(m_tokens + s_tokens))
        y_true_l = [1 if token in r_tokens else 0 for token in all_tokens_long]
        y_true_s = [1 if token in s_tokens else 0 for token in all_tokens_short]
        y_pred_l = [1 if token in m_tokens else 0 for token in all_tokens_long]
        y_pred_s = [1 if token in m_tokens else 0 for token in all_tokens_short]

        if sum(y_true_l) > 0:
            f1_l = sklearn_f1_score(y_true_l, y_pred_l, average='binary', zero_division=0)
            f1_scores_long.append(f1_l)

        if sum(y_true_s) > 0:
            f1_s = sklearn_f1_score(y_true_s, y_pred_s, average='binary', zero_division=0)
            f1_scores_short.append(f1_s)

    average_f1_long = np.mean(f1_scores_long) if f1_scores_long else 0
    average_f1_short = np.mean(f1_scores_short) if f1_scores_short else 0

    return {
        "short_f1_score": average_f1_short,
        "long_f1_score": average_f1_long
    }
