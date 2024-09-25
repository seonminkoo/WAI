import os
from get_scores import get_short_EM_score, get_long_EM_score, get_rouge_score, get_f1_score, get_em_evidence
from utils import parse_args, save_results_to_csv

if __name__ == '__main__':
    args = parse_args()
    file_base_path = args.input_file

    results = []

    for file_name in os.listdir(file_base_path):
        file_path = os.path.join(file_base_path, file_name)

        print(f"Getting Results for {file_name}...")

        short_em_score = get_short_EM_score(file_path)
        evidence_em_score = short_em_score['Evidence EM Score']
        no_evidence_proportion = short_em_score['No Evidence Proportion']
        short_answers_em_score = short_em_score['Short answers EM Score']

        long_em_score = get_long_EM_score(file_path)

        rouge_score = get_rouge_score(file_path)
        short_rouge_l_f1 = rouge_score['short_rouge_l_f1']
        long_rouge_l_f1 = rouge_score['long_rouge_l_f1']

        f1_score = get_f1_score(file_path)
        short_f1_score = f1_score['short_f1_score']
        long_f1_score = f1_score['long_f1_score']

        em_evidence_score = get_em_evidence(file_path)
        correct_em_zero = em_evidence_score['Correct, EM0']
        correct_em_plus = em_evidence_score['Correct, EM+']
        correct_em_minus = em_evidence_score['Correct, EM-']
        wrong_em_zero = em_evidence_score['Wrong, EM0']
        wrong_em_plus = em_evidence_score['Wrong, EM+']
        wrong_em_minus = em_evidence_score['Wrong, EM-']
        
        results.append({
            "file_name": file_name,
            "evidence_em_score": evidence_em_score,
            "no_evidence_proportion": no_evidence_proportion,
            "short_em_score": short_answers_em_score,
            "long_em_score": long_em_score,
            "short_rouge_l_f1_score": short_rouge_l_f1,
            "long_rouge_l_f1_score": long_rouge_l_f1,
            "short_f1_score": short_f1_score,
            "long_f1_score": long_f1_score,
            "evidence_correct_em_zero": correct_em_zero,
            "evidence_correct_em_plus": correct_em_plus,
            "evidence_correct_em_minus": correct_em_minus,
            "evidence_wrong_em_zero": wrong_em_zero,
            "evidence_wrong_em_plus": wrong_em_plus,
            "evidence_wrong_em_minus": wrong_em_minus
        })

    save_results_to_csv(results, args.output_file)