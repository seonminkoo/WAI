import re
import argparse
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model results.")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input file.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output file.')
    return parser.parse_args()

def save_results_to_csv(results, output_path):
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)

def extract_answer_and_evidence(pred):
    if not isinstance(pred, str):
        pred = ""

    match = re.search(r'(.*)\s*\(Evidence\s*(\d+)\)$', pred, re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
        evidence_num = int(match.group(2).strip())
    else:
        match = re.match(r'Evidence num[:\s]*(\d+)\s*(.*)', pred, re.IGNORECASE)
        if match:
            evidence_num = int(match.group(1).strip())
            answer = match.group(2).strip()
        else:
            match = re.search(r'(.*?)(?:\s*\(?(?:Evidence num[:\s]*(\d+)|Evidence\s*(\d+)|\[(\d+)\]|\((\d+)\))\)?)', pred, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                evidence_num = match.group(2) or match.group(3) or match.group(4) or match.group(5)
                evidence_num = int(evidence_num.strip()) if evidence_num else None
            else:
                parts = pred.split('.')
                if len(parts) > 1 and re.search(r'\(?(?:Evidence num[:\s]*(\d+)|Evidence\s*(\d+)|\[(\d+)\]|\((\d+)\))\)?', parts[-1], re.IGNORECASE):
                    answer = '.'.join(parts[:-1]).strip()
                    evidence_num = re.search(r'\(?(?:Evidence num[:\s]*(\d+)|Evidence\s*(\d+)|\[(\d+)\]|\((\d+)\))\)?', parts[-1], re.IGNORECASE).group(1).strip()
                    evidence_num = int(evidence_num)
                elif pred.strip().isdigit():
                    answer = ""
                    evidence_num = int(pred.strip())
                else:
                    answer = pred.strip()
                    evidence_num = None

    answer = re.sub(r'^A:\s*', '', answer)
    answer = re.sub(r'\s*\(?Evidence\s*\d*\)?', '', answer)
    answer = re.sub(r'\s*Evidence num[:\s]*\(?\d+\)?', '', answer, flags=re.IGNORECASE)
    answer = re.sub(r'\s*This is evidenced in the context in$', '', answer)
    answer = re.sub(r'\(\s*$', '', answer)
    answer = re.sub(r'\.\s*$', '', answer)
    answer = re.sub(r'\s*$', '', answer)
    answer = re.sub(r'\bnum:\s*$', '', answer)

    return answer, evidence_num