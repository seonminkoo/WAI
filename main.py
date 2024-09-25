import argparse
from setproctitle import setproctitle
from utils import *
from prompt_template import PromptTemplate
from tqdm import tqdm
import pandas as pd


setproctitle("your process name")

def read_json(file_path):
    with open(file_path, "r") as f:
        datas = json.load(f)

    return datas


def arrange_papragraph(document, long_answer, related_information, doc_setting, d_type):
    res_paras = []

    if d_type == "rel":
        for num in range(len(document)):
            if num not in related_information:
                ans_para_num = num

        ans_para = document[int(ans_para_num)]

        if doc_setting == "rar":
            document.pop(int(ans_para_num))
            other_para = document

            index = int(round(len(other_para) / 2))
            other_para_head = document[:index]
            other_para_tail = document[index:]

            res_paras.extend(other_para_head)
            res_paras.append(ans_para)
            res_paras.extend(other_para_tail)

        elif doc_setting =="arr":
            document.pop(int(ans_para_num))
            other_para = document

            res_paras.append(ans_para)
            res_paras.extend(other_para)

        elif doc_setting =="rra":
            document.pop(int(ans_para_num))
            other_para = document

            res_paras.extend(other_para)
            res_paras.append(ans_para)

    elif d_type == "unrel":
        index = int(round(len(document)/2))
        other_para_head = document[:index]
        other_para_tail = document[index:]
        ans_para = long_answer

        if doc_setting == "iai":
            res_paras.extend(other_para_head)
            res_paras.append(ans_para)
            res_paras.extend(other_para_tail)
        elif doc_setting == "aii":
            res_paras.append(ans_para)
            res_paras.extend(other_para_head)
            res_paras.extend(other_para_tail)
        elif doc_setting == "iia":
            res_paras.extend(other_para_head)
            res_paras.extend(other_para_tail)
            res_paras.append(ans_para)

    elif d_type == "mixed":
        index = len(related_information)
        other_para_rel = document[:index]
        other_para_unrel = document[index:]
        ans_para = long_answer

        if doc_setting == "rai":
            res_paras.extend(other_para_rel)
            res_paras.append(ans_para)
            res_paras.extend(other_para_unrel)
        elif doc_setting == "iar":
            res_paras.extend(other_para_unrel)
            res_paras.append(ans_para)
            res_paras.extend(other_para_rel)
        elif doc_setting == "ari":
            res_paras.append(ans_para)
            res_paras.extend(other_para_rel)
            res_paras.extend(other_para_unrel)
        elif doc_setting == "air":
            res_paras.append(ans_para)
            res_paras.extend(other_para_unrel)
            res_paras.extend(other_para_rel)
        elif doc_setting == "ria":
            res_paras.extend(other_para_rel)
            res_paras.extend(other_para_unrel)
            res_paras.append(ans_para)
        elif doc_setting == "ira":
            res_paras.extend(other_para_unrel)
            res_paras.extend(other_para_rel)
            res_paras.append(ans_para)

    return res_paras

def insert_p_num(doc_texts):
    result = ""

    for i, para in enumerate(doc_texts):
        result += f"[{i}]{para}"

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", default="", type=str, required=True)
    parser.add_argument("--output_path", default="", type=str, required=True)
    parser.add_argument("--api_key", default="", type=str)
    parser.add_argument("--start_idx", default=0, type=int)
    parser.add_argument("--model_type", default="gpt", type=str)
    parser.add_argument("--doc_setting", default="", type=str, required=True)

    args = parser.parse_args()

    file_path = args.input_path

    if args.model_type == "gpt":
        if args.api_key != "":
            api_key = args.api_key
        else:
            with open("./openai_api_key.txt", 'r') as f:
                api_key = f.readline().strip()
    elif args.model_type in "mixtral":
        if args.api_key != "":
            api_key = args.api_key
        else:
            with open("./mixtral_api_key.txt", 'r') as f:
                api_key = f.readline().strip()
    elif args.model_type == "claude3":
        if args.api_key != "":
            api_key = args.api_key
        else:
            with open("./claude3_api_key.txt", 'r') as f:
                api_key = f.readline().strip()
    else:
        api_key = ""

    # Data Load
    datas = read_json(file_path)

    chat_res = []
    save_datas = []

    if args.start_idx != 0:
        datas = datas[args.start_idx:]

    if "_rel" in os.path.split(args.output_path)[1]:
        label = "rel"
    elif "_unrel" in os.path.split(args.output_path)[1]:
        label = "unrel"
    elif "_mixed" in os.path.split(args.output_path)[1]:
        label = "mixed"

    print("##################################################################")
    print(f"MODEL: {args.model_type}, dataset type: {label}")
    print("##################################################################")

    for idx, data in enumerate(tqdm(datas[:])):
        title = data["title"]
        document_text = data["document_text"]
        question_text = data["question_text"]
        long_answers = data["annotations"]["long_answer"]
        short_answers = data["annotations"]["short_answers"]
        document_url = data["document_url"]
        example_id = data["example_id"]

        llm_tagged = []

        if label != "unrel":
            related_information = data["related_information"]
            document_text = arrange_papragraph(document_text, long_answers, related_information, args.doc_setting, label)
        else:
            document_text = arrange_papragraph(document_text, long_answers, "UNREL", args.doc_setting, label)

        input_paras = insert_p_num(document_text)

        Writer = MinuteWriter(input_paras, question_text, api_key, args.model_type)

        if "claude3" in args.model_type:
            prompt_template = PromptTemplate("prompts/default.jinja")

            temp_answer = long_claude3_res(input_paras, question_text, Writer, prompt_template)
            answer = temp_answer[0].replace("\n", " ").strip()


            if label == "unrel":
                chat_res = [
                    [example_id, title, document_text, question_text, long_answers, short_answers,
                     document_url, answer]]
                res = pd.DataFrame(data=chat_res,
                                   columns=["example_id", "title", "document_text",
                                            "question_text", "long_answers", "short_answers", "document_url", "pred"])

            else:
                chat_res = [
                    [example_id, title, document_text, related_information, question_text, long_answers, short_answers,
                     document_url, answer]]
                res = pd.DataFrame(data=chat_res,
                                   columns=["example_id", "title", "document_text", "related_information",
                                            "question_text", "long_answers", "short_answers", "document_url", "pred"])

        elif "mixtral" in args.model_type:
            prompt_template = PromptTemplate("prompts/default.jinja")

            temp_answer = long_mixtral_res(input_paras, question_text, Writer, prompt_template)
            answer = temp_answer[0].replace("\n", " ").strip()


            if label == "unrel":
                chat_res = [
                    [example_id, title, document_text, question_text, long_answers, short_answers,
                     document_url, answer]]
                res = pd.DataFrame(data=chat_res,
                                   columns=["example_id", "title", "document_text",
                                            "question_text", "long_answers", "short_answers", "document_url", "pred"])

            else:
                chat_res = [
                    [example_id, title, document_text, related_information, question_text, long_answers, short_answers,
                     document_url, answer]]
                res = pd.DataFrame(data=chat_res,
                                   columns=["example_id", "title", "document_text", "related_information",
                                            "question_text", "long_answers", "short_answers", "document_url", "pred"])

        else:  # gpt
            if 'chain' in args.output_path:
                typ = 'chain'
                prompt_template = PromptTemplate("prompts/cot.jinja")

                reasoning_path = long_vanila_res(input_paras, question_text, Writer, prompt_template)
                reasoning_path = reasoning_path[0].replace("\n", " ").strip()

                temp_answer = long_cot_res(input_paras, question_text, Writer, prompt_template, reasoning_path, typ)
                answer = temp_answer[0].replace("\n", " ").strip()


            elif 'one-shots' in args.output_path:
                typ = 'one-shots'
                prompt_template = PromptTemplate("prompts/one-shots.jinja")

                temp_answer = long_vanila_res(input_paras, question_text, Writer, prompt_template)
                answer = temp_answer[0].replace("\n", " ").strip()

            else:
                typ = 'baseline'
                prompt_template = PromptTemplate("prompts/default.jinja")

                temp_answer = long_vanila_res(input_paras, question_text, Writer, prompt_template)
                answer = temp_answer[0].replace("\n", " ").strip()


            # Configure values to save dataframe
            if ('chain' in args.output_path) or ('decomposed' in args.output_path):
                if label == "unrel":
                    chat_res = [
                        [example_id, title, document_text, question_text, long_answers, short_answers,
                         document_url, answer, reasoning_path]]
                    res = pd.DataFrame(data=chat_res,
                                       columns=["example_id", "title", "document_text",
                                                "question_text", "long_answers", "short_answers", "document_url",
                                                "pred", "llm_reasoning"])
                else:
                    chat_res = [
                        [example_id, title, document_text, related_information, question_text, long_answers, short_answers,
                         document_url, answer, reasoning_path]]
                    res = pd.DataFrame(data=chat_res,
                                       columns=["example_id", "title", "document_text", "related_information",
                                                "question_text", "long_answers", "short_answers", "document_url", "pred", "llm_reasoning"])
            else: # baseline saveing value
                if label == "unrel":
                    chat_res = [
                        [example_id, title, document_text, question_text, long_answers, short_answers,
                         document_url, answer]]
                    res = pd.DataFrame(data=chat_res,
                                       columns=["example_id", "title", "document_text",
                                                "question_text", "long_answers", "short_answers", "document_url", "pred"])

                else:


                    chat_res = [[example_id, title, document_text, related_information, question_text, long_answers, short_answers, document_url, answer]]
                    res = pd.DataFrame(data=chat_res, columns=["example_id", "title", "document_text", "related_information", "question_text", "long_answers", "short_answers", "document_url", "pred"])

        output_path = args.output_path

        output_dir = "/".join(output_path.split("/")[:-1])
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(output_path):
            res.to_csv(output_path, index=False, mode='w')
        else:
            res.to_csv(output_path, index=False, mode='a', header=False)

    print(f"saved file: {output_path}")

