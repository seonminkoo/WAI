import openai
import time
from prompt_template import PromptTemplate

class MinuteWriter():
    def __init__(self, content, question, api_key, model_type):

        if 'gpt' in model_type:
            openai.api_key = api_key

            self.model = "gpt-3.5-turbo-16k"
            # self.model = "gpt-4"

        elif "claude3" in model_type:
            import anthropic

            self.model = anthropic.Anthropic(
                api_key=api_key)

        else:
            from llamaapi import LlamaAPI
            self.model = LlamaAPI(api_key)

        self.content = content


    def mixtral_write(self, content, question, pt:PromptTemplate, reasoning_path=None, typ=None):
        if reasoning_path is not None:
            return
        else:
            prompt = pt.prompting(**{'context': content, 'question': question})

            user_content = ""

        api_request_json = {
            "model": "mixtral-8x7b-instruct",
            # "max_length": 512,
            "messages": [
                {"role": "system",
                 "content": prompt},
                {"role": "user", "content": user_content},
            ]
        }


        retries = 0
        while(1):
            try:
                response = self.model.run(api_request_json)

                answer = response.json()['choices'][0]['message']['content']

                return [answer]
            except:
                retries += 1
                print(f"Request failed. Retrying ({retries} …")
                time.sleep(2 ** retries)

    def claude3_write(self, content, question, pt: PromptTemplate, reasoning_path=None, typ=None):
        if reasoning_path is not None:
            return
        else:
            prompt = pt.prompting(**{'context': content, 'question': question})

            message = self.model.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                temperature=0.0,
                system="",
                # system="""Generate an answer (A) to the question (Q) based on the given context.
                # Also, provide the evidence paragraph number (Evidence num) that supports your answer.""",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            result = message.content[0].text

            return [result]

    def write(self, content, question, pt:PromptTemplate, reasoning_path=None, typ=None):
        if reasoning_path is not None:
            if "chain" in typ:
                # prompt = pt.prompting()
                prompt = pt.prompting(**{'context': content, 'question': question})

                cot_prompt = pt.cot_prompting(**{'reasoning_path': reasoning_path})

                merged_prompt = prompt + "\n" + cot_prompt

                print("merged_prompt: ", merged_prompt)

                self.messages = [
                    {"role": "system",
                     "content": merged_prompt}
                ]

        else:
            prompt = pt.prompting(**{'context': content, 'question': question})

            self.messages = [
                {"role": "system",
                 "content": prompt}
            ]

        user_content = ""


        self.messages.append({"role": "user", "content": user_content})

        results = []
        retries = 0

        while(1):
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=self.messages
                )
                result = response['choices'][0]['message']['content']
                self.messages.append(dict(response['choices'][0]['message']))

                results.append(result)

                return results
            except:
                retries += 1
                print(f"Request failed. Retrying ({retries} …")
                time.sleep(2 ** retries)

