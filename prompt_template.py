import os
from jinja2 import Template


class PromptTemplate():
    def __init__(self, template_path):
        with open(template_path, 'r') as fp:
            self.template = Template(fp.read())
        self.prompt = self.template.blocks['prompt']

    def init_prompting(self):
        return self.prompt

    def prompting(self, **kwargs):
        context = self.template.new_context(kwargs)
        return ''.join(self.prompt(context)).strip()

    def cot_prompting(self, **kwargs):
        cot_prompt = self.template.blocks['cot']
        context = self.template.new_context(kwargs)
        return ''.join(cot_prompt(context)).strip()


if __name__ == '__main__':
    template_path = 'prompts/default.jinja'