import os
import json
import random
from prompt_template import PromptTemplate
from minutes_writer import *
import sys
import pandas as pd

def long_vanila_res(content, question, pred_model, prompt_template):
    res = pred_model.write(content, question, prompt_template)

    return res

def long_cot_res(content, question, pred_model, prompt_template, reasoning_path, typ):
    res = pred_model.write(content, question, prompt_template, reasoning_path, typ)

    return res


def long_mixtral_res(content, question, pred_model, prompt_template):
    res = pred_model.mixtral_write(content, question, prompt_template)

    return res


def long_claude3_res(content, question, pred_model, prompt_template):
    res = pred_model.claude3_write(content, question, prompt_template)

    return res

