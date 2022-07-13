import configparser
import subprocess
import json
import re
import copy

config = configparser.ConfigParser()
config.read('config.ini')

def get_tokenizer_cmd(language) -> list:
    cmd = ""
    if (language is None):
        cmd = None
    elif (language == 'C'):
        cmd = [config['TOKENIZER']['PATH'], "-m", "json", "-l", language, "-n"]
    elif (language == 'CPP'):
        cmd = [config['TOKENIZER']['PATH'], "-m", "json", "-l", language, "-n"]
    return cmd


def tokenize_by_file(filepath: str, language) -> list:
    cmd = get_tokenizer_cmd(language)
    cmd.append(filepath)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    list_json = []
    try:
        if (proc.stderr != None):
            err = proc.stderr.decode("utf8").strip()
        if (len(err) > 0):
            return None
        list_json = json.loads(proc.stdout.decode("utf8"))
        # return json.loads(proc.stdout.decode("utf8"))
    except json.decoder.JSONDecodeError:
        list_json = None
    return list_json

def tokenize_by_code(code: str, language) -> list:
    cmd = get_tokenizer_cmd(language)
    proc = subprocess.run(cmd, input=code.encode("utf8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        if (proc.stderr != None):
            err = proc.stderr.decode("utf8").strip()
            if (len(err) > 0):
                return None
        return json.loads(proc.stdout.decode("utf8"))
    except json.decoder.JSONDecodeError:
        return None


def detokenize(tokens: list) -> str:
    if (tokens == None):
        return None
    code = ""
    current_line_tokens = []
    indent_count = 0
    flag_keywords = False
    flag_cout_cin = False
    flag_define = False
    keywords = ['if', 'while', 'for', 'switch']
    parent_count = 0
    before_token = tokens[0]

    for token in tokens:
        cls = token["class"]
        raw = token["token"]
        before_cls = before_token["class"]
        before_raw = before_token["token"]

        if (cls == "newline" and flag_define is True):
            code += format_line_by_tokens(current_line_tokens, indent_count)
            current_line_tokens = []
            flag_define is False
        elif(flag_define is True):
            current_line_tokens.append(token)
        elif (cls == "newline" and (flag_keywords is False and flag_cout_cin is False)):
            if(flag_define is True):
                flag_define = False
            if (len(current_line_tokens) == 0):
                pass
            else:
                code += format_line_by_tokens(current_line_tokens, indent_count)
                current_line_tokens = []
            continue
        elif (raw == ';' and flag_keywords is False):
            if (len(current_line_tokens) == 0 and (code[-2:] == ')\n' or before_cls == 'identifier')):
                code = code.strip()
                current_line_tokens.append(token)
                code += format_line_by_tokens(current_line_tokens, indent_count) 
                current_line_tokens = []
            else:
                current_line_tokens.append(token)
                code += format_line_by_tokens(current_line_tokens, indent_count) 
                current_line_tokens = []
        elif (raw == '}' or raw == '{'):
            if (len(current_line_tokens) != 0):
                code += format_line_by_tokens(current_line_tokens, indent_count)
                current_line_tokens = []
            current_line_tokens.append(token)
            code += format_line_by_tokens(current_line_tokens, indent_count)
            current_line_tokens = []
        # elif (before_raw == ')' and cls == "identifier" and flag_cout_cin is False and flag_keywords is False and flag_define is False):
        #     code += format_line_by_tokens(current_line_tokens, indent_count)
        #     current_line_tokens = []
        #     current_line_tokens.append(token)
        else:
            if(cls != "newline"):
                current_line_tokens.append(token)
        if(raw == '('):
            parent_count += 1
        elif(raw == ')'):
            parent_count -= 1

        if(raw in keywords and before_cls != 'preprocessor'):
            flag_keywords = True
        elif (flag_keywords is True and parent_count == 0):
            flag_keywords = False

        if(raw in ['cin', 'cout']):
            flag_cout_cin = True
        elif(raw == ';' and flag_cout_cin is True):
            flag_cout_cin = False
        if(before_cls == 'preprocessor' and raw == 'define'):
            flag_define = True

        before_token = token


    # 最後に改行がないとき append されないので
    if (len(current_line_tokens) > 0):
        code += format_line_by_tokens(current_line_tokens, indent_count)
    return code

def format_line_by_tokens(tokens: list, indent_count: int) -> str:
    LITERALS = [ "integer", "floating", "string", "character" ]
    code = ""
    for i, token in enumerate(tokens):
        cls = token["class"]
        raw = token["token"]
        if(i > 0):
            code += " "
        code += str(raw)
    code += '\n'
    return code

def detokenize_lines(tokens: list) -> list:
    if (tokens == None):
        return None
    result = []
    current_line_tokens = []
    keywords = ["if", "while", "for", "switch"]
    flag_keywords = False
    flag_cout_cin = False
    flag_define = False
    indent_count = 0
    parent_count = 0
    before_token = tokens[0]
    for token in tokens:
        cls = token["class"]
        raw = token["token"]
        before_cls = before_token["class"]
        before_raw = before_token["token"]

        if (cls == "newline" and flag_define is True):
            result.append(format_line_by_tokens_hoge(current_line_tokens, indent_count))
            current_line_tokens = []
            flag_define is False
        elif(flag_define is True):
            current_line_tokens.append(token)
        elif (cls == "newline" and (flag_keywords is False and flag_cout_cin is False)):
            if (len(current_line_tokens) == 0):
                continue
            result.append(format_line_by_tokens_hoge(current_line_tokens, indent_count))
            current_line_tokens = []
            continue
        elif (raw == ';' and flag_keywords is False):
            if (len(current_line_tokens) == 0 and (result[-1][-1] == ')' or before_cls == "identifier")):
                result[-1].append(';')
            else:
                current_line_tokens.append(token)
                result.append(format_line_by_tokens_hoge(current_line_tokens, indent_count))
                current_line_tokens = []
        elif (raw == '}' or raw == '{'):
            if (len(current_line_tokens) != 0):
                result.append(format_line_by_tokens_hoge(current_line_tokens, indent_count))
                current_line_tokens = []
            current_line_tokens.append(token)
            result.append(format_line_by_tokens_hoge(current_line_tokens, indent_count))
            current_line_tokens = []
        # elif (before_raw == ')' and cls == "identifier" and flag_cout_cin is False and flag_keywords is False and flag_define is False):
        #     result.append(format_line_by_tokens_hoge(current_line_tokens, indent_count))
        #     current_line_tokens = []
        #     current_line_tokens.append(token)
        else:
            if(cls != "newline"):
                current_line_tokens.append(token)

        if(raw == '('):
            parent_count += 1
        elif(raw == ')'):
            parent_count -= 1

        if(raw in keywords and before_cls != 'preprocessor'):
            flag_keywords = True
        elif (flag_keywords is True and parent_count == 0):
            flag_keywords = False

        if(raw in ['cin', 'cout']):
            flag_cout_cin = True
        elif(raw == ';' and flag_cout_cin is True):
            flag_cout_cin = False
        if(before_cls == 'preprocessor' and raw == 'define'):
            flag_define = True
        

        before_token = token
    # 最後に改行がないとき append されないので
    if (len(current_line_tokens) > 0):
        result.append(format_line_by_tokens_hoge(current_line_tokens, indent_count))
    return result

def format_line_by_tokens_hoge(tokens: list, indent_count: int) -> list:
    LITERALS = [ "integer", "floating", "string", "character" ]
    result = [''] * indent_count
    for i, token in enumerate(tokens):
        cls = token["class"]
        raw = token["token"]
        # リテラル
        if(i > 0):
            result.append(" ")
        result.append(str(raw))
    return result

