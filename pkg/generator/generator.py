import os
import re
import numpy as np
import pathlib
from tqdm import tqdm
from collections import OrderedDict
import pickle
import subprocess
import glob
import copy
np.set_printoptions(threshold=np.inf)

def get_dq_and_sq(removed_comment):
    dq_count = removed_comment.count('"')
    dq_lists = []
    start = 0
    end = 0
    i = 0
    while True:
        if(dq_count % 2 == 1):
            break
        if(dq_count / 2 != i):
            start = removed_comment.find('"', end)
            end = removed_comment.find('"', start+1)
            e_string = removed_comment[start:end+1]
            end += 1
            dq_lists.append(e_string)
        else:
            break
        i += 1

    sq_count = removed_comment.count("'")
    sq_lists = []
    start = 0
    end = 0
    i = 0

    while True:
        if(sq_count % 2 == 1):
            break
        if(sq_count / 2 != i):
            start = removed_comment.find("'", end)
            end = removed_comment.find("'", start+1)
            e_string = removed_comment[start:end+1]
            end += 1
            sq_lists.append(e_string)
        else:
            break
        i += 1

    return dq_lists, sq_lists


def get_tokens_and_line_numbers(tokens, splited_lines):
    i = 0
    tokens_and_line_numbers = [['<sos>', 1, 0]]
    len_splited_lines = len(splited_lines)
    for j in range(len_splited_lines):
        while(i != len(tokens)):
            if(tokens[i] in splited_lines[j]):
                splited_lines[j] = splited_lines[j].replace(tokens[i], "", 1)
                tokens_and_line_numbers.append([tokens[i], j+1, i+2])
            else:
                break
            i += 1
    return tokens_and_line_numbers



def get_lists_and_line_numbers(tokens, splited_lines):
    i = 0
    tokens_and_line_numbers = []
    len_splited_lines = len(splited_lines)
    for j in range(len_splited_lines):
        while(i != len(tokens)):
            if(tokens[i] in splited_lines[j][0]):
                tokens_and_line_numbers.append([tokens[i], splited_lines[j][1], i])
            else:
                break
            i += 1
    return tokens_and_line_numbers

def split_meta_and_code(code: str):
    line_strip = code.splitlines()
    include_statement = '\n'.join([line for line in line_strip if "include" in line])
    not_include_statement = '\n'.join([line for line in line_strip if not("include" in line or '#' == line or '' == line)])
    return include_statement, not_include_statement

def get_code(judge_id) -> str:
    file_path = '.' + os.sep + "source_code os.sep" + os.sep + str(0) +  str(judge_id) + ".c"
    if os.path.isfile(file_path):
        with open(file_path, encoding="utf-8") as f:
            return f.read()
    else:
        return None

def remove_comment(content: str) -> str:
    """
    remove
    """
    index = 0
    comment_line_inside = False
    comment_block_level = 0
    result = []
    while(index < len(content)):
        if(content[index] == '/' and index + 1 < len(content) \
                and content[index+1] == '*'):
            comment_block_level += 1
        elif(content[index] == '/' and content[index-1] == '*' and comment_block_level != 0):
            comment_block_level -= 1
        elif(content[index] == '/' and index + 1 < len(content) \
                and content[index + 1] == '/'):
            comment_line_inside = True
        elif(content[index] == '\n' and comment_line_inside):
            result.append('\n')
            comment_line_inside = False
        elif(not comment_line_inside and comment_block_level == 0):
            result.append(content[index])
        index += 1
    return ''.join(result)


def add_space(dq_lists, sq_lists, preprocessed_text):
    dq_flag = False
    j = 0
    if(len(dq_lists) > 0):
        for i in range(len(preprocessed_text)):
            if(preprocessed_text[i] == '"'):
                if(dq_flag is False):
                    dq_flag = True
                else:
                    dq_flag = False
                    if(len(dq_lists) == j):
                        break
                    else:
                        dq_lists[j] = dq_lists[j].replace(preprocessed_text[i], "", 1)
                        pass
                    j += 1
            if(dq_flag is True):
                dq_lists[j] = dq_lists[j].replace(preprocessed_text[i], "", 1)
                if(dq_lists[j][0] == ' '):
                    preprocessed_text.insert(i+1, ' ')

    j = 0
    for i in range(len(preprocessed_text)):
        if(preprocessed_text[i] == "'" and preprocessed_text[i+1] == "'"):
            preprocessed_text.insert(i+1, ' ')

    return preprocessed_text


def ld_tokens(s1: list, s2: list, i1: list, i2: list):
    flag = False
    if(len(s1) > len(s2)):
        s1, s2 = s2, s1
        i1, i2 = i2, i1
        flag = True
    l1 = len(s1)
    l2 = len(s2)
    dp = [[0] * (l2 + 1) for i in range(l1 + 1)]
    for i in range(l1 + 1):
        dp[i][0] = i
    for j in range(l2 + 1):
        dp[0][j] = j

    for i in range(1, l1+1):
        for j in range(1, l2+1):
            cost = 2
            if(s1[i-1] == s2[j-1]):
                cost = 0
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    i = 0

    x = l2-1
    y = l1-1
    ses_s1 = []
    ses_s2 = []
    while True:
        if(y == 0 or x == 0):
            break
        ses = [dp[y-1][x-1], dp[y-1][x], dp[y][x-1]]
        min_idx = ses.index(min(ses))
        min_val = min(ses)
        if(min_val == dp[y][x] and min_idx == 0):
            y -= 1
            x -= 1
            continue
        if(min_idx == 0):
            y -= 1
            x -= 1
            ses_s1.append([s1[y], i1[y], s2[x], i2[x], y, x, s1[y-1], i1[y-1]])
            ses_s2.append([s2[x], i2[x], s1[y], i1[y], x, y, s2[x-1], i2[x-1]])
            continue
        if(min_idx == 1):
            y -= 1
            ses_s1.append([s1[y], i1[y], s2[x], i2[x], y, x, s1[y-1], i1[y-1]])
            continue
        if(min_idx == 2):
            x -= 1
            ses_s2.append([s2[x], i2[x], s1[y], i1[y], x, y, s2[x-1], i2[x-1]])
            continue
    ses_s1 = ses_s1[::-1]
    ses_s2 = ses_s2[::-1]
    if(flag is True):
        ses_s1, ses_s2 = ses_s2, ses_s1
        i1, i2 = i2, i1
        
    return dp[-1][-1], ses_s1, ses_s2


def convert_keywords_to_dict(fname: str) -> OrderedDict:
    """
        convert
    """
    keywords = OrderedDict()
    p = pathlib.Path(fname)
    value = 201
    with p.open(mode='r', encoding="utf8") as read_file:
        for line in read_file:
            line = line.rstrip('\n')
            keywords[value] = line
            value += 1
    return keywords


def convert_keywords_to_list(fname: str):
    """
        convert
    """
    p = pathlib.Path(fname)
    keywords = []
    with p.open(mode='r', encoding="utf8") as read_file:
        for line in read_file:
            line = line.rstrip('\n')
            keywords.append(line)
    return keywords


def extract_variable_and_method(tokens: list, keywords: OrderedDict, function_names) -> list:
    variable_names = []
    other_names = []

    pattern1 = r'([+-]?0b[0-9a-fA-F]+)'
    pattern2 = r'([+-]?[0-9]+\.?[0-9]?[eE][+-]?[0-9]+)'
    pattern3 = r'([+-]?[0-9]+\.?[0-9]?)'
    declaration_flag = False
    lists = []

    for token in tokens:
        lists += re.findall(pattern1, token)
        lists += re.findall(pattern2, token)
        lists += re.findall(pattern3, token)
    constant_set = set(lists)
    lists = list(constant_set)
    dq_flag = False
    sq_flag = False
    j = 1
    while j < len(tokens)-2:
        try:
            if(tokens[j] == '"'):
                dq_flag = not dq_flag
            if(tokens[j] == "'"):
                sq_flag = not sq_flag
            if(sq_flag or dq_flag):
                j += 1
                continue
            if('typedef' == tokens[j]):
                while(tokens[j] != ';'):
                    j += 1
                other_names.append(tokens[j-1])
                j += 1
                continue
            if('#' == tokens[j-1] and 'define' == tokens[j] and tokens[j+2] in lists):
                variable_names.append(tokens[j+1])
                j += 1
                continue
            elif(tokens[j+1] == '=' and tokens[j] != '-' \
                    and tokens[j] != ',' and tokens[j] != '=' \
                    and tokens[j] != '*' and tokens[j] != '[' \
                    and tokens[j] != ']' and tokens[j] != '{' \
                    and tokens[j] != '}' and tokens[j] != '+' \
                    and tokens[j] != '|' and tokens[j] != '&' \
                    and tokens[j] != '/' and tokens[j] != '%' \
                    and tokens[j] != '<' and tokens[j] != '>' \
                    and tokens[j] != '!' and tokens[j] not in variable_names):

                variable_names.append(tokens[j])
                j += 1
                continue
            elif(flag_basictype(tokens[j-1])):
                declaration_flag = True
            elif(tokens[j] == ';' or tokens[j] == ')'):
                declaration_flag = False
            if(declaration_flag is True):
                if(tokens[j] in keywords[0:72]):
                    pass
                elif(tokens[j] not in function_names and tokens[j+1] == '(' and tokens[j] not in keywords):
                    function_names.append(tokens[j])

                elif(tokens[j] not in variable_names and tokens[j] != '-' \
                    and tokens[j] != ',' and tokens[j] != '=' \
                    and tokens[j] != '*' and tokens[j] != '[' \
                    and tokens[j] != ']' and tokens[j] != '{' \
                    and tokens[j] != '}' and tokens[j] != '+' \
                    and tokens[j] != '|' and tokens[j] != '&' \
                    and tokens[j] != '/' and tokens[j] != '%' \
                    and tokens[j] != '<' and tokens[j] != '>' \
                    and tokens[j] != '!' and tokens[j] not in lists):
                    variable_names.append(tokens[j])
            else:
                if(tokens[j] in keywords[0:32]):
                    pass
                elif(str.isalpha(tokens[j]) and tokens[j+1] == '(' and tokens[j] not in function_names and tokens[j] not in keywords):
                    function_names.append(tokens[j])
        except IndexError:
            break
        j += 1
    set_variable = set(variable_names)
    variable_names = []
    if('main' in function_names):
        function_names = function_names.remove('main')
    reversed_tokens = tokens[::-1]
    j = 1
    try:
        while j < len(reversed_tokens)-2:
            if(reversed_tokens[j] in set_variable):
                if(not (reversed_tokens[j+1] == '%' or reversed_tokens[j+1] == '\\')):
                    variable_names.append(reversed_tokens[j])
                    set_variable.discard(reversed_tokens[j])
            j += 1
    except IndexError:
        pass
    return variable_names


def flag_basictype(token: str):
    return token == 'int' or token == 'char' or \
        token == 'long' or token == 'String' or \
        token == 'short' or token == 'float' or \
        token == 'double' or token == 'bool'

def allocate_keywords(ses_token: list, keywords, x, y):
    no_variable = 20
    no_function = 30
    lep = []

    before_fix_keywords = keywords.copy()
    variables = [0] * no_variable
    functions = [0] * no_function
    if(ses_token[7] <= no_variable):
        variables[ses_token[7]-1] = ses_token[5]-1
    elif(ses_token[7] <= no_variable + no_function):
        functions[ses_token[7]-(no_variable+1)] = ses_token[5]-1
    else:
        before_fix_keywords[ses_token[6]] = ses_token[5]-1

    before_fix_feature_vector = [0] * len(keywords)
    k = 0
    for _, val in before_fix_keywords.items():
        before_fix_feature_vector[k] = val
        k += 1
    before_fix_feature_vector = variables.copy() + functions.copy() + before_fix_feature_vector
    lep.append(ses_token[6])

    fix_keywords = keywords.copy()
    variables = [0] * no_variable
    functions = [0] * no_function
    if(ses_token[x] <= no_variable):
        variables[ses_token[x]-1] = ses_token[5]
    elif(ses_token[x] <= no_variable + no_function):
        functions[ses_token[x]-(no_variable+1)] = ses_token[5]
    else:
        fix_keywords[ses_token[x-1]] = ses_token[5]

    fix_feature_vector = [0] * len(keywords)
    k = 0
    for _, val in fix_keywords.items():
        fix_feature_vector[k] = val
        k += 1
    fix_feature_vector = variables.copy() + functions.copy() + fix_feature_vector
    lep.append(ses_token[x-1])

    variables = [0] * no_variable
    functions = [0] * no_function
    repair_keywords = keywords.copy()
    if(ses_token[y] <= no_variable):
        variables[ses_token[y]-1] = ses_token[5]
    elif(ses_token[y] <= no_variable + no_function):
        functions[ses_token[y]-(no_variable+1)] = ses_token[5]
    else:
        repair_keywords[ses_token[y-1]] = ses_token[5]

    repair_feature_vector = [0] * len(keywords)
    k = 0
    for _, val in repair_keywords.items():
        repair_feature_vector[k] = val
        k += 1
    repair_feature_vector = variables.copy() + functions.copy() + repair_feature_vector
    lep.append(ses_token[y-1])

    merge = before_fix_feature_vector + fix_feature_vector + repair_feature_vector
    return merge, lep

def split_tokens(text: str) -> list:
    preprocessed_text = remove_comment(text)
    dq_lists, sq_lists = get_dq_and_sq(preprocessed_text)
    preprocessed_text = preprocessed_text.replace('\\r', '')
    preprocessed_text = preprocessed_text.replace('\n', ' ')
    preprocessed_text = preprocessed_text.replace('\\', ' \\ ')
    preprocessed_text = preprocessed_text.replace('\t', ' ')
    preprocessed_text = preprocessed_text.replace(';', ' ; ')
    preprocessed_text = preprocessed_text.replace('#', ' # ')
    preprocessed_text = preprocessed_text.replace('"', ' " ')
    preprocessed_text = preprocessed_text.replace("'", " ' ")
    preprocessed_text = preprocessed_text.replace(':', ' : ')
    preprocessed_text = preprocessed_text.replace('{', ' { ')
    preprocessed_text = preprocessed_text.replace('}', ' } ')
    preprocessed_text = preprocessed_text.replace(',', ' , ')
    preprocessed_text = preprocessed_text.replace('(', ' ( ')
    preprocessed_text = preprocessed_text.replace(')', ' ) ')
    preprocessed_text = preprocessed_text.replace('[', ' [ ')
    preprocessed_text = preprocessed_text.replace(']', ' ] ')
    preprocessed_text = preprocessed_text.replace('+', ' + ')
    preprocessed_text = preprocessed_text.replace('-', ' - ')
    preprocessed_text = preprocessed_text.replace('*', ' * ')
    preprocessed_text = preprocessed_text.replace('/', ' / ')
    preprocessed_text = preprocessed_text.replace('%', ' % ')
    preprocessed_text = preprocessed_text.replace('=', ' = ')
    preprocessed_text = preprocessed_text.replace('<', ' < ')
    preprocessed_text = preprocessed_text.replace('>', ' > ')
    preprocessed_text = preprocessed_text.replace('&', ' & ')
    preprocessed_text = preprocessed_text.replace('|', ' | ')
    preprocessed_text = preprocessed_text.replace('^', ' ^ ')
    preprocessed_text = preprocessed_text.replace('.', ' . ')
    preprocessed_text = preprocessed_text.replace('?', ' ? ')
    preprocessed_text = preprocessed_text.replace('!', ' ! ')
    preprocessed_text = preprocessed_text.split()
    add_space_text = add_space(dq_lists, sq_lists, preprocessed_text)
    return add_space_text


def convert_tokens_to_ids(tokens: list, variables: list, functions: list, keywords: list):
    order_tokens = []
    order_ids = []
    variables_ids = []
    functions_ids = []
    keywords_ids = []
    no_variable = 50
    no_function = 150
    i = 1
    while(i < len(variables)):
        variables_ids.append(i)
        i += 1
    i = no_variable + 1
    tmp = i
    while(i < len(functions)+tmp):
        functions_ids.append(i)
        i += 1
    tmp = no_variable + no_function + 1
    while(i < len(keywords)+tmp):
        keywords_ids.append(i)
        i += 1
    before_token = ""
    before_before_token = ""
    flag_sq = False
    flag_dq = False
    flag_include = False
    for token in tokens:
        if(token == '"'):
            flag_dq = not flag_dq
        if(token == "'"):
            flag_sq = not flag_sq
        if('#' == token):
            flag_include = True
        if('>' and flag_include):
            flag_include = False
        if(token in variables and flag_dq is False and flag_sq is False and flag_include is False):
            order_tokens.append(token)
            order_ids.append(variables.index(token)+1)
        elif(token in functions and flag_dq is False and flag_sq is False and flag_include is False):
            order_tokens.append(token)
            order_ids.append(functions.index(token)+no_variable+1)
        elif(token in keywords and flag_dq is False and flag_sq is False):
            order_tokens.append(token)
            order_ids.append(keywords.index(token)+no_variable+no_function+1)
        else:
            for character in token:
                order_tokens.append(character)
                order_ids.append(keywords.index(character)+no_variable+no_function+1)
        if(before_before_token == "" and before_token == ""):
            before_token = copy.copy(token)
        else:
            before_before_token = copy.copy(before_token)
            before_token = copy.copy(token)
    return order_tokens, order_ids


def tokens2code(words: list) -> str:
    strFlag = True
    code = ""
    for x in words:
        if(x == '\"'):
            strFlag = not strFlag
        if(x == ' ' and strFlag is True):
            code +=  x
        elif(x == ' '):
            pass
        elif(x == '{' or x == ';'):
            code += x + '\n'
        elif(x == '}'):
            code += '\n' + x + '\n'
        else:
            code += ' ' + x

    return code


def extract_variable_and_method_re(tokens: list, keywords: OrderedDict, function_names) -> list:
    variable_names = []
    other_names = []

    pattern1 = r'([+-]?0b[0-9a-fA-F]+)'
    pattern2 = r'([+-]?[0-9]+\.?[0-9]?[eE][+-]?[0-9]+)'
    pattern3 = r'([+-]?[0-9]+\.?[0-9]?)'
    declaration_flag = False
    lists = []

    for token in tokens:
        lists += re.findall(pattern1, token)
        lists += re.findall(pattern2, token)
        lists += re.findall(pattern3, token)
    constant_set = set(lists)
    lists = list(constant_set)
    dq_flag = False
    sq_flag = False
    equal_flag = False
    j = 1
    while j < len(tokens)-2:
        try:
            if(tokens[j] == '"'):
                dq_flag = not dq_flag
            if(tokens[j] == "'"):
                sq_flag = not sq_flag
            if(sq_flag is True or dq_flag is True):
                j += 1
                continue
            if('typedef' == tokens[j]):
                while(tokens[j] != ';'):
                    j += 1
                other_names.append(tokens[j-1])
                j += 1
                continue
            if('#' == tokens[j-1] and 'define' == tokens[j] and tokens[j+2] in lists):
                variable_names.append(tokens[j+1])
                j += 1
                continue
            elif(flag_basictype(tokens[j-1])):
                declaration_flag = True
            elif(tokens[j] == ';' or tokens[j] == ')'):
                declaration_flag = False
                equal_flag = False
            if(declaration_flag is True):
                if(tokens[j] == '='):
                    equal_flag = True
                elif(tokens[j] == ','):
                    equal_flag = False
                if(tokens[j] in keywords[0:73]):
                    pass
                elif(tokens[j] not in function_names and tokens[j+1] == '(' and tokens[j] not in keywords):
                    function_names.append(tokens[j])
                elif(equal_flag is False and tokens[j] not in variable_names and tokens[j] not in lists \
                    and tokens[j] != '-' and tokens[j] != '!'\
                    and tokens[j] != ',' and tokens[j] != '=' \
                    and tokens[j] != '*' and tokens[j] != '[' \
                    and tokens[j] != ']' and tokens[j] != '{' \
                    and tokens[j] != '}' and tokens[j] != '+' \
                    and tokens[j] != '|' and tokens[j] != '&' \
                    and tokens[j] != '/' and tokens[j] != '%' \
                    and tokens[j] != '<' and tokens[j] != '>' \
                    and tokens[j] != ')'):
                    variable_names.append(tokens[j])
            else:
                if(str.isalpha(tokens[j]) and tokens[j+1] == '(' and tokens[j] not in function_names and tokens[j] not in keywords):
                    function_names.append(tokens[j])

        except IndexError:
            break
        j += 1
    set_variable = set(variable_names)
    variable_names = []
    if('main' in function_names):
        function_names = function_names.remove('main')
    flag = False
    j = 1
    try:
        while j < len(tokens)-2:
            if(tokens[j] == 'scanf' or tokens[j] == 'for'):
                flag = True
            if(tokens[j] == ';' or tokens[j] == ')' or tokens[j] == '{'):
                flag = False
            if(not (tokens[j-1] == '%' or tokens[j-1] == '\\')):
                if(tokens[j] in set_variable and flag is True):
                    variable_names.append(tokens[j])
                    set_variable.discard(tokens[j])
            j += 1
    except IndexError:
        pass
    # reversed_tokens = tokens[::-1]
    # j = 1
    # try:
    #     while j < len(reversed_tokens)-2:
    #         if(reversed_tokens[j] in set_variable):
    #             if(not (reversed_tokens[j+1] == '%' or reversed_tokens[j+1] == '\\')):
    #                 variable_names.append(reversed_tokens[j])
    #                 set_variable.discard(reversed_tokens[j])
    #         j += 1
    # except IndexError:
    #     pass
    j = 1
    try:
        while j < len(tokens)-2:
            if(tokens[j] in set_variable):
                if(not (tokens[j-1] == '.' or tokens[j-1] == '%' or tokens[j-1] == '\\')):
                    variable_names.append(tokens[j])
                    set_variable.discard(tokens[j])
            j += 1
    except IndexError:
        pass

    return variable_names
