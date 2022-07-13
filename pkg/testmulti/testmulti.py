# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import pathlib
import pickle
import requests
import sys
import matplotlib.pyplot as plt
import subprocess
import glob
import copy
import more_itertools
import itertools
from tqdm import tqdm
from collections import OrderedDict
from statistics import mean, variance
from scipy import stats
from joblib import Parallel, delayed

def judge_compile_error_source_code(s):
    name = "a.c"
    with open(name, 'w', encoding="utf8") as f:
        f.write(s)

    proc = subprocess.CompletedProcess
    if(os.name == 'nt'):
        compile_command = ["gcc", "-O0", "-Wl,--stack,134217728", "-W", name]
        # proc = subprocess.run(compile_command, stdout=subprocess.PIPE, check=True)
        proc = subprocess.run(compile_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    elif(os.name == 'posix'):
        compile_command = ["gcc", "-O0", "-W", name, "-lm"]
        # proc = subprocess.run(compile_command, stdout=subprocess.PIPE, check=True)
        proc = subprocess.run(compile_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    if(len(proc.stdout) == 0):
        flag = True
    else:
        flag = False
    return flag

def generate_feature_vector(code, lists):
    pid =  'res'
    name = pid + ".c"
    flag = True
    with open(name, 'w', encoding="utf8") as f:
        f.write(code)
    feature_vector = []
    proc = subprocess.CompletedProcess
    try:
        if(os.name == 'nt'):
            compile_command = ["gcc", "-O0", "-Wl,--stack,134217728", "-Wall", name, "-o", pid]
            proc = subprocess.run(compile_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
        elif(os.name == 'posix'):
            compile_command = ["gcc", "-O0", '-w', name, "-lm", "-o", pid]
            proc = subprocess.run(compile_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError:
        flag = False
        return False, []
        # print(s)
    if(len(proc.stdout) == 0):
        for test_case in lists:
            try:
                in_txt = test_case[0]
                out_txt = test_case[1]
                in_file = open(in_txt, 'r')
                out_file = open(pid + ".out", 'w')
                if(os.name == 'nt'):
                    cmd = ["." + os.sep + pid + '.exe']
                    proc = subprocess.run(cmd, stdin=in_file, capture_output=True, check=True, timeout=1)
                    out_file.write(proc.stdout.decode('utf8'))
                    out_file.close()
                    proc = subprocess.run(['diff', out_txt, pid + ".out"], capture_output=True, check=False)
                elif(os.name == 'posix'):
                    cmd = ["." + os.sep + pid]
                    proc = subprocess.run(cmd, stdin=in_file, capture_output=True, check=True, timeout=1)
                    out_file.write(proc.stdout.decode('utf8'))
                    out_file.close()
                    proc = subprocess.run(['echo', '$?'], capture_output=True, check=False, shell=True)
                    proc = subprocess.run(['diff', '-q', out_txt, pid + ".out"], capture_output=True, check=False)
                if(len(proc.stdout) != 0):
                    feature_vector.append(0)
                    flag = False
                else:
                    feature_vector.append(1)
                in_file.close()
                out_file.close()
            except subprocess.CalledProcessError:
                feature_vector.append(0)
                flag = False
                # print(s)
            except subprocess.TimeoutExpired:
                feature_vector.append(0)
                flag = False
            except UnicodeDecodeError:
                feature_vector.append(0)
                flag = False

    # if(os.name == 'nt'):
    #     if(os.path.exists(name) is True):
    #         os.remove(name)
    #     if(os.path.exists(pid + '.exe') is True):
    #         os.remove(pid + '.exe')
    #     if(os.path.exists(pid + '.out') is True):
    #         os.remove(pid + '.out')
    # elif(os.name == 'posix'):
    #     if(os.path.exists(name) is True):
    #         os.remove(name)
    #     if(os.path.exists(pid) is True):
    #         os.remove(pid)
    #     if(os.path.exists(pid + '.out') is True):
    #         os.remove(pid + '.out')
    return flag, feature_vector


def extract_users_list(target_problem_course, target_name):
    users = OrderedDict()
    if(False):
        url = "https://judgeapi.u-aizu.ac.jp/submission_records/problems/" + target_name + "/?size=1500000"
        w_file = open('..' + os.sep + target_problem_course + os.sep + target_name + os.sep + 'judged_data.pickle', 'wb')
        response = requests.get(url)
        records = response.json()
        list.reverse(records)
        pickle.dump(records, w_file)
    r_file = open('..' + os.sep + target_problem_course + os.sep + target_name + os.sep + 'judged_data.pickle', "rb")
    records = pickle.load(r_file)
    for record in tqdm(records):
        if((record["language"] != "C")):
            continue

        if(record["userId"] not in users and record["status"] == 4):
            code = get_code(record["judgeId"])
            if(code is None):
                continue
            try:
                code_size = len(remove_comment(code))
                if(code_size == 0):
                    continue
                if(code.find("__asm__") == -1 and code.find("#ifdef") == -1 and code.find("#endif") == -1 and code.find("#pragma") == -1):
                # if(code.find("__asm__") == -1 and judge_function_macro(code) is False and code.find("#ifdef") == -1 and code.find("#pragma") == -1):
                    if(code.count('"') % 2 == 1  or code.count("'") % 2 == 1):
                        continue
                    users[record["userId"]] = [record["judgeId"]]
                    users[record["userId"]].append(record["status"])
            except UnicodeError:
                continue
            except ValueError:
                continue
            except UnicodeEncodeError:
                print("error")
                continue

    for record in tqdm(records):
        # print(record)
        if((record["language"] != "C") or record["status"] == 4 or record["status"] <= 0):
            continue

        if(record["userId"] in users):
            if(users[record["userId"]][0] < record["judgeId"]):
                continue
            code = get_code(record["judgeId"])
            if(code is None):
                continue
            try:
                if(code_size == 0):
                    continue
                if(code.find("__asm__") == -1 and code.find("#ifdef") == -1 and code.find("#endif") == -1 and code.find("#pragma") == -1):
                    if(code.count('"') % 2 == 1  or code.count("'") % 2 == 1):
                        continue

                    users[record["userId"]].append(record["judgeId"])
                    users[record["userId"]].append(record["status"])
            except UnicodeError:
                continue
            except ValueError:
                continue
            except UnicodeEncodeError:
                print("error")
                continue
    pairs = []
    for user, pair in tqdm(users.items()):
        if(len(pair) == 4):
            pairs.append(pair)
    correct_only_codes = []
    for user, pair in tqdm(users.items()):
        if(len(pair) == 2):
            correct_only_codes.append(pair)
    del records
    return pairs, correct_only_codes


def judge_function_macro(code: str):
    flag = False
    removed_comment = remove_comment(code)
    preprocessed_text = preprocessing_text(removed_comment)
    i = 2
    while i < len(preprocessed_text):
        try:
            if(preprocessed_text[i-2] == '#' and preprocessed_text[i-1] == 'define' and preprocessed_text[i+1] == '('):
                flag = True
                break
            if(preprocessed_text[i-2] == '#' and preprocessed_text[i-1] == 'define' and preprocessed_text[i+2] == '('):
                flag = True
                break
        except IndexError:
            break
        i += 1
    return flag


def get_code(judge_id) -> str:
    file_path = "../../aoj_archive/archive_migrate/" + str(judge_id) + ".txt"
    if os.path.isfile(file_path):
        with open(file_path, encoding="utf-8", newline='\n') as f:
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


def add_space(dq_lists, sq_lists, preprocessed_text):
    dq_flag = False
    sq_flag = False
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


def preprocessing_text(text: str) -> list:
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
    preprocessed_text = preprocessed_text.replace('.', ' . ')
    preprocessed_text = preprocessed_text.replace('?', ' ? ')
    preprocessed_text = preprocessed_text.replace('!', ' ! ')
    preprocessed_text = preprocessed_text.replace('^', ' ^ ')
    preprocessed_text = preprocessed_text.split()
    add_space_text = add_space(dq_lists, sq_lists, preprocessed_text)

    return add_space_text


def ld_tokens(s1: list, s2: list):
    flag = False
    if(len(s1) > len(s2)):
        s1, s2 = s2, s1
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
            if(s1[i-1][0] == s2[j-1][0]):
                cost = 0
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    i = 0

    x = l2
    y = l1
    ses_s1 = []
    ses_s2 = []
    while True:
        if(y == 0 or x == 0):
            break
        l = [dp[y-1][x-1], dp[y-1][x], dp[y][x-1]]
        min_idx = l.index(min(l))
        min_val = min(l)
        if(min_val == dp[y][x] and min_idx == 0):
            y -= 1
            x -= 1
            continue
        if(min_idx == 0):
            y -= 1
            x -= 1
            ses_s1.append(s1[y])
            ses_s2.append(s2[x])
            continue
        if(min_idx == 1):
            y -= 1
            ses_s1.append(s1[y])
            continue
        if(min_idx == 2):
            x -= 1
            ses_s2.append(s2[x])
            continue
    ses_s1 = ses_s1[::-1]
    ses_s2 = ses_s2[::-1]
    if(flag is True):
        ses_s1, ses_s2 = ses_s2, ses_s1
    return dp[-1][-1], ses_s1, ses_s2


def ld_str(s1: list, s2: list):
    flag = False
    if(len(s1) > len(s2)):
        s1, s2 = s2, s1
        flag = True
    if(s1 == s2):
        if(flag is True):
            s1, s2 = s2, s1
        return len(s1), s1, s2
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
    x = l2
    y = l1
    ses_s1 = []
    ses_s2 = []
    while True:
        if(y == 0 or x == 0):
            break
        l = [dp[y-1][x-1], dp[y-1][x], dp[y][x-1]]
        min_idx = l.index(min(l))
        min_val = min(l)
        if(min_val == dp[y][x] and min_idx == 0):
            y -= 1
            x -= 1
            continue
        if(min_idx == 0):
            y -= 1
            x -= 1
            ses_s1.append(s1[y])
            ses_s2.append(s2[x])
            continue
        if(min_idx == 1):
            y -= 1
            ses_s1.append(s1[y])
            continue
        if(min_idx == 2):
            x -= 1
            ses_s2.append(s2[x])
            continue
    if(flag is True):
        ses_s1, ses_s2 = ses_s2, ses_s1
    return dp[-1][-1], ses_s1, ses_s2

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


def extract_variable_and_method(tokens: list, keywords: OrderedDict) -> list:
    variable_names = []
    function_names = []
    other_names = []

    pattern1 = r'([+-]?0b[0-9a-fA-F]+)'
    pattern2 = r'([+-]?[0-9]+\.?[0-9]?[eE][+-]?[0-9]+)'
    pattern3 = r'([+-]?[0-9]+\.?[0-9]?)'
    j = 1
    declaration_flag = False
    lists = []

    for token in tokens:
        lists += re.findall(pattern1, token)
        lists += re.findall(pattern2, token)
        lists += re.findall(pattern3, token)
    constant_set = set(lists)
    lists = list(constant_set)
    while j < len(tokens)-2:
        try:
            if('main' == tokens[j]):
                function_names.append(tokens[j])
                if(declaration_flag is True):
                    declaration_flag = False
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
            elif(flag_basictype(tokens[j-1]) and tokens[j-2] != '(' and tokens[j] != ')'):
                declaration_flag = True
            elif(tokens[j] == ';' or tokens[j] == ')'):
                declaration_flag = False
            if(declaration_flag is True):
                # print(tokens[j])
                if(tokens[j] in keywords[0:50]):
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
    # variable_set = set(variable_names)
    # function_set = set(function_names)
    # variable_names = list(variable_set)
    # function_names = list(function_set)
    return variable_names, function_names


def flag_basictype(token: str):
    return token == 'int' or token == 'char' or \
        token == 'long' or token == 'String' or \
        token == 'short' or token == 'float' or \
        token == 'double' or token == 'bool'


def get_tokens_and_line_numbers(tokens, splited_lines):
    i = 0
    tokens_and_line_numbers = []
    len_splited_lines = len(splited_lines)
    pattern_include = r'(#include\s*<\S*>)'
    flag_end_include = False
    for j in range(len_splited_lines):
        result = re.match(pattern_include, splited_lines[j])
        while(i != len(tokens)):
            if(tokens[i] in splited_lines[j] and result):
                if(flag_end_include is False):
                    splited_lines[j] = splited_lines[j].replace(tokens[i], "", 1)
                    tokens_and_line_numbers.append([tokens[i], j+1, i])
                else:
                    splited_lines[j] = splited_lines[j].replace(tokens[i], "", 1)
                    del(tokens[i])
                    continue
                if(tokens[i] == '>'):
                    flag_end_include = True
            elif(tokens[i] in splited_lines[j]):
                splited_lines[j] = splited_lines[j].replace(tokens[i], "", 1)
                tokens_and_line_numbers.append([tokens[i], j+1, i])
            else:
                flag_end_include = False
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


def convert_tokens_to_ids(tokens: list, variables: list, functions: list, keywords: list):
    order_tokens = []
    order_ids = []
    variables_ids = []
    functions_ids = []
    keywords_ids = []
    i = 1
    while(i < len(variables)):
        variables_ids.append(i)
        i += 1
    i = 21
    tmp = i
    while(i < len(functions)+tmp):
        functions_ids.append(i)
        i += 1
    tmp = 41
    while(i < len(keywords)+tmp):
        keywords_ids.append(i)
        i += 1
    i = 0
    before_token = ""
    before_before_token = ""
    flag = False
    flag_sq = False
    flag_dq = False
    for token in tokens:
        if(before_before_token == '#' and before_token == 'include' and token == '<'):
            flag = True
        elif(flag is True and token == '>'):
            flag = False
        if(token == '"'):
            flag_dq = not flag_dq
        if(token == "'"):
            flag_sq = not flag_sq
        if(token in variables and flag_dq is False and flag_sq is False and flag is False):
            order_tokens.append(token)
            order_ids.append(variables.index(token)+1)
        elif(token in functions and flag_dq is False and flag_sq is False and flag is False):
            order_tokens.append(token)
            order_ids.append(functions.index(token)+21)
        elif(token in keywords):
            order_tokens.append(token)
            order_ids.append(keywords.index(token)+41)
        elif(token not in keywords):
            for character in token:
                order_tokens.append(character)
                order_ids.append(keywords.index(character)+41)
        if(before_before_token == "" and before_token == ""):
            before_token = copy.copy(token)
        else:
            before_before_token = copy.copy(before_token)
            before_token = copy.copy(token)
    return order_tokens, order_ids


def set_files(target) -> list:
    target_names = []
    if(target == 'all'):
        # target_names.append("ITP1_1_A")
        # target_names.append("ITP1_1_B")
        # target_names.append("ITP1_1_C")
        # target_names.append("ITP1_1_D")
        # target_names.append("ITP1_2_A")
        # target_names.append("ITP1_2_B")
        # target_names.append("ITP1_2_C")
        # target_names.append("ITP1_2_D")
        # target_names.append("ITP1_3_A")
        # target_names.append("ITP1_3_B")
        # target_names.append("ITP1_3_C")
        # target_names.append("ITP1_3_D")
        # target_names.append("ITP1_4_A")
        # target_names.append("ITP1_4_B")
        # target_names.append("ITP1_4_C")
        # target_names.append("ITP1_4_D")
        # target_names.append("ITP1_5_A")
        # target_names.append("ITP1_5_B")
        # target_names.append("ITP1_5_C")
        # target_names.append("ITP1_5_D")
        # target_names.append("ITP1_6_A")
        # target_names.append("ITP1_6_B")
        # target_names.append("ITP1_6_C")
        # target_names.append("ITP1_6_D")
        # target_names.append("ITP1_7_A")
        # target_names.append("ITP1_7_B")
        # target_names.append("ITP1_7_C")
        # target_names.append("ITP1_7_D")
        # target_names.append("ITP1_8_A")
        # target_names.append("ITP1_8_B")
        # target_names.append("ITP1_8_C")
        # target_names.append("ITP1_8_D")
        # target_names.append("ITP1_9_A")
        # target_names.append("ITP1_9_B")
        # target_names.append("ITP1_9_C")
        # target_names.append("ITP1_9_D")
        # target_names.append("ITP1_10_A")
        # target_names.append("ITP1_10_B")
        # target_names.append("ITP1_10_C")
        # target_names.append("ITP1_10_D")
        # target_names.append("ITP1_11_A")
        # target_names.append("ITP1_11_B")
        # target_names.append("ITP1_11_C")
        # target_names.append("ITP1_11_D")
        # target_names.append("ALDS1_1_A")
        target_names.append("ALDS1_1_D")
    else:
        target_names.append(target)
    return target_names


def make_directories(problem_course, target_names):
    current_dir = os.listdir('..')
    os.chdir('..')
    if(problem_course not in current_dir):
        os.mkdir(problem_course)
    dirs = os.listdir(problem_course)
    for folder in dirs:
        if(problem_course not in current_dir):
            os.mkdir(problem_course)
    os.chdir(problem_course)
    current_dir = os.listdir('.')
    for target_name in target_names:
        if(target_name not in current_dir):
            os.mkdir(target_name)


