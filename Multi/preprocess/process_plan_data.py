import re
import copy
import json
import sympy
import numpy as np
from english import find_numbers_in_text
from EasyData.NLPHandler import is_math_num

def load_raw_data(filename):  # load the json data to list(dict()) for MATH 23K
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    js = ""
    data = []
    for i, s in enumerate(f):
        js += s
        i += 1
        if i % 7 == 0:  # every 7 line is a json
            data_d = json.loads(js)
            if "千米/小时" in data_d["equation"]:
                data_d["equation"] = data_d["equation"][:-5]
            data.append(data_d)
            js = ""

    return data

def from_infix_to_prefix(expression):
    st = list()
    res = list()
    priority = {"=":-1, "+": 0, "-": 0, "*": 1, "/": 1, "^": 2}
    expression = copy.deepcopy(expression)
    expression.reverse()
    for e in expression:
        if e in [")", "]"]:
            st.append(e)
        elif e == "(":
            c = st.pop()
            while c != ")":
                res.append(c)
                c = st.pop()
        elif e == "[":
            c = st.pop()
            while c != "]":
                res.append(c)
                c = st.pop()
        elif e in priority:
            while len(st) > 0 and st[-1] not in [")", "]"] and priority[e] < priority[st[-1]]:
                res.append(st.pop())
            st.append(e)
        else:
            res.append(e)
    while len(st) > 0:
        res.append(st.pop())
    res.reverse()
    return res

def from_infix_to_postfix(expression):
    st = list()
    res = list()
    priority = {"=":-1, "+": 0, "-": 0, "*": 1, "/": 1, "^": 2}
    for e in expression:
        if e in ["(", "["]:
            st.append(e)
        elif e == ")":
            c = st.pop()
            while c != "(":
                res.append(c)
                c = st.pop()
        elif e == "]":
            c = st.pop()
            while c != "[":
                res.append(c)
                c = st.pop()
        elif e in priority:
            while len(st) > 0 and st[-1] not in ["(", "["] and priority[e] <= priority[st[-1]]:
                res.append(st.pop())
            st.append(e)
        else:
            res.append(e)
    while len(st) > 0:
        res.append(st.pop())
    return res

def from_prefix_to_infix(prefix):
    st = []
    operators = ["+", "-", "^", "*", "/", "=", ";"]
    prefix = copy.deepcopy(prefix)
    prefix.reverse()
    for p in prefix:
        if p not in operators:
            st.append(p)
        elif p == "+" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(" ".join([a, "+", b]))
        elif p == "*" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(" ".join(["(", a, ")", "*", "(", b, ")"]))
        elif p == "/" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(" ".join(["(", a, ")", "/", "(", b, ")"]))
        elif p == "-" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(" ".join(["(", a, ")", "-", "(", b, ")"]))
        elif p == "^" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(" ".join(["(", a, ")", "^", "(", b, ")"]))
        elif p == "=" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(" ".join([a, "=", b]))
        elif p == ";" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(" ".join([a, ";", b]))
        else:
            return None
    if len(st) == 1:
        return st.pop()

def from_postfix_to_infix(postfix):
    st = []
    operators = ["+", "-", "^", "*", "/", "="]
    for p in postfix:
        if p not in operators:
            st.append(p)
        elif p == "+" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(" ".join([b, "+", a]))
        elif p == "*" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(" ".join(["(", b, ")", "*", "(", a, ")"]))
        elif p == "/" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(" ".join(["(", b, ")", "/", "(", a, ")"]))
        elif p == "-" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(" ".join(["(", b, ")", "-", "(", a, ")"]))
        elif p == "^" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(" ".join(["(", b, ")", "^", "(", a, ")"]))
        elif p == "=" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(" ".join([b, "=", a]))
        else:
            return None
    if len(st) == 1:
        return st.pop()

def compute_ans(ans_infix):
    equs = ans_infix.split(' ; ')
    if len(equs) == 1:
        equal_pos1 = equs[0].find("=")
        infix1_sy = sympy.Eq(sympy.sympify(equs[0][:equal_pos1]), sympy.sympify(equs[0][equal_pos1+1:]))
        X_0 = sympy.symbols("X_0")
        infix_res = sympy.solve([infix1_sy], [X_0])
    else:
        equal_pos1 = equs[0].find("=")
        equal_pos2 = equs[1].find("=")
        infix1_sy = sympy.Eq(sympy.sympify(equs[0][:equal_pos1]), sympy.sympify(equs[0][equal_pos1+1:]))
        infix2_sy = sympy.Eq(sympy.sympify(equs[1][:equal_pos2]), sympy.sympify(equs[1][equal_pos2+1:]))
        X_0 = sympy.symbols("X_0")
        X_1 = sympy.symbols("X_1")
        infix_res = sympy.solve([infix1_sy, infix2_sy], [X_0, X_1])
    return infix_res

def out_expression_list(test, nums):
    res = []
    for i in test:
        if i[0] == 'N':
            res.append(nums[int(i[2:])%len(nums)])
        elif i[0] == 'C':
            if i == 'C_1_NEG':
                res.append('-1')
            else:
                res.append(i[2:].replace('_', '.'))
        else:
            res.append(i)
    return res

def compute_prefix_expression(pre_fix):
    st = list()
    operators = ["+", "-", "^", "*", "/"]
    pre_fix = copy.deepcopy(pre_fix)
    pre_fix.reverse()
    for p in pre_fix:
        if p not in operators:
            st.append(eval(p))
        elif p == "+" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a + b)
        elif p == "*" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a * b)
        elif p == "/" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            if b == 0:
                return None
            st.append(a / b)
        elif p == "-" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a - b)
        elif p == "^" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a ** b)
        else:
            return None
    if len(st) == 1:
        return st.pop()
    return None


def transfer_num(data):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pairs = []
    for i in range(len(data)):
        d = data[i]
        if d['id'] in [739292, 15575326, 1373548]:
            continue
        nums = []
        input_seq = []

        seg = d["original_text"].strip().split()
        input_seq, nums = find_numbers_in_text(d['original_text'].strip())
        # for s in seg:
        #     pos = re.search(pattern, s) # 搜索每个词的数字位置
        #     if pos and pos.start() == 0:
        #         nums.append(s[pos.start():pos.end()])
        #         # input_seq.append('_'+s[pos.start():pos.end()]+'[N]')

        #         if pos.end() < len(s):
        #             input_seq.append(s[pos.end():])
        #     elif s != "":
        #         input_seq.append(s)


        # equations = d["equation"]
        equations = [str(float(eval(t.strip()))) if is_math_num(t) else t for t in d["equation"].split()]
        equations = " ".join(equations)


        nums_fraction = []
        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True) # 从大到小排序

        float_nums = []
        for num in nums:
            if ',' in num:
                new_num = []
                for c in num:
                    if c == ',':
                        continue
                    new_num.append(c)
                num = ''.join(new_num)
                float_nums.append(str(float(eval(num.strip()))))
            elif '%' in num:
                float_nums.append(str(float(eval(num[:-1].strip()) / 100)))
            elif len(num) > 1 and num[0] == '0':
                float_nums.append(str(float(eval(num[1:].strip()))))
            else:
                float_nums.append(str(float(eval(num.strip()))))

        float_nums_fraction = []
        for num in nums_fraction:
            if ',' in num:
                new_num = []
                for c in num:
                    if c == ',':
                        continue
                    new_num.append(c)
                num = ''.join(new_num)
                float_nums_fraction.append(str(float(eval(num.strip()))))
            elif '%' in num:
                float_nums.append(str(float(eval(num[:-1].strip()) / 100)))
            else:
                float_nums_fraction.append(str(float(eval(num.strip()))))
        # print(float_nums)
        # print(float_nums_fraction)
        nums = float_nums
        nums_fraction = float_nums_fraction

        def seg_and_tag(st):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) >= 1:
                        res.append("N_" + str(nums.index(n)))
                    else:
                        if eval(n) == round(eval(n), 1):
                            n = str(round(eval(n), 1))
                        n = "C_" + n
                        n = n.replace('.', '_')
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res

            pos_st = re.search("\d+\.\d+%?|\d+%?", st) # 带百分号的数字数
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) >= 1:
                    res.append("N_"+str(nums.index(st_num)))
                else:
                    if eval(st_num) == round(eval(st_num), 1):
                        st_num = str(round(eval(st_num), 1))
                    st_num = "C_" + st_num
                    st_num = st_num.replace('.', '_')
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        out_seq = seg_and_tag(equations)
        new_out_seq = []
        for seq in out_seq:
            if seq == ' ' or seq == '':
                continue
            if seq == ';':
                new_out_seq.append(';')
                continue
            new_out_seq.append(seq)
        
        num_values = []
        for p in nums:
            pos1 = re.search("\d+\(", p)
            pos2 = re.search("\)\d+", p)
            if pos1:
                num_values.append(str(eval(p[pos1.start(): pos1.end() - 1] + "+" + p[pos1.end() - 1:])))
            elif pos2:
                num_values.append(str(eval(p[:pos2.start() + 1] + "+" + p[pos2.start() + 1: pos2.end()])))
            elif p[-1] == "%":
                num_values.append(str(float(p[:-1]) / 100))
            else:
                num_values.append(str(eval(p)))
        
        out_seq = new_out_seq
        out_seq = ' '.join(out_seq)
        out_seq = out_seq.replace('x', 'X_0')
        out_seq = out_seq.replace('y', 'X_1')
        out_seq = out_seq.split(' ; ')
        out_seq = [x.split(' ') for x in out_seq]
        temp = {}
        temp['id'] = str(d['id'])
        temp['text'] = ''.join(input_seq)
        temp['original_text'] = d['original_text']
        prefix = [from_infix_to_prefix(x) for x in out_seq]
        postfix = [from_infix_to_postfix(x) for x in out_seq]
        if len(postfix) > 2:
            print(d)
            continue
        prefix = sum(prefix, [])
        prefix = [';'] * (prefix.count('=')-1) + prefix
        temp['prefix'] = prefix
        temp['postfix'] = postfix
        ans_infix = from_prefix_to_infix(out_expression_list(prefix, num_values))
        ans = compute_ans(ans_infix)
        if type(ans) == type(dict()):
            ans = list(ans.values())
        else:
            new_ans = []
            for a in ans:
                for b in a:
                    new_ans.append(b)
            ans = new_ans
        ans = [float(x) for x in ans]
        ans.sort()
        real_ans = d['ans']
        real_ans.sort()
        result = True
        if len(ans) != len(real_ans):
            result = False
        if result:
            for i,value in enumerate(ans):
                if abs(ans[i] - real_ans[i]) > 1e-3:
                    result = False
        if not result:
            print(d)

        temp['nums'] = num_values
        temp['answer'] = ans
        pairs.append(temp)

    return pairs

def transfer_english_num(data):
    print("Transfer numbers...")
    pattern = re.compile("\d+,\d+|\d+\.\d+|\d+|-\d+")
    pairs = []
    for d in data:
        nums = []
        input_seq = []
        # seg = d["original_text"].strip().split(" ")
        equations = copy.deepcopy(d['expr'])
        equations = [x[1] for x in equations if x[0] == 0]
        equations = [x.replace('C_1_NEG', 'C_-1') for x in equations]
        equations = [x.split(' ') for x in equations]
        input_seq, nums = find_numbers_in_text(d['text'].strip())

        out_seq = [from_postfix_to_infix(x) for x in equations]
        out_seq = [x.split(' ') for x in out_seq]
        num_values = []
        for p in nums:
            pos1 = re.search("\d+\(", p)
            pos2 = re.search("\)\d+", p)
            if pos1:
                num_values.append(str(eval(p[pos1.start(): pos1.end() - 1] + "+" + p[pos1.end() - 1:])))
            elif pos2:
                num_values.append(str(eval(p[:pos2.start() + 1] + "+" + p[pos2.start() + 1: pos2.end()])))
            elif p[-1] == "%":
                num_values.append(str(float(p[:-1]) / 100))
            else:
                num_values.append(str(eval(p)))
        
        temp = {}
        temp['id'] = str(d['id'])
        # temp['answer'] = d['ans']
        temp['text'] = input_seq
        temp['original_text'] = d['original']['sQuestion']
        temp['infix'] = out_seq
        prefix = [from_infix_to_prefix(x) for x in out_seq]
        postfix = [from_infix_to_postfix(x) for x in out_seq]
        if len(prefix) > 1:
            prefix_sep_token = []
            for sep_pos in range(len(prefix)-1):
                prefix_sep_token += [';'] + prefix[sep_pos]
            prefix_sep_token += prefix[sep_pos+1]
            prefix = prefix_sep_token
        else:
            prefix = prefix[0]
        temp['prefix'] = prefix
        temp['postfix'] = postfix
        ans_infix = from_prefix_to_infix(out_expression_list(prefix, num_values))
        ans = compute_ans(ans_infix)
        if type(ans) == type(dict()):
            ans = list(ans.values())
        else:
            new_ans = []
            for a in ans:
                for b in a:
                    new_ans.append(b)
            ans = new_ans
        ans = [float(x) for x in ans]
        ans.sort()
        real_ans = d['original']['lSolutions']
        real_ans.sort()
        result = True
        if len(ans) != len(real_ans):
            result = False
        for i,value in enumerate(ans):
            if abs(ans[i] - real_ans[i]) > 1e-3:
                result = False
        if not result:
            print(d)

        temp['nums'] = num_values
        temp['answer'] = ans
        pairs.append(temp)

    return pairs

def load_data(filename):
    data = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# train_data = load_data('../data/draw/draw_train.orig.jsonl')
# dev_data = load_data('../data/draw/draw_dev.orig.jsonl')
# test_data = load_data('../data/draw/draw_test.orig.jsonl')
# train_data = transfer_english_num(train_data)
# dev_data = transfer_english_num(dev_data)
# test_data = transfer_english_num(test_data)
# idx = 0
# f = open('../data/draw/dev_test/DRAW_train.jsonl', 'w')
# for d in train_data:
#     d['id'] = str(idx)
#     json.dump(d, f, ensure_ascii=False)
#     f.write("\n")
#     idx += 1
# f.close()
# f = open('../data/draw/dev_test/DRAW_dev.jsonl', 'w')
# for d in dev_data:
#     d['id'] = str(idx)
#     json.dump(d, f, ensure_ascii=False)
#     f.write("\n")
#     idx += 1
# f.close()
# f = open('../data/draw/dev_test/DRAW_test.jsonl', 'w')
# for d in test_data:
#     d['id'] = str(idx)
#     json.dump(d, f, ensure_ascii=False)
#     f.write("\n")
#     idx += 1
# f.close()


import argparse
from EasyData.FileHandler import check2mkdir, path_append, abs_current_dir
import os
import sys
CUR_FILE_DIR = os.path.dirname(os.path.abspath(__file__))


def add_args(parser):
    parser.add_argument('-data_dir', type=str, default=None)
    parser.add_argument('-work_mode',
                        type=str,
                        choices=['gold', 'pred'],
                        default="pred")
    parser.add_argument('--data_name', type=str, default='make')
    parser.add_argument('--data_version', type=int, default=97)


parser = argparse.ArgumentParser(description='[Get args for wrok data]')
add_args(parser)
work_opt = parser.parse_args()

DATA_NAME = work_opt.data_name
# 读取 题目数据
DATA_DICT = {
    "make": ["chinese", "muti_equation"],
    "hmwp": ["chinese", "muti_equation"],
    "dophin": ["english", "muti_equation"],
    "equation": ["chinese", "muti_equation"],

    "arithmetic": ["english", "linear_expression"],
    "mawps": ["english", "linear_expression"],
    "math23k_en": ["english", "linear_expression"],
    "linear": ["english", "linear_expression"],
    "nl4opt": ["english", "optimization"],
}
DATA_TYPE = DATA_DICT[DATA_NAME][1]
DATA_VERSION = work_opt.data_version

data_dir = work_opt.data_dir
work_mode = work_opt.work_mode

def revocer_num(_token, _memery_num_to_idx):
    t = _memery_num_to_idx.get(_token, _token)
    return str(t)

def rectity_ques(items):
    rec_items = []
    for idx, item in enumerate(items):
        _memery_num_to_idx = item["memery_num_to_idx"]
        _memery_idx_to_num = {f"num_{v}": k for k,v in _memery_num_to_idx.items()}
        if "disturb_num_to_idx" in item:
            # 弃用 disturb_num 模板化
            memery_disturb_idx_to_num = {f"disturb_{i}": d for d, i in item["disturb_num_to_idx"].items()}
            for i, sent in enumerate(item["question_sents"]):
                item["question_sents"][i] = [memery_disturb_idx_to_num.get(t, t) for t in sent]
            ques_tokens = [revocer_num(t, _memery_idx_to_num) for s in item["question_sents"] for t in s]
        else:
            ques_tokens = [revocer_num(t, _memery_idx_to_num) for s in item["question_sents"] for t in s]

        if len(item["adjust_equations"]) > 1:
            equation = " ; ".join(item["adjust_equations"])
        else:
            equation = " ".join(item["adjust_equations"])

        if "answer" not in item:
            answer = []
        else:
            answer = [ eval(num) if isinstance(num, str) else num for num in item["answer"]]
        rec_items.append({
            "id": idx,
            "original_text": " ".join(ques_tokens),
            "equation": equation,
            "ans": answer,
        })
    return rec_items


def work_test_data():
    test_data_path = f"{data_dir}/{DATA_NAME}_test.json"
    print(f"Load data from {test_data_path} ... ")
    
    # save_data_path = path_append(abs_current_dir(__file__), f'../data/{DATA_NAME}/{DATA_VERSION}/{DATA_NAME}_test.jsonl', to_str=True) 
    
    save_data_path = f"{data_dir}/{DATA_NAME}_test.jsonl"
    check2mkdir(save_data_path)

    test_data = json.load(open(test_data_path, 'r', encoding="utf-8"))
    test_data = rectity_ques(test_data)

    test_data = transfer_num(test_data)

    f = open(save_data_path, 'w', encoding="utf-8")
    for d in test_data:
        json.dump(d, f, ensure_ascii=False)
        f.write("\n")
    f.close()
    print(f"Save data to {save_data_path} ... ")
    pass


def work_gold_data(mode):
    # work_data_path = f"E:\\Workustc\\Math-Plan\\data\\{DATA_TYPE}\\{DATA_NAME}\\{DATA_VERSION}\\{DATA_NAME}_{mode}.json"
    work_data_path = f"/data/qlh/Math-Plan/data/{DATA_TYPE}/{DATA_NAME}/{DATA_VERSION}/{DATA_NAME}_{mode}.json"
    print(f"Load data from {work_data_path} ... ")

    if mode == "valid":
        mode = "dev"
    save_data_path = path_append(abs_current_dir(__file__), f'../data/{DATA_NAME}/dev_test/{DATA_VERSION}/{DATA_NAME}_{mode}.jsonl', to_str=True) 
    check2mkdir(save_data_path)

    test_data = json.load(open(work_data_path, 'r', encoding="utf-8"))
    test_data = rectity_ques(test_data)

    test_data = transfer_num(test_data)

    f = open(save_data_path, 'w', encoding="utf-8")
    for d in test_data:
        json.dump(d, f, ensure_ascii=False)
        f.write("\n")
    f.close()
    print(f"Save data to {save_data_path} ... ")
    print(f"Finishing ... {mode}")

if work_mode == "pred":
    work_test_data()
elif work_mode == "gold":
    work_gold_data("train")
    work_gold_data("valid")
    work_gold_data("test")
