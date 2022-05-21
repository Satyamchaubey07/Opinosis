import re
from collections import Counter, defaultdict
from operator import itemgetter
from typing import Generator

import networkx as nx
import nltk
import numpy as np

INPUT_FILE_PATH = "input.txt"
MAX_SUMMARIZE_SENTENCES = 2
START_WORD_MAX_POSITION = 3
MAX_GAP = 7
MIN_REDUNDANCY = 1
DO_COLLAPSE = True


def preprocess_input_file():
    updated_text = ""
    input_text = ""

    with open(INPUT_FILE_PATH, "r") as f:
        input_text = f.read()
        updated_text = input_text.replace(". ", ".\n")
        updated_text = updated_text.replace("\n\n", "\n")

    if input_text != updated_text:
        with open(INPUT_FILE_PATH, "w") as f:
            f.write(updated_text)


def parse_input():
    with open(INPUT_FILE_PATH) as f:
        lines = f.readlines()
        for text in lines:
            tokens = nltk.word_tokenize(text.lower())
            tag = nltk.pos_tag(tokens)
            final = " ".join([(f"{word}/{pos}") for word, pos in tag])
            yield final


def create_graph():
    edges = []
    nodes_pri = defaultdict(list)
    for i, line in enumerate(parse_input()):
        line = line.strip()
        if not line:
            continue
        words = line.split()
        words_l = words[:-1]
        words_r = words[1:]
        w_pairs = zip(words_l, words_r)
        edges.extend(w_pairs)
        for j, word in enumerate(words):
            nodes_pri[word].append((i, j))
    edges_cnt = Counter(edges)
    return edges_cnt, nodes_pri


def valid_start_node(node: str, nodes_pri: defaultdict[str, list]):
    start_tag = set(["JJ", "RB", "PRP$", "VBG", "NN", "DT"])
    start_word = set(
        [
            "its",
            "the",
            "when",
            "a",
            "an",
            "this",
            "the",
            "they",
            "it",
            "i",
            "we",
            "our",
            "if",
            "for",
        ]
    )
    pri = nodes_pri[node]
    position = [e[1] for e in pri]
    median = np.median(position)
    if median <= START_WORD_MAX_POSITION:
        word, pos = node.split("/")
        if word in start_word or pos in start_tag:
            return True
    return False


def intersect(pri_so_far: list[list], pri_node: defaultdict[str, list]):
    pri_new = []
    for pri in pri_so_far:
        last_sid, last_pid = pri[-1]
        for sid, pid in pri_node:
            if sid == last_sid and 0 < pid - last_pid <= MAX_GAP:
                pri_tmp = pri[:]
                pri_tmp.append((sid, pid))
                pri_new.append(pri_tmp)
    return pri_new


def valid_end_node(graph: Generator[any, None, None], node: str):
    if "/." in node or "/," in node:
        return True
    elif len(graph[node]) <= 0:
        return True
    else:
        return False


def valid_candidate(sentence: str):
    last = sentence[-1]
    sentence_str = "".join(sentence)
    w, t = last.split("/")
    if t in set(["TO", "VBZ", "IN", "CC", "WDT", "PRP", "DT", ","]):
        return False
    if re.match(".*(/NN)+.*(/VB)+.*(/JJ)+.*", sentence_str):
        return True
    elif re.match(".*(/JJ)+.*(/TO)+.*(/VB).*", sentence_str):
        return True
    elif re.match(".*(/RB)*.*(/JJ)+.*(/NN)+.*", sentence_str) and not re.match(
        ".*(/DT).*", sentence_str
    ):
        return True
    elif re.match(".*(/RB)+.*(/IN)+.*(/NN)+.*", sentence_str):
        return True
    else:
        return False


def path_score(redundancy: int, sen_len: int):
    return np.log2(sen_len) * redundancy


def collapsible(node: str):
    if re.match(".*(/VB[A-Z]|/IN)", node):
        return True
    else:
        return False


def stitch(canchor: int, cc: defaultdict[any, int]):
    if len(cc) == 1:
        return list(cc.keys())[0]
    cc_keys = [pom for pom in cc.keys()]
    ready_keys = []
    for cck in cc_keys:
        cck_cuted = cck.split()[len(canchor) :]
        if len(cck_cuted) == 1:
            tmp = " ".join(cc_keys).split()
            if cck_cuted[0] in tmp and dict(Counter(tmp).items())[cck_cuted[0]] > 1:
                continue
        ready_keys.append(cck_cuted)

    result = print_sentence(" ".join(canchor))
    for i, rk in enumerate(ready_keys):
        if i == 0:
            result = result + print_sentence(" ".join(rk))
        elif i == len(ready_keys) - 1:
            result = result + " and" + print_sentence(" ".join(rk))
        else:
            result = result + "," + print_sentence(" ".join(rk))

    return result


def traverse(
    graph: Generator[any, None, None],
    nodes_pri: defaultdict[str, list],
    node: str,
    sentence: str,
    pri_so_far: list[list],
    score: int,
    clist: defaultdict[any, int],
    collapsed: bool,
):
    redundancy = len(pri_so_far)
    if redundancy >= MIN_REDUNDANCY or valid_end_node(graph, node):
        if valid_end_node(graph, node):
            if sentence[-1].split("/")[1] == "." or sentence[-1].split("/")[1] == ",":
                del sentence[-1]
            if valid_candidate(sentence):
                final_score = score / float(len(sentence))
                clist[" ".join(sentence)] = final_score
            return

        for neighbor in graph[node]:
            redundancy = len(pri_so_far)
            new_sentence = sentence[:]
            new_sentence.append(neighbor)
            new_score = score + path_score(redundancy, len(new_sentence))
            pri_new = intersect(pri_so_far, nodes_pri[neighbor])

            if DO_COLLAPSE and collapsible(neighbor) and not collapsed:
                canchor = new_sentence
                cc = defaultdict(int)
                anchor_score = new_score + path_score(redundancy, len(new_sentence) + 1)
                for vx in graph[neighbor]:
                    pri_vx = intersect(pri_new, nodes_pri[vx])
                    vx_sentence = new_sentence[:]
                    vx_sentence.append(vx)
                    traverse(
                        graph,
                        nodes_pri,
                        vx,
                        vx_sentence,
                        pri_vx,
                        anchor_score,
                        cc,
                        True,
                    )
                if cc:
                    cc_path_score = np.mean(list(cc.values()))
                    final_score = (
                        float(anchor_score) / len(new_sentence) + cc_path_score
                    )
                    stitched_sent = stitch(canchor, cc)
                    clist[stitched_sent] = final_score
            else:
                traverse(
                    graph,
                    nodes_pri,
                    neighbor,
                    new_sentence,
                    pri_new,
                    new_score,
                    clist,
                    False,
                )


def summarize(graph: Generator[any, None, None], nodes_pri: defaultdict[str, list]):
    candidate_list = defaultdict(int)
    for node in nodes_pri:
        if valid_start_node(node, nodes_pri):
            score = 0
            clist = defaultdict(int)
            sentence = [node]
            pri = nodes_pri[node]
            pri_so_far = [[e] for e in pri]
            traverse(graph, nodes_pri, node, sentence, pri_so_far, score, clist, False)
            candidate_list.update(clist)
    return candidate_list


def print_sentence(sen):
    result = ""
    words = sen.split()
    for word in words:
        result += " " + word.split("/")[0]
    return result


if __name__ == "__main__":
    preprocess_input_file()
    edges_cnt, nodes_pri = create_graph()

    with open("edges.tmp", "w") as f:
        for edge in edges_cnt:
            f.write(edge[0] + " " + edge[1] + " " + str(edges_cnt[edge]) + "\n")

    G = nx.read_edgelist("edges.tmp", create_using=nx.DiGraph(), data=(("count", int),))
    candidates = summarize(G, nodes_pri)

    li = list(candidates.items())
    li.sort(key=itemgetter(1), reverse=True)
    for i, sentence in enumerate(li):
        if i < MAX_SUMMARIZE_SENTENCES:
            print(print_sentence(sentence[0]))
            print()
