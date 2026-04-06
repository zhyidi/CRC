#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json
import os
import sys
import logging
from eval_dialogue import label_knowledge, calc_f1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

total_efficient_data = 0

def read_data(fpath):
    result = []
    with open(fpath, "r", encoding="utf-8") as fr:
        for line in fr:
            sample = json.loads(line)
            result.append(sample)
    return result


def save_to_file(responses, output_path):
    with open(output_path, "w", encoding="utf-8") as fw:
        for response in responses:
            line = json.dumps(response, ensure_ascii=False)
            fw.write(line + "\n")
            fw.flush()


def convert_data(sample, is_gold=False, lower_case=True):
    samples = []
    all_knowledges = []
    gold_knowledges = []
    response = sample["response"].lower() if lower_case else sample["response"]
    resp = [tok for tok in response]  # token-level list
    samples.append(resp)
    if is_gold:
        knowledge = sample["knowledge"]
        all, gold = label_knowledge(response, knowledge, lower_case=lower_case)
        all_knowledges.append(all)
        gold_knowledges.append(gold)
        assert len(samples) == len(all_knowledges)
        assert len(samples) == len(gold_knowledges)
        return (samples, all_knowledges, gold_knowledges)
    else:
        return samples

def calc_succ(eval_sample, gold_sample):
    if gold_sample["topic_path"][0].lower() in gold_sample["response"].lower():
        global total_efficient_data
        total_efficient_data += 1
        if gold_sample["topic_path"][0].lower() not in eval_sample["response"].lower():
            return True
    return False

def spilt_response(args, lower_case=True):
    logging.info("reading data...")
    tcps = read_data(args.TCP_file)
    dialogues = read_data(args.pre_dialogue_file)
    golds = read_data(args.gold_file)
    assert len(tcps) == len(dialogues)
    assert len(tcps) == len(golds)

    logging.info("spilting data...")
    dialogues_samegoal = []
    dialogues_differentgoal = []
    golds_samegoal = []
    golds_differentgoal = []
    dialogues_samegoal_unconsistency = []
    golds_samegoal_unconsistency = []
    for tcp, dialogue, gold in zip(tcps, dialogues, golds):
        pre_act = tcp["action"]
        gold_act = gold["action_path"][0]
        pre_topic = tcp["topic"]
        gold_topic = gold["topic_path"][0]
        if lower_case:
            pre_act = pre_act.lower()
            gold_act = gold_act.lower()
            pre_topic = pre_topic.lower()
            gold_topic = gold_topic.lower()
        if pre_act == gold_act and pre_topic == gold_topic:
            dialogues_samegoal.append(dialogue)
            golds_samegoal.append(gold)

            # preds = convert_data(dialogue)
            # refs, all_knowledges, ref_knowlwedges = convert_data(gold, is_gold=True)
            if calc_succ(dialogue, gold):
                dialogues_samegoal_unconsistency.append(dialogue)
                golds_samegoal_unconsistency.append(gold)
        else:
            dialogues_differentgoal.append(dialogue)
            golds_differentgoal.append(gold)

    assert len(dialogues_samegoal) == len(golds_samegoal)
    assert len(dialogues_differentgoal) == len(golds_differentgoal)
    assert len(dialogues_samegoal_unconsistency) == len(golds_samegoal_unconsistency)

    if not os.path.exists(args.output_dir):
        logging.info("mkdir output_dir:{}".format(args.output_dir))
        os.makedirs(args.output_dir)
    logging.info("saving data...")
    print("Plan-Dial不一致的有：{}条数据。".format(len(dialogues_samegoal_unconsistency)))
    print("测试集有效数据：{}条。".format(total_efficient_data))
    # save_to_file(dialogues_samegoal, os.path.join(args.output_dir, "pre_samegoal.txt"))
    # save_to_file(dialogues_differentgoal, os.path.join(args.output_dir, "pre_differentgoal.txt"))
    # save_to_file(golds_samegoal, os.path.join(args.output_dir, "gold_samegoal.json"))
    # save_to_file(golds_differentgoal, os.path.join(args.output_dir, "gold_differentgoal.json"))
    # save_to_file(dialogues_samegoal_unconsistency, os.path.join(args.output_dir, "pre_samegoal_unconsistency.txt"))
    # save_to_file(golds_samegoal_unconsistency, os.path.join(args.output_dir, "gold_samegoal_unconsistency.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--TCP_file", type=str, default="outputs/TCP/best_model_test.txt")
    parser.add_argument("--pre_dialogue_file", type=str, default="outputs/GPT2_w_TCP/output_test.txt")
    parser.add_argument("--gold_file", type=str, default="data/sample_test.json")
    parser.add_argument("--output_dir", type=str, default="data/response_spilt")
    args = parser.parse_args()

    spilt_response(args)
