import json
import sys

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

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <param1> <param2>")
        sys.exit(1)
    fp = sys.argv[1]
    model = sys.argv[2]

    pre_stage2 = read_data(fp)
    test_stage3 = read_data("data/sample_test.json")
    assert len(pre_stage2) == len(test_stage3)
    for i in range(len(pre_stage2)):
        pre_rsp_Q = pre_stage2[i]["response"].lower()
        if "###stage2" not in pre_rsp_Q:
            list_pre_stage2 = pre_rsp_Q.split("###rq:")
        if "###stage2" in pre_rsp_Q and "##rq:" in pre_rsp_Q:
            pre_rsp_Q = pre_rsp_Q.replace("###stage2", '')
            list_pre_stage2 = pre_rsp_Q.split("##rq:")
        if "###stage2" in pre_rsp_Q and "##rq:" not in pre_rsp_Q:
            list_pre_stage2 = pre_rsp_Q.split("###stage2:")


        test_stage3[i]["pred_response"] = list_pre_stage2[0]
        if len(list_pre_stage2) > 1:
            test_stage3[i]["response_quality"] = list_pre_stage2[1]

    op = "data/sample_test_" + model + "_stage3.json"
    save_to_file(test_stage3, op)