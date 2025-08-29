import os
import ast
import json
import numpy as np

from collections import Counter
import matplotlib.pyplot as plt


def get_mme_results(results_dir):
    eval_type_dict = {
        "Perception": ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork", "OCR"],
        "Cognition": ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]
    }

    result_list = []

    for eval_type, task_name_list in eval_type_dict.items():
        for task_name in task_name_list:
            task_txt = os.path.join(results_dir, task_name + ".txt")

            lines = open(task_txt, 'r').readlines()
            
            for img_item in lines:
                img_name, question, gt_ans, pred_ans, patch_list = img_item.split("\t")
                
                gt_ans = gt_ans.lower()
                pred_ans = pred_ans.lower()

                pred_label = None
                if pred_ans in ["yes", "no"]:
                    pred_label = pred_ans
                else:
                    prefix_pred_ans = pred_ans[:4]
                    if "yes" in prefix_pred_ans:
                        pred_label = "yes"
                    elif "no" in prefix_pred_ans:
                        pred_label = "no"
                    else:
                        pred_label = "other"
                
                patch_list = ast.literal_eval(patch_list)

                result_list.append((gt_ans, pred_label, patch_list))
    
    return result_list

def analysis_mme(logit_lens_dir, patch_elimination_dir):
    ll_results = get_mme_results(logit_lens_dir)
    pe_results = get_mme_results(patch_elimination_dir)
    assert len(ll_results) == len(pe_results)

    patch_num = 0
    acc_list = []
    overlap = {
        (0,0): [],
        (0,1): [],
        (1,0): [],
        (1,1): [],
    }

    for ll_item, pe_item in zip(ll_results, pe_results):
        ll_gt, ll_pred, ll_patch_list = ll_item
        pe_gt, pe_pred, pe_patch_list = pe_item
        patch_num = len(ll_patch_list)
        overlapped_list = list(set(ll_patch_list) & set(pe_patch_list))
        acc_list.append((ll_gt==ll_pred, pe_gt==pe_pred))
        overlap[(ll_gt==ll_pred, pe_gt==pe_pred)].append(len(overlapped_list))

    counts = Counter(acc_list)
    all_combinations = [(0, 0), (0, 1), (1, 0), (1, 1)]
    combination_num = {pair: counts.get(pair, 0) for pair in all_combinations}
    print(combination_num)

    fig1 = plt.figure()
    plt.xlabel('Value')
    plt.ylabel('Frequency') 
    plt.title(f"MME - {logit_lens_dir.split('/')[-2]} & {patch_elimination_dir.split('/')[-2]}")
    
    for key in overlap.keys():
        counts = Counter(overlap[key])
        x = list(range(patch_num+1))
        y = [counts.get(i, 0) / len(overlap[key]) for i in x]
        plt.plot(x, y, label=str(key))
        plt.legend()
    fig_dir = "./plots/" + logit_lens_dir.split('/')[-2] + "__" + patch_elimination_dir.split('/')[-2] + "_mme.png"
    fig1.savefig(fig_dir, dpi=100)



def get_pope_results(results_dir):
    res_path = os.path.join(results_dir, "answers.json")
    answers = [json.loads(q) for q in open(res_path, 'r')]

    result_list = []
    for ans_item in answers:
        gt = ans_item["label"]
        ans_text = ans_item["answer"]
        patch_list = ans_item["patches"]

        gt = gt.lower()
        ans_text = ans_text.lower()
        ans = "other"

        if ans_text.find('.') != -1:
            ans_text = ans_text.split('.')[0]
        ans_text = ans_text.replace(',', '')
        words = ans_text.split(' ')
        if 'No' in words or 'not' in words or 'no' in words:
            ans = 'no'
        else:
            ans = 'yes'
        
        # patch_list = ast.literal_eval(patch_list)
        result_list.append((gt, ans, patch_list))

    return result_list


def analysis_pope(logit_lens_dir, patch_elimination_dir):
    ll_results = get_pope_results(logit_lens_dir)
    pe_results = get_pope_results(patch_elimination_dir)

    patch_num = 0
    acc_list = []
    overlap = {
        (0,0): [],
        (0,1): [],
        (1,0): [],
        (1,1): [],
    }

    for ll_item, pe_item in zip(ll_results, pe_results):
        ll_gt, ll_pred, ll_patch_list = ll_item
        pe_gt, pe_pred, pe_patch_list = pe_item
        patch_num = len(ll_patch_list)
        overlapped_list = list(set(ll_patch_list) & set(pe_patch_list))
        acc_list.append((ll_gt==ll_pred, pe_gt==pe_pred))
        overlap[(ll_gt==ll_pred, pe_gt==pe_pred)].append(len(overlapped_list))

    counts = Counter(acc_list)
    all_combinations = [(0, 0), (0, 1), (1, 0), (1, 1)]
    combination_num = {pair: counts.get(pair, 0) for pair in all_combinations}
    print(combination_num)

    fig1 = plt.figure()
    plt.xlabel('Value')
    plt.ylabel('Frequency') 
    plt.title(f"POPE - {logit_lens_dir.split('/')[-2]} & {patch_elimination_dir.split('/')[-2]}")
    
    for key in overlap.keys():
        counts = Counter(overlap[key])
        x = list(range(patch_num+1))
        y = [counts.get(i, 0) / len(overlap[key]) for i in x]
        plt.plot(x, y, label=str(key))
        plt.legend()
    fig_dir = "./plots/" + logit_lens_dir.split('/')[-2] + "__" + patch_elimination_dir.split('/')[-2] + "_pope.png"
    fig1.savefig(fig_dir, dpi=100)
    

if __name__ == "__main__":

    # MME
    print("-"*10, "MME", "-"*10)
    # logit_lens_dir = "./results/ppe_1_0_0625/mme"
    # patch_elimination_dir = "./results/ppe_4_0_5/mme"

    # logit_lens_dir = "./results/ppe_1_0_25/mme"
    # patch_elimination_dir = "./results/ppe_2_0_5/mme"

    logit_lens_dir = "./results/ppe_1_0_015625_09/mme"
    patch_elimination_dir = "./results/ppe_6_0_5_t_09/mme"

    analysis_mme(logit_lens_dir, patch_elimination_dir)


    # POPE
    print("-"*10, "POPE", "-"*10)
    # logit_lens_dir = "./results/ppe_1_0_0625/pope"
    # patch_elimination_dir = "./results/ppe_4_0_5/pope"

    # logit_lens_dir = "./results/ppe_1_0_25/pope"
    # patch_elimination_dir = "./results/ppe_2_0_5/pope"

    logit_lens_dir = "./results/ppe_1_0_015625_09/pope"
    patch_elimination_dir = "./results/ppe_6_0_5_t_09/pope"

    analysis_pope(logit_lens_dir, patch_elimination_dir)


    # print("-"*10, "Check overlapped patches", "-"*10)

    # ll_mme_results_1 = get_mme_results("./results/ppe_1_0_0625/mme")
    # ll_mme_results_2 = get_mme_results("./results/ppe_1_0_25/mme")
    # ll_pope_results_1 = get_pope_results("./results/ppe_1_0_0625/pope")
    # ll_pope_results_2 = get_pope_results("./results/ppe_1_0_25/pope")

    # pe_mme_results_1 = get_mme_results("./results/ppe_4_0_5/mme")
    # pe_mme_results_2 = get_mme_results("./results/ppe_2_0_5/mme")
    # pe_pope_results_1 = get_pope_results("./results/ppe_4_0_5/pope")
    # pe_pope_results_2 = get_pope_results("./results/ppe_4_0_5/pope")


    # # for i in range(1):
    # a = []
    # for i in range(len(ll_mme_results_1)):
    #     ll_patch_list_1 = ll_mme_results_1[i][2]
    #     ll_patch_list_2 = ll_mme_results_2[i][2]

    #     pe_patch_list_1 = pe_mme_results_1[i][2]
    #     pe_patch_list_2 = pe_mme_results_2[i][2]

    #     overlapped_list_1 = list(set(ll_patch_list_1) & set(pe_patch_list_1))
    #     overlapped_list_2 = list(set(ll_patch_list_2) & set(pe_patch_list_2))


    #     temp1 = []
    #     for patch in ll_patch_list_1:
    #         if patch not in overlapped_list_1:
    #             temp1.append(patch)
        
    #     temp2 = []
    #     for patch in ll_patch_list_2:
    #         if patch not in overlapped_list_2:
    #             temp2.append(patch)

    #     # print(list(set(temp1) & set(temp2)))
    #     # print(temp1)
    #     # print(temp2)

    #     l1 = list(set(ll_patch_list_1) | set(pe_patch_list_1))
    #     a.append(len(set(l1) & set(overlapped_list_2)) / len(l1))
    #     # print(len(set(l1) & set(overlapped_list_2)) / len(l1))

    # print(f"patches for 36 patch over the overlapped patches in 144: {np.mean(a)}")