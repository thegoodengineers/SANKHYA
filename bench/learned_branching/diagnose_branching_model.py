import sys
import os
import math
import glob
import csv
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import assemble_dataset
from learned_branching_models import CART, spearmanr

def get_stats(data):
    if not data: return {'min':0, 'median':0, 'mean':0, 'p90':0, 'p95':0, 'max':0}
    s = sorted(data)
    n = len(s)
    return {'min': s[0], 'median': s[n//2], 'mean': sum(s)/n, 'p90': s[int(n*0.90)], 'p95': s[int(n*0.95)], 'max': s[-1]}

def get_train_data(train_instances):
    train_data = []
    for inst in train_instances:
        pattern = f"scratch/learned_branching/train/{inst}_seed*.csv"
        for fpath in glob.glob(pattern):
            with open(fpath, 'r', newline='') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 12: continue
                    try:
                        frac, obj, deg, pcd, pcu, md, mu = map(float, row[1:8])
                        if math.isnan(frac) or math.isnan(obj) or math.isnan(deg) or math.isnan(pcd) or math.isnan(pcu) or math.isnan(md) or math.isnan(mu): continue
                        is_unmeasured = md < 0 or mu < 0
                        is_md_inf = math.isinf(md)
                        is_mu_inf = math.isinf(mu)
                        
                        target = -1
                        cat = "unmeasured"
                        if not is_unmeasured:
                            if not is_md_inf and not is_mu_inf:
                                target = max(md, 1e-6) * max(mu, 1e-6)
                                cat = "fin_fin"
                            elif is_md_inf and is_mu_inf:
                                cat = "inf_inf"
                            else:
                                cat = "fin_inf"
                                
                        train_data.append({
                            'instance': inst,
                            'features': [frac, obj, deg, pcd, pcu],
                            'target_raw': target,
                            'category': cat,
                            'measured': not is_unmeasured
                        })
                    except ValueError:
                        pass
    return train_data

def get_test_decision_groups(test_instances):
    all_groups = []
    for inst in test_instances:
        groups = {}
        pattern = f"scratch/learned_branching/full13/test/{inst}_seed*.csv"
        for fpath in glob.glob(pattern):
            seed_str = fpath.split('_seed')[-1].replace('.csv', '')
            try: seed = int(seed_str)
            except: seed = 0
            
            with open(fpath, 'r', newline='') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 13: continue
                    try:
                        frac, obj, deg, pcd, pcu, md, mu = map(float, row[1:8])
                        existing_score = float(row[10])
                        selected = int(row[11])
                        branch_decision_id = int(row[12])
                        
                        key = (inst, seed, branch_decision_id)
                        if key not in groups:
                            groups[key] = []
                        groups[key].append({
                            'features': [frac, obj, deg, pcd, pcu],
                            'md': md,
                            'mu': mu,
                            'existing_score': existing_score,
                            'selected': selected
                        })
                    except ValueError:
                        pass
        all_groups.append((inst, groups))
    return all_groups

def inspect_tree(node, depth, info):
    if node.is_leaf:
        info['leaves'] += 1
        info['leaf_vals'].append(node.value)
        info['depths'].append(depth)
        return
    info['splits'][node.feature_idx] += 1
    inspect_tree(node.left, depth + 1, info)
    inspect_tree(node.right, depth + 1, info)

def get_ranks_desc(arr):
    # Higher score = rank 1
    s = sorted([(val, i) for i, val in enumerate(arr)], key=lambda x: -x[0])
    ranks = [0.0] * len(arr)
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j][0] == s[i][0]:
            j += 1
        avg_rank = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[s[k][1]] = avg_rank + 1
        i = j
    return ranks

def main():
    print("=== DIAGNOSIS: LEARNED BRANCHING MODEL ===")
    
    train_manifest = assemble_dataset.load_manifest("bench/learned_branching/train_instances.txt")
    test_manifest = assemble_dataset.load_manifest("bench/learned_branching/test_instances.txt")
    
    train_data = get_train_data(train_manifest)
    
    # Train the existing model
    train_X_reg = [r['features'] for r in train_data if r['category'] == 'fin_fin']
    train_y_reg = [math.log1p(r['target_raw']) for r in train_data if r['category'] == 'fin_fin']
    
    regressor = CART(max_depth=4, min_samples_leaf=5, mode='regression')
    regressor.fit(train_X_reg, train_y_reg)
    
    print("\n1. PREDICTION SEMANTICS")
    print("The regression model is trained on `math.log1p(target_raw)`.")
    print("When it predicts, it outputs `log1p(target)`. Therefore, it predicts B. log1p(target).")
    
    print("\n2. TREE INSPECTION")
    info = {'leaves': 0, 'splits': defaultdict(int), 'leaf_vals': [], 'depths': []}
    inspect_tree(regressor.tree, 0, info)
    feat_names = ["frac", "obj", "deg", "pcd", "pcu"]
    print(f"Depth max: {max(info['depths'])}")
    print(f"Total leaves: {info['leaves']}")
    print("Split features used:")
    for f_idx, count in info['splits'].items():
        print(f"  {feat_names[f_idx]}: {count} splits")
    print(f"Leaf values range: min={min(info['leaf_vals']):.4f}, max={max(info['leaf_vals']):.4f}")
    
    print("\n3. SELECTION-BIAS CHECK (TRAIN DATA)")
    meas = [r['features'] for r in train_data if r['measured']]
    unmeas = [r['features'] for r in train_data if not r['measured']]
    print(f"Measured candidates: {len(meas)}")
    print(f"Unmeasured candidates: {len(unmeas)}")
    for i, name in enumerate(feat_names):
        m_vals = [f[i] for f in meas]
        u_vals = [f[i] for f in unmeas]
        m_s = get_stats(m_vals)
        u_s = get_stats(u_vals)
        print(f"  {name:4s} | Measured median: {m_s['median']:.4g} | Unmeasured median: {u_s['median']:.4g}")
        
    print("\n4. FEATURE DIAGNOSTIC (TRAIN DATA)")
    print("Checking variation of each feature:")
    for i, name in enumerate(feat_names):
        m_vals = [f[i] for f in meas]
        if m_vals:
            t_s = get_stats(m_vals)
            # basic variance proxy
            q_diff = t_s['p90'] - t_s['min']
            print(f"  {name:4s} | min={t_s['min']:.4g}, median={t_s['median']:.4g}, max={t_s['max']:.4g} | variation (p90-min)={q_diff:.4g}")
            if q_diff == 0:
                print(f"    -> WARNING: {name} has almost zero variation.")
                
    print("\n4.1 INSTANCE DOMINATION (TRAIN DATA)")
    inst_counts = defaultdict(int)
    for r in train_data:
        if r['category'] == 'fin_fin':
            inst_counts[r['instance']] += 1
    total_fin = sum(inst_counts.values())
    for inst, count in inst_counts.items():
        print(f"  {inst}: {count} fin/fin obs ({count/total_fin*100:.1f}%)")
        
    print("\n5. EVALUATING DECISIONS (TEST DATA)")
    test_groups = get_test_decision_groups(test_manifest)
    
    global_trues = []
    global_preds = []
    
    within_spearmans = []
    
    # Top-k metrics
    def evaluate_top_k(scores, true_scores):
        ranks = get_ranks_desc(scores)
        best_true = max(true_scores)
        best_true_indices = [i for i, x in enumerate(true_scores) if x == best_true]
        
        # rank of oracle best (take best rank if multiple oracle bests)
        best_rank = min(ranks[i] for i in best_true_indices)
        
        top1_agree = 1 if best_rank == 1 else 0
        top3_agree = 1 if best_rank <= 3 else 0
        top5_agree = 1 if best_rank <= 5 else 0
        
        return top1_agree, top3_agree, top5_agree, best_rank
        
    c_t1, c_t3, c_t5 = 0, 0, 0
    l_t1, l_t3, l_t5 = 0, 0, 0
    c_ranks = []
    l_ranks = []
    
    wrong_severity = []
    
    log1p_ordering_identical = True
    
    for inst, groups in test_groups:
        for key, candidates in groups.items():
            
            # Check eligibility
            def has_finite_measurement(c):
                if math.isnan(c['md']) or math.isnan(c['mu']) or c['md'] < 0 or c['mu'] < 0: return False
                if math.isinf(c['md']) or math.isinf(c['mu']): return False
                return True
                
            finite_cands = [c for c in candidates if has_finite_measurement(c)]
            if not finite_cands: continue
            
            # Learn scores
            for c in finite_cands:
                pred_log1p = regressor.predict([c['features']])[0]
                c['learned_score'] = pred_log1p
                c['learned_raw'] = math.expm1(pred_log1p)
                c['oracle_score'] = max(c['md'], 1e-6) * max(c['mu'], 1e-6)
                
            oracle_scores = [c['oracle_score'] for c in finite_cands]
            curr_scores = [c['existing_score'] for c in finite_cands]
            learned_scores = [c['learned_score'] for c in finite_cands]
            learned_raw = [c['learned_raw'] for c in finite_cands]
            
            # Check log-transform ordering
            rank_log = get_ranks_desc(learned_scores)
            rank_raw = get_ranks_desc(learned_raw)
            if rank_log != rank_raw:
                log1p_ordering_identical = False
                
            global_trues.extend(oracle_scores)
            global_preds.extend(learned_scores)
            
            if len(oracle_scores) > 1:
                sp = spearmanr(oracle_scores, learned_scores)
                within_spearmans.append(sp)
                
            c_a1, c_a3, c_a5, c_rk = evaluate_top_k(curr_scores, oracle_scores)
            l_a1, l_a3, l_a5, l_rk = evaluate_top_k(learned_scores, oracle_scores)
            
            c_t1 += c_a1; c_t3 += c_a3; c_t5 += c_a5
            l_t1 += l_a1; l_t3 += l_a3; l_t5 += l_a5
            c_ranks.append(c_rk)
            l_ranks.append(l_rk)
            
            # Wrong severity for learned model
            if l_a1 == 0:
                learned_selected_cands = [c for c, r in zip(finite_cands, rank_log) if r == 1]
                if learned_selected_cands:
                    ls = learned_selected_cands[0]
                    oracle_best = max(oracle_scores)
                    ratio = ls['oracle_score'] / oracle_best if oracle_best > 1e-9 else 0
                    wrong_severity.append(ratio)
                    
    print("\n6. WITHIN-NODE VS GLOBAL RANKING")
    global_sp = spearmanr(global_trues, global_preds) if len(global_trues) > 1 else 0
    print(f"Global Spearman: {global_sp:.4f}")
    if within_spearmans:
        s_stats = get_stats(within_spearmans)
        # Using p10 is hard to calculate properly with get_stats since it only has p90
        ws = sorted(within_spearmans)
        p10 = ws[int(len(ws)*0.10)]
        print(f"Within-node Spearman: mean={s_stats['mean']:.4f}, median={s_stats['median']:.4f}, p10={p10:.4f}, p90={s_stats['p90']:.4f}")
        
    print("\n7. TOP-K / RANK QUALITY (over eligible decisions)")
    N = len(c_ranks)
    print(f"Total eligible for ranking: {N}")
    if N > 0:
        print(f"Current top-1: {c_t1/N*100:.1f}%, top-3: {c_t3/N*100:.1f}%, top-5: {c_t5/N*100:.1f}%")
        print(f"Learned top-1: {l_t1/N*100:.1f}%, top-3: {l_t3/N*100:.1f}%, top-5: {l_t5/N*100:.1f}%")
        
        def rr(ranks): return sum(1.0/r for r in ranks)/N if N else 0
        print(f"Current MRR: {rr(c_ranks):.4f}, median rank: {get_stats(c_ranks)['median']}, p90: {get_stats(c_ranks)['p90']}")
        print(f"Learned MRR: {rr(l_ranks):.4f}, median rank: {get_stats(l_ranks)['median']}, p90: {get_stats(l_ranks)['p90']}")
        
    print("\n8. WRONG-SELECTION SEVERITY (Learned)")
    total_wrong = len(wrong_severity)
    print(f"Total wrong decisions: {total_wrong}")
    if total_wrong > 0:
        b_exactly1 = sum(1 for r in wrong_severity if r > 0.9999)
        b_90 = sum(1 for r in wrong_severity if 0.9 <= r <= 0.9999)
        b_75 = sum(1 for r in wrong_severity if 0.75 <= r < 0.9)
        b_50 = sum(1 for r in wrong_severity if 0.5 <= r < 0.75)
        b_less = sum(1 for r in wrong_severity if r < 0.5)
        
        print(f"  exactly 1.0 (ties): {b_exactly1} ({b_exactly1/total_wrong*100:.1f}%)")
        print(f"  >= 0.90 (close misses): {b_90} ({b_90/total_wrong*100:.1f}%)")
        print(f"  >= 0.75 (moderate misses): {b_75} ({b_75/total_wrong*100:.1f}%)")
        print(f"  >= 0.50 (bad misses): {b_50} ({b_50/total_wrong*100:.1f}%)")
        print(f"  < 0.50 (catastrophic misses): {b_less} ({b_less/total_wrong*100:.1f}%)")
        
    print("\n9. LOG-TRANSFORM INVESTIGATION")
    if log1p_ordering_identical:
        print("A (log1p prediction) and B (expm1 prediction) produce IDENTICAL rankings.")
        print("The log1p transform itself CANNOT explain ranking differences because inverse transformation is monotonic.")
    else:
        print("Rankings differ?! This shouldn't happen with expm1.")

if __name__ == '__main__':
    main()
