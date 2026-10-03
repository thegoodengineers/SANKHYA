import math
import os
import sys
import glob
import csv
from collections import defaultdict

# Import assembly module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import assemble_dataset
from learned_branching_models import CART, spearmanr, get_ranks

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
                        if md < 0 or mu < 0: continue
                        is_md_inf = math.isinf(md)
                        is_mu_inf = math.isinf(mu)
                        if not is_md_inf and not is_mu_inf:
                            t_raw = max(md, 1e-6) * max(mu, 1e-6)
                            train_data.append({
                                'features_v1': [frac, obj, deg, pcd, pcu],
                                'features_v2': [frac, obj, deg],
                                'target_raw': t_raw,
                                'category': 'fin_fin',
                                'instance': inst
                            })
                    except ValueError:
                        pass
    return train_data

def get_test_decision_groups(test_instances):
    all_groups = []
    test_rows_finfin = []
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
                        branch_decision_id = int(row[12])
                        if math.isnan(frac) or math.isnan(obj) or math.isnan(deg) or math.isnan(md) or math.isnan(mu): continue
                        
                        is_md_inf = math.isinf(md)
                        is_mu_inf = math.isinf(mu)
                        if not is_md_inf and not is_mu_inf and md >= 0 and mu >= 0:
                            test_rows_finfin.append({
                                'features_v1': [frac, obj, deg, pcd, pcu],
                                'features_v2': [frac, obj, deg],
                                'target_raw': max(md, 1e-6) * max(mu, 1e-6),
                                'instance': inst
                            })
                        
                        key = (inst, seed, branch_decision_id)
                        if key not in groups:
                            groups[key] = []
                        groups[key].append({
                            'features_v1': [frac, obj, deg, pcd, pcu],
                            'features_v2': [frac, obj, deg],
                            'md': md,
                            'mu': mu
                        })
                    except ValueError:
                        pass
        all_groups.append((inst, groups))
    return all_groups, test_rows_finfin

def get_ranks_desc(arr):
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

def generate_cpp_v2(node, indent="    "):
    if node.is_leaf:
        return f"{indent}return {node.value};\n"
    
    features = ["frac", "obj", "deg"]
    f_name = features[node.feature_idx]
    
    code = f"{indent}if ({f_name} <= {node.threshold}) {{\n"
    code += generate_cpp_v2(node.left, indent + "    ")
    code += f"{indent}}} else {{\n"
    code += generate_cpp_v2(node.right, indent + "    ")
    code += f"{indent}}}\n"
    return code

def inspect_tree(node, depth, info, feature_names):
    if node.is_leaf:
        info['leaves'] += 1
        info['leaf_vals'].append(node.value)
        info['depths'].append(depth)
        return
    info['splits'][feature_names[node.feature_idx]] += 1
    inspect_tree(node.left, depth + 1, info, feature_names)
    inspect_tree(node.right, depth + 1, info, feature_names)

def main():
    print("=== MODEL V2 EXPERIMENT ===")
    
    train_manifest = assemble_dataset.load_manifest("bench/learned_branching/train_instances.txt")
    test_manifest = assemble_dataset.load_manifest("bench/learned_branching/test_instances.txt")
    
    train_data = get_train_data(train_manifest)
    test_groups, test_rows_finfin = get_test_decision_groups(test_manifest)
    
    print(f"Loaded {len(train_data)} TRAIN fin/fin observations.")
    
    # Train V1 (5 features)
    X1 = [r['features_v1'] for r in train_data]
    y = [math.log1p(r['target_raw']) for r in train_data]
    reg_v1 = CART(max_depth=4, min_samples_leaf=5, mode='regression')
    reg_v1.fit(X1, y)
    
    # Train V2 (3 features)
    X2 = [r['features_v2'] for r in train_data]
    reg_v2 = CART(max_depth=4, min_samples_leaf=5, mode='regression')
    reg_v2.fit(X2, y)
    
    # Output V2 C++
    os.makedirs("scratch/generated", exist_ok=True)
    out_path = "scratch/generated/learned_branching_v2.cpp"
    with open(out_path, "w") as f:
        f.write("// GENERATED BY bench/learned_branching/learned_branching_models_v2.py\n")
        f.write("// DO NOT EDIT DIRECTLY\n\n")
        f.write("double ScoreBranchingCandidateV2(float frac, double obj, int deg) {\n")
        f.write(generate_cpp_v2(reg_v2.tree, "    "))
        f.write("}\n")
    print(f"Generated V2 C++ saved to {out_path}")
    
    # Tree inspection V2
    info2 = {'leaves': 0, 'splits': defaultdict(int), 'leaf_vals': [], 'depths': []}
    inspect_tree(reg_v2.tree, 0, info2, ["frac", "obj", "deg"])
    print("\n=== V2 TREE STRUCTURE ===")
    print(f"Depth max: {max(info2['depths']) if info2['depths'] else 0}")
    print(f"Total leaves: {info2['leaves']}")
    print("Split features used:")
    for f, count in info2['splits'].items():
        print(f"  {f}: {count} splits")
    if info2['leaf_vals']:
        print(f"Leaf values range: min={min(info2['leaf_vals']):.4f}, max={max(info2['leaf_vals']):.4f}")
    
    # Regression metrics
    test_y_true = [math.log1p(r['target_raw']) for r in test_rows_finfin]
    test_y_true_raw = [r['target_raw'] for r in test_rows_finfin]
    
    if len(test_y_true) > 0:
        pred_v1_log = reg_v1.predict([r['features_v1'] for r in test_rows_finfin])
        pred_v1_raw = [math.expm1(p) for p in pred_v1_log]
        pred_v2_log = reg_v2.predict([r['features_v2'] for r in test_rows_finfin])
        pred_v2_raw = [math.expm1(p) for p in pred_v2_log]
        
        def calc_mae_rmse(y_t, y_p):
            mae = sum(abs(t - p) for t, p in zip(y_t, y_p)) / len(y_t)
            rmse = math.sqrt(sum((t - p)**2 for t, p in zip(y_t, y_p)) / len(y_t))
            return mae, rmse
            
        m_log1, rm_log1 = calc_mae_rmse(test_y_true, pred_v1_log)
        m_log2, rm_log2 = calc_mae_rmse(test_y_true, pred_v2_log)
        m_raw1, rm_raw1 = calc_mae_rmse(test_y_true_raw, pred_v1_raw)
        m_raw2, rm_raw2 = calc_mae_rmse(test_y_true_raw, pred_v2_raw)
        
        sp1 = spearmanr(test_y_true_raw, pred_v1_raw)
        sp2 = spearmanr(test_y_true_raw, pred_v2_raw)
        
        print("\n=== GLOBAL REGRESSION METRICS (TEST fin/fin) ===")
        print(f"N = {len(test_y_true)}")
        print(f"V1 Global Spearman: {sp1:.4f}")
        print(f"V2 Global Spearman: {sp2:.4f}")
        print(f"V1 MAE (log1p): {m_log1:.4f}, RMSE (log1p): {rm_log1:.4f}, MAE (raw): {m_raw1:.4f}, RMSE (raw): {rm_raw1:.4f}")
        print(f"V2 MAE (log1p): {m_log2:.4f}, RMSE (log1p): {rm_log2:.4f}, MAE (raw): {m_raw2:.4f}, RMSE (raw): {rm_raw2:.4f}")
        
    # Node-level policy
    v1_within_sps = []
    v2_within_sps = []
    
    v1_t1, v1_t3, v1_t5 = 0, 0, 0
    v2_t1, v2_t3, v2_t5 = 0, 0, 0
    v1_ranks, v2_ranks = [], []
    v1_regrets, v2_regrets = [], []
    v1_norms, v2_norms = [], []
    
    v1_wrong_severity = []
    v2_wrong_severity = []
    
    for inst, groups in test_groups:
        for key, candidates in groups.items():
            def has_finite_measurement(c):
                if math.isnan(c['md']) or math.isnan(c['mu']) or c['md'] < 0 or c['mu'] < 0: return False
                if math.isinf(c['md']) or math.isinf(c['mu']): return False
                return True
                
            finite_cands = [c for c in candidates if has_finite_measurement(c)]
            if not finite_cands: continue
            
            oracle_scores = []
            v1_scores = []
            v2_scores = []
            
            for c in finite_cands:
                osc = max(c['md'], 1e-6) * max(c['mu'], 1e-6)
                oracle_scores.append(osc)
                v1_scores.append(reg_v1.predict([c['features_v1']])[0])
                v2_scores.append(reg_v2.predict([c['features_v2']])[0])
                
            if len(oracle_scores) > 1:
                v1_within_sps.append(spearmanr(oracle_scores, v1_scores))
                v2_within_sps.append(spearmanr(oracle_scores, v2_scores))
                
            def evaluate_top_k(scores, true_scores):
                ranks = get_ranks_desc(scores)
                best_true = max(true_scores)
                best_true_indices = [i for i, x in enumerate(true_scores) if x == best_true]
                best_rank = min(ranks[i] for i in best_true_indices)
                
                t1 = 1 if best_rank == 1 else 0
                t3 = 1 if best_rank <= 3 else 0
                t5 = 1 if best_rank <= 5 else 0
                
                selected_idx = [i for i, r in enumerate(ranks) if r == 1]
                sel_oracle = true_scores[selected_idx[0]] if selected_idx else 0
                
                return t1, t3, t5, best_rank, sel_oracle, best_true
                
            t1_1, t3_1, t5_1, rk_1, sel_1, obest = evaluate_top_k(v1_scores, oracle_scores)
            t1_2, t3_2, t5_2, rk_2, sel_2, obest = evaluate_top_k(v2_scores, oracle_scores)
            
            v1_t1 += t1_1; v1_t3 += t3_1; v1_t5 += t5_1; v1_ranks.append(rk_1)
            v2_t1 += t1_2; v2_t3 += t3_2; v2_t5 += t5_2; v2_ranks.append(rk_2)
            
            v1_regrets.append(obest - sel_1)
            v2_regrets.append(obest - sel_2)
            
            if obest > 1e-9:
                v1_norms.append(sel_1 / obest)
                v2_norms.append(sel_2 / obest)
                if t1_1 == 0: v1_wrong_severity.append(sel_1 / obest)
                if t1_2 == 0: v2_wrong_severity.append(sel_2 / obest)
                
    N = len(v1_ranks)
    print("\n=== NODE-LEVEL POLICY EVALUATION ===")
    print(f"Total eligible decisions: {N}")
    if N > 0:
        s1 = get_stats(v1_within_sps)
        s2 = get_stats(v2_within_sps)
        print(f"V1 Within-Node Spearman: mean={s1['mean']:.4f}, median={s1['median']:.4f}")
        print(f"V2 Within-Node Spearman: mean={s2['mean']:.4f}, median={s2['median']:.4f}")
        
        print("\nTop-K Agreement:")
        print(f"V1 top-1: {v1_t1/N*100:.1f}%, top-3: {v1_t3/N*100:.1f}%, top-5: {v1_t5/N*100:.1f}%")
        print(f"V2 top-1: {v2_t1/N*100:.1f}%, top-3: {v2_t3/N*100:.1f}%, top-5: {v2_t5/N*100:.1f}%")
        
        def rr(ranks): return sum(1.0/r for r in ranks)/N if N else 0
        r1 = get_stats(v1_ranks)
        r2 = get_stats(v2_ranks)
        print("\nOracle-Best Rank Stats:")
        print(f"V1 MRR: {rr(v1_ranks):.4f}, median rank: {r1['median']}, p90: {r1['p90']}")
        print(f"V2 MRR: {rr(v2_ranks):.4f}, median rank: {r2['median']}, p90: {r2['p90']}")
        
        n1 = get_stats(v1_norms)
        n2 = get_stats(v2_norms)
        rg1 = get_stats(v1_regrets)
        rg2 = get_stats(v2_regrets)
        print("\nRegret Metrics:")
        print(f"V1 Median Normalized Score: {n1['median']:.4f}")
        print(f"V2 Median Normalized Score: {n2['median']:.4f}")
        print(f"V1 Median Regret: {rg1['median']:.4f}")
        print(f"V2 Median Regret: {rg2['median']:.4f}")
        
    print("\n=== WRONG-SELECTION SEVERITY COMPARISON ===")
    def count_buckets(arr):
        tot = len(arr)
        if tot == 0: return [0]*5
        b1 = sum(1 for r in arr if r > 0.9999)
        b90 = sum(1 for r in arr if 0.9 <= r <= 0.9999)
        b75 = sum(1 for r in arr if 0.75 <= r < 0.9)
        b50 = sum(1 for r in arr if 0.5 <= r < 0.75)
        bless = sum(1 for r in arr if r < 0.5)
        return [b1, b90, b75, b50, bless]
        
    def print_buckets(name, arr):
        tot = len(arr)
        print(f"{name} total wrong decisions: {tot}")
        if tot > 0:
            b = count_buckets(arr)
            print(f"  exactly 1.0 (ties):         {b[0]:4d} ({b[0]/tot*100:.1f}%)")
            print(f"  >= 0.90 (close misses):     {b[1]:4d} ({b[1]/tot*100:.1f}%)")
            print(f"  >= 0.75 (moderate misses):  {b[2]:4d} ({b[2]/tot*100:.1f}%)")
            print(f"  >= 0.50 (bad misses):       {b[3]:4d} ({b[3]/tot*100:.1f}%)")
            print(f"  < 0.50 (catastrophic):      {b[4]:4d} ({b[4]/tot*100:.1f}%)")
            
    print_buckets("V1", v1_wrong_severity)
    print_buckets("V2", v2_wrong_severity)
    
    print("\n=== FINAL DECISION ===")
    if N > 0 and (v2_t1/N > 0.5 or n2['median'] > 0.8):
        print("V2 VALIDATED")
    else:
        print("RANKING OBJECTIVE REQUIRED")

if __name__ == '__main__':
    main()
