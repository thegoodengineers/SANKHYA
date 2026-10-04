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

def get_train_data_old(train_instances):
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
                                'target_raw': t_raw
                            })
                    except ValueError:
                        pass
    return train_data

def get_decision_groups(instances, split):
    all_groups = []
    for inst in instances:
        groups = {}
        pattern = f"scratch/learned_branching/full13/{split}/{inst}_seed*.csv"
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
                            t_raw = max(md, 1e-6) * max(mu, 1e-6)
                            key = (inst, seed, branch_decision_id)
                            if key not in groups:
                                groups[key] = []
                            groups[key].append({
                                'features_v1': [frac, obj, deg, pcd, pcu],
                                'features_v2': [frac, obj, deg],
                                'features_v3': [frac, obj, deg, pcd, pcu],
                                'target': t_raw,
                                'md': md,
                                'mu': mu
                            })
                    except ValueError:
                        pass
        all_groups.append((inst, groups))
    return all_groups

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

def train_ranking_model(groups_train):
    X_diff = []
    n_decisions = 0
    for inst, groups in groups_train:
        n_decisions += len(groups)
        for key, cands in groups.items():
            n = len(cands)
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    t_i = cands[i]['target']
                    t_j = cands[j]['target']
                    if t_i > t_j * (1 + 1e-9):
                        diff = [cands[i]['features_v3'][k] - cands[j]['features_v3'][k] for k in range(5)]
                        X_diff.append(diff)
                        
    n_pairs = len(X_diff)
    print(f"Number of training decisions: {n_decisions}")
    print(f"Number of pairwise training examples: {n_pairs}")
    
    w = [0.0] * 5
    lr = 0.1
    epochs = 100
    
    if n_pairs > 0:
        for ep in range(epochs):
            grad = [0.0] * 5
            for diff in X_diff:
                z = sum(w[k] * diff[k] for k in range(5))
                z = max(min(z, 20.0), -20.0)
                sig = 1.0 / (1.0 + math.exp(-z))
                err = sig - 1.0
                for k in range(5):
                    grad[k] += err * diff[k]
            
            for k in range(5):
                w[k] -= lr * (grad[k] / n_pairs)
                
    return w

def generate_cpp_v3(w):
    code = f"double ScoreBranchingCandidateV3(float frac, double obj, int deg, double pcd, double pcu) {{\n"
    code += f"    double score = {w[0]:.6f} * frac + {w[1]:.6f} * obj + {w[2]:.6f} * deg + {w[3]:.6f} * pcd + {w[4]:.6f} * pcu;\n"
    code += "    return score;\n"
    code += "}\n"
    return code

def main():
    print("=== MODEL V3 EXPERIMENT (PAIRWISE RANKING) ===")
    
    train_manifest = assemble_dataset.load_manifest("bench/learned_branching/train_instances.txt")
    test_manifest = assemble_dataset.load_manifest("bench/learned_branching/test_instances.txt")
    
    # Train V1 and V2 for baseline comparisons
    train_data_old = get_train_data_old(train_manifest)
    X1 = [r['features_v1'] for r in train_data_old]
    X2 = [r['features_v2'] for r in train_data_old]
    y_old = [math.log1p(r['target_raw']) for r in train_data_old]
    
    reg_v1 = CART(max_depth=4, min_samples_leaf=5, mode='regression')
    reg_v1.fit(X1, y_old)
    reg_v2 = CART(max_depth=4, min_samples_leaf=5, mode='regression')
    reg_v2.fit(X2, y_old)
    
    # Train V3
    print("\nTraining V3...")
    groups_train = get_decision_groups(train_manifest, "train")
    w3 = train_ranking_model(groups_train)
    print(f"Learned weights (frac, obj, deg, pcd, pcu): {[round(x, 6) for x in w3]}")
    
    os.makedirs("scratch/generated", exist_ok=True)
    out_path = "scratch/generated/learned_branching_v3.cpp"
    with open(out_path, "w") as f:
        f.write("// GENERATED BY bench/learned_branching/learned_branching_models_v3.py\n")
        f.write("// DO NOT EDIT DIRECTLY\n\n")
        f.write(generate_cpp_v3(w3))
    print(f"Generated V3 C++ saved to {out_path}")
    
    print("\n=== EVALUATING ON NEW 13-COL TEST DATA ===")
    groups_test = get_decision_groups(test_manifest, "test")
    
    metrics = {
        'v1': {'sps': [], 'ranks': [], 'norms': [], 'regrets': [], 't1':0, 't3':0, 't5':0, 'wrong_sev':[]},
        'v2': {'sps': [], 'ranks': [], 'norms': [], 'regrets': [], 't1':0, 't3':0, 't5':0, 'wrong_sev':[]},
        'v3': {'sps': [], 'ranks': [], 'norms': [], 'regrets': [], 't1':0, 't3':0, 't5':0, 'wrong_sev':[]}
    }
    
    total_eligible = 0
    inst_metrics = defaultdict(lambda: {'n': 0, 'v1_t1':0, 'v2_t1':0, 'v3_t1':0})
    
    for inst, groups in groups_test:
        for key, cands in groups.items():
            if not cands: continue
            
            oracle_scores = []
            s1, s2, s3 = [], [], []
            
            for c in cands:
                osc = c['target']
                oracle_scores.append(osc)
                s1.append(reg_v1.predict([c['features_v1']])[0])
                s2.append(reg_v2.predict([c['features_v2']])[0])
                score_v3 = sum(w3[k] * c['features_v3'][k] for k in range(5))
                s3.append(score_v3)
                
            if len(oracle_scores) > 1:
                metrics['v1']['sps'].append(spearmanr(oracle_scores, s1))
                metrics['v2']['sps'].append(spearmanr(oracle_scores, s2))
                metrics['v3']['sps'].append(spearmanr(oracle_scores, s3))
                
            def evaluate_top_k(scores, true_scores, m_dict, inst_key):
                ranks = get_ranks_desc(scores)
                best_true = max(true_scores)
                best_true_indices = [i for i, x in enumerate(true_scores) if x == best_true]
                best_rank = min(ranks[i] for i in best_true_indices)
                
                t1 = 1 if best_rank == 1 else 0
                m_dict['t1'] += t1
                m_dict['t3'] += 1 if best_rank <= 3 else 0
                m_dict['t5'] += 1 if best_rank <= 5 else 0
                m_dict['ranks'].append(best_rank)
                inst_metrics[inst][inst_key] += t1
                
                selected_idx = [i for i, r in enumerate(ranks) if r == 1]
                sel_oracle = true_scores[selected_idx[0]] if selected_idx else 0
                m_dict['regrets'].append(best_true - sel_oracle)
                
                if best_true > 1e-9:
                    m_dict['norms'].append(sel_oracle / best_true)
                    if t1 == 0: m_dict['wrong_sev'].append(sel_oracle / best_true)
                    
            evaluate_top_k(s1, oracle_scores, metrics['v1'], 'v1_t1')
            evaluate_top_k(s2, oracle_scores, metrics['v2'], 'v2_t1')
            evaluate_top_k(s3, oracle_scores, metrics['v3'], 'v3_t1')
            
            total_eligible += 1
            inst_metrics[inst]['n'] += 1
            
    print(f"Number of eligible TEST decisions: {total_eligible}")
    
    if total_eligible > 0:
        for ver, v in [('V1', 'v1'), ('V2', 'v2'), ('V3', 'v3')]:
            m = metrics[v]
            print(f"\n--- {ver} ---")
            sp = get_stats(m['sps'])
            print(f"Within-node Spearman: mean={sp['mean']:.4f}, median={sp['median']:.4f}")
            print(f"Top-1: {m['t1']/total_eligible*100:.1f}%, Top-3: {m['t3']/total_eligible*100:.1f}%, Top-5: {m['t5']/total_eligible*100:.1f}%")
            
            rr = sum(1.0/r for r in m['ranks'])/total_eligible
            rs = get_stats(m['ranks'])
            print(f"MRR: {rr:.4f}, median oracle rank: {rs['median']}, p90 oracle rank: {rs['p90']}")
            
            nrm = get_stats(m['norms'])
            rg = get_stats(m['regrets'])
            print(f"Median normalized oracle score: {nrm['median']:.4f}")
            print(f"Oracle regret: mean={rg['mean']:.4f}, median={rg['median']:.4f}")
            
            b90 = sum(1 for r in m['wrong_sev'] if 0.9 <= r)
            b75 = sum(1 for r in m['wrong_sev'] if 0.75 <= r < 0.9)
            b50 = sum(1 for r in m['wrong_sev'] if 0.5 <= r < 0.75)
            bless = sum(1 for r in m['wrong_sev'] if r < 0.5)
            print("Wrong-selection severity distribution:")
            print(f"  >= 0.90: {b90}, 0.75-0.90: {b75}, 0.50-0.75: {b50}, < 0.50: {bless}")
            
        print("\n--- PER-INSTANCE TEST METRICS ---")
        for inst, m in inst_metrics.items():
            n = m['n']
            if n > 0:
                print(f"{inst:15s} (n={n}): V1 top-1={m['v1_t1']/n*100:5.1f}% | V2 top-1={m['v2_t1']/n*100:5.1f}% | V3 top-1={m['v3_t1']/n*100:5.1f}%")

    print("\n=== FINAL DECISION ===")
    v3_t1_pct = metrics['v3']['t1'] / total_eligible if total_eligible > 0 else 0
    v1_t1_pct = metrics['v1']['t1'] / total_eligible if total_eligible > 0 else 0
    if total_eligible >= 30 and v3_t1_pct > v1_t1_pct + 0.15:
        print("RANKING OBJECTIVE SHOWS PROMISE")
    elif total_eligible < 30:
        print("RANKING OBJECTIVE INCONCLUSIVE")
    else:
        print("RANKING OBJECTIVE FAILS")

if __name__ == '__main__':
    main()
