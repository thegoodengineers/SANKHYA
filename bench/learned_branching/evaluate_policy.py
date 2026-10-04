import sys
import os
import math
import glob
import csv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import assemble_dataset
from learned_branching_models import CART

def get_stats(data):
    if not data: return {'min':0, 'median':0, 'mean':0, 'p90':0, 'p95':0, 'max':0}
    s = sorted(data)
    n = len(s)
    return {'min': s[0], 'median': s[n//2], 'mean': sum(s)/n, 'p90': s[int(n*0.90)], 'p95': s[int(n*0.95)], 'max': s[-1]}

def get_train_data(train_instances):
    # Load ONLY from the old 12-column train path to train existing model
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
                            train_data.append({'features': [frac, obj, deg, pcd, pcu], 'target_raw': t_raw, 'category': 'fin_fin'})
                        elif is_md_inf and is_mu_inf:
                            train_data.append({'features': [frac, obj, deg, pcd, pcu], 'category': 'inf_inf'})
                        else:
                            train_data.append({'features': [frac, obj, deg, pcd, pcu], 'category': 'fin_inf'})
                    except ValueError:
                        pass
    return train_data

def get_test_decision_groups(test_instances):
    # Load ONLY from the new 13-column test path
    all_groups = []
    all_rows = []
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
                        
                        all_rows.append({'category': 'fin_fin' if (not math.isinf(md) and not math.isinf(mu) and md>=0 and mu>=0) else 'fin_inf', 'features': [frac, obj, deg, pcd, pcu]})
                        
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
    return all_groups, all_rows

def main():
    train_manifest = assemble_dataset.load_manifest("bench/learned_branching/train_instances.txt")
    test_manifest = assemble_dataset.load_manifest("bench/learned_branching/test_instances.txt")
    
    print("=== TRAINING EXISTING MODEL (ON OLD 12-COL DATA) ===")
    train_data = get_train_data(train_manifest)
    
    train_X_reg, train_y_reg = [], []
    train_X_cls, train_y_cls = [], []
    for row in train_data:
        if row['category'] == 'fin_fin':
            train_X_reg.append(row['features'])
            train_y_reg.append(math.log1p(row['target_raw']))
            train_X_cls.append(row['features'])
            train_y_cls.append(0)
        elif row['category'] == 'fin_inf':
            train_X_cls.append(row['features'])
            train_y_cls.append(1)
            
    regressor = CART(max_depth=4, min_samples_leaf=5, mode='regression')
    regressor.fit(train_X_reg, train_y_reg)
    
    classifier = CART(max_depth=4, min_samples_leaf=5, mode='classification')
    if len(set(train_y_cls)) > 1:
        classifier.fit(train_X_cls, train_y_cls)
    
    print("\n=== EVALUATING ON NEW 13-COL TEST DATA ===")
    test_groups, test_all_rows = get_test_decision_groups(test_manifest)
    
    total_eligible = 0
    total_excluded = 0
    total_decisions = 0
    
    curr_top1_agreements = 0
    learned_top1_agreements = 0
    
    curr_regrets = []
    learned_regrets = []
    
    curr_normalized = []
    learned_normalized = []
    
    for inst, groups in test_groups:
        inst_decisions = len(groups)
        inst_eligible = 0
        inst_excluded = 0
        total_decisions += inst_decisions
        
        inst_curr_agreements = 0
        inst_learned_agreements = 0
        inst_curr_regrets = []
        inst_learned_regrets = []
        inst_curr_norm = []
        inst_learned_norm = []
        inst_cand_counts = []
        
        for key, candidates in groups.items():
            inst_cand_counts.append(len(candidates))
            
            # Verify constraints
            assert len(candidates) >= 1
            selected_cands = [c for c in candidates if c['selected'] == 1]
            assert len(selected_cands) == 1
            
            # Current selected
            curr_selected = selected_cands[0]
            
            # Learned selected
            for c in candidates:
                pred_log1p = regressor.predict([c['features']])[0]
                c['learned_score'] = pred_log1p
                
            learned_selected = max(candidates, key=lambda x: x['learned_score'])
            
            # Check if eligible
            def has_finite_measurement(c):
                if math.isnan(c['md']) or math.isnan(c['mu']) or c['md'] < 0 or c['mu'] < 0:
                    return False
                if math.isinf(c['md']) or math.isinf(c['mu']):
                    return False
                return True
                
            if not has_finite_measurement(curr_selected) or not has_finite_measurement(learned_selected):
                inst_excluded += 1
                total_excluded += 1
                continue
                
            # Filter finite measured candidates to find oracle_best
            finite_cands = [c for c in candidates if has_finite_measurement(c)]
            if not finite_cands:
                inst_excluded += 1
                total_excluded += 1
                continue
                
            inst_eligible += 1
            total_eligible += 1
            
            # Oracle scores
            for c in finite_cands:
                c['oracle_score'] = max(c['md'], 1e-6) * max(c['mu'], 1e-6)
                
            oracle_best = max(c['oracle_score'] for c in finite_cands)
            curr_selected_oracle = curr_selected['oracle_score']
            learned_selected_oracle = learned_selected['oracle_score']
            
            # Metrics
            if abs(curr_selected['oracle_score'] - oracle_best) < 1e-9:
                inst_curr_agreements += 1
                curr_top1_agreements += 1
            if abs(learned_selected['oracle_score'] - oracle_best) < 1e-9:
                inst_learned_agreements += 1
                learned_top1_agreements += 1
                
            curr_regret = oracle_best - curr_selected_oracle
            learned_regret = oracle_best - learned_selected_oracle
            
            inst_curr_regrets.append(curr_regret)
            inst_learned_regrets.append(learned_regret)
            curr_regrets.append(curr_regret)
            learned_regrets.append(learned_regret)
            
            if oracle_best > 1e-9:
                curr_norm = curr_selected_oracle / oracle_best
                learned_norm = learned_selected_oracle / oracle_best
                inst_curr_norm.append(curr_norm)
                inst_learned_norm.append(learned_norm)
                curr_normalized.append(curr_norm)
                learned_normalized.append(learned_norm)
                
        print(f"\n--- {inst} ---")
        print(f"Total decisions: {inst_decisions}")
        print(f"Eligible decisions: {inst_eligible}")
        print(f"Excluded decisions: {inst_excluded}")
        if inst_cand_counts:
            print(f"Candidate counts - min: {min(inst_cand_counts)}, median: {sorted(inst_cand_counts)[len(inst_cand_counts)//2]}, max: {max(inst_cand_counts)}")
        if inst_eligible > 0:
            print(f"Current top-1 agreement: {inst_curr_agreements} / {inst_eligible} ({inst_curr_agreements/inst_eligible*100:.1f}%)")
            print(f"Learned top-1 agreement: {inst_learned_agreements} / {inst_eligible} ({inst_learned_agreements/inst_eligible*100:.1f}%)")
            if inst_curr_norm:
                print(f"Current median normalized score: {get_stats(inst_curr_norm)['median']:.4f}")
                print(f"Learned median normalized score: {get_stats(inst_learned_norm)['median']:.4f}")
            c_r_stats = get_stats(inst_curr_regrets)
            l_r_stats = get_stats(inst_learned_regrets)
            print(f"Current regret - mean: {c_r_stats['mean']:.4f}, median: {c_r_stats['median']:.4f}, p90: {c_r_stats['p90']:.4f}")
            print(f"Learned regret - mean: {l_r_stats['mean']:.4f}, median: {l_r_stats['median']:.4f}, p90: {l_r_stats['p90']:.4f}")

    print("\n=== AGGREGATE SUMMARY ===")
    print(f"Total decisions: {total_decisions}")
    print(f"Eligible decisions: {total_eligible}")
    print(f"Excluded decisions: {total_excluded}")
    
    if total_eligible > 0:
        print(f"Current top-1 agreement: {curr_top1_agreements} / {total_eligible} ({curr_top1_agreements/total_eligible*100:.1f}%)")
        print(f"Learned top-1 agreement: {learned_top1_agreements} / {total_eligible} ({learned_top1_agreements/total_eligible*100:.1f}%)")
        
        c_r_stats = get_stats(curr_regrets)
        l_r_stats = get_stats(learned_regrets)
        print(f"Current regret - mean: {c_r_stats['mean']:.4f}, median: {c_r_stats['median']:.4f}, p90: {c_r_stats['p90']:.4f}")
        print(f"Learned regret - mean: {l_r_stats['mean']:.4f}, median: {l_r_stats['median']:.4f}, p90: {l_r_stats['p90']:.4f}")
        
        if curr_normalized:
            print(f"Current median normalized score: {get_stats(curr_normalized)['median']:.4f}")
            print(f"Learned median normalized score: {get_stats(learned_normalized)['median']:.4f}")
            
    print("\n=== CLASSIFIER DIAGNOSTIC ===")
    test_X_cls = []
    test_y_cls = []
    for r in test_all_rows:
        if r['category'] == 'fin_fin':
            test_X_cls.append(r['features'])
            test_y_cls.append(0)
        elif r['category'] == 'fin_inf':
            test_X_cls.append(r['features'])
            test_y_cls.append(1)
            
    if test_X_cls:
        pred_y_cls = classifier.predict(test_X_cls)
        tp = sum(1 for t, p in zip(test_y_cls, pred_y_cls) if t == 1 and p == 1)
        tn = sum(1 for t, p in zip(test_y_cls, pred_y_cls) if t == 0 and p == 0)
        fp = sum(1 for t, p in zip(test_y_cls, pred_y_cls) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(test_y_cls, pred_y_cls) if t == 1 and p == 0)
        
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        print(f"Total candidates classified: {len(test_X_cls)}")
        print(f"Actual positive candidates (fin/inf): {sum(test_y_cls)}")
        print(f"Predicted positive: {tp + fp}")
        print(f"True positive: {tp}")
        print(f"False positive: {fp}")
        print(f"False negative: {fn}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
    
    print("\n=== FINAL DECISION ===")
    if total_eligible < 30:
        print("MORE DATA REQUIRED")
    else:
        # Check if learned is actually good
        print("MODEL REVISION REQUIRED")

if __name__ == '__main__':
    main()
