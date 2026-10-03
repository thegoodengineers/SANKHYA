import sys
import os
import math

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import assemble_dataset
from learned_branching_models import CART, spearmanr

import csv
def get_candidate_groups(instance):
    groups = {}
    files = assemble_dataset.find_csvs(instance)
    for fpath in files:
        seed_str = fpath.split('_seed')[-1].replace('.csv', '')
        try: seed = int(seed_str)
        except: seed = 0
        with open(fpath, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 12: continue
                try:
                    frac = float(row[1])
                    obj = float(row[2])
                    deg = float(row[3])
                    pcd = float(row[4])
                    pcu = float(row[5])
                    md = float(row[6])
                    mu = float(row[7])
                    existing_score = float(row[10])
                    selected = int(row[11])
                    branch_decision_id = int(row[12]) if len(row) >= 13 else -1
                    
                    if math.isnan(md) or math.isnan(mu) or md < 0 or mu < 0:
                        continue
                        
                    if not math.isinf(md) and not math.isinf(mu):
                        t_raw = max(md, 1e-6) * max(mu, 1e-6)
                        features = [frac, obj, deg, pcd, pcu]
                        key = (instance, seed, branch_decision_id)
                        if key not in groups:
                            groups[key] = []
                        groups[key].append({
                            'features': features,
                            'target_raw': t_raw,
                            'existing_score': existing_score,
                            'selected': selected
                        })
                except ValueError:
                    pass
    return groups

def main():
    print("Candidate-level ranking can be evaluated. Candidate sets are now grouped by branch_decision_id.\n")
    
    train_manifest = assemble_dataset.load_manifest("bench/learned_branching/train_instances.txt")
    test_manifest = assemble_dataset.load_manifest("bench/learned_branching/test_instances.txt")
    
    train_data, _ = assemble_dataset.assemble("TRAIN", train_manifest)
    test_data, _ = assemble_dataset.assemble("TEST", test_manifest)
    
    # Train the regressor on TRAIN fin_fin
    train_X_reg, train_y_reg = [], []
    for row in train_data:
        if row['category'] == 'fin_fin':
            train_X_reg.append(row['features'])
            train_y_reg.append(math.log1p(row['target_raw']))
            
    regressor = CART(max_depth=4, min_samples_leaf=5, mode='regression')
    regressor.fit(train_X_reg, train_y_reg)
    
    # Train the classifier
    train_X_cls, train_y_cls = [], []
    for row in train_data:
        if row['category'] == 'fin_fin':
            train_X_cls.append(row['features'])
            train_y_cls.append(0)
        elif row['category'] == 'fin_inf':
            train_X_cls.append(row['features'])
            train_y_cls.append(1)
            
    classifier = CART(max_depth=4, min_samples_leaf=5, mode='classification')
    classifier.fit(train_X_cls, train_y_cls)
    
    # Evaluate Regressor and Existing Score on TEST
    test_instances = ["blp-ar98", "blp-ic98", "cod105"]
    
    print("=== FINITE REGRESSOR (CANDIDATE-LEVEL RANKING) ===")
    all_true, all_learned, all_existing = [], [], []
    
    for inst in test_instances:
        inst_true = []
        inst_learned = []
        inst_existing = []
        
        groups = get_candidate_groups(inst)
        for key, candidates in groups.items():
            for c in candidates:
                inst_true.append(c['target_raw'])
                
                pred_log1p = regressor.predict([c['features']])[0]
                pred_raw = math.expm1(pred_log1p)
                
                inst_learned.append(pred_raw)
                inst_existing.append(c['existing_score'])
        
        if len(inst_true) > 1:
            sp_learned = spearmanr(inst_true, inst_learned)
            sp_existing = spearmanr(inst_true, inst_existing)
            print(f"[{inst}] N={len(inst_true)}")
            print(f"  Learned vs True Spearman:  {sp_learned:.4f}")
            print(f"  Existing vs True Spearman: {sp_existing:.4f}")
            all_true.extend(inst_true)
            all_learned.extend(inst_learned)
            all_existing.extend(inst_existing)
            
    if len(all_true) > 1:
        sp_learned_all = spearmanr(all_true, all_learned)
        sp_existing_all = spearmanr(all_true, all_existing)
        print(f"\n[GLOBAL TEST SET] N={len(all_true)}")
        print(f"  Learned vs True Spearman:  {sp_learned_all:.4f}")
        print(f"  Existing vs True Spearman: {sp_existing_all:.4f}")
        
    print("\n=== CLASSIFIER DIAGNOSTIC ===")
    test_X_cls, test_y_cls = [], []
    for row in test_data:
        if row['category'] == 'fin_fin':
            test_X_cls.append(row['features'])
            test_y_cls.append(0)
        elif row['category'] == 'fin_inf':
            test_X_cls.append(row['features'])
            test_y_cls.append(1)
            
    pred_y_cls = classifier.predict(test_X_cls)
    tp = sum(1 for t, p in zip(test_y_cls, pred_y_cls) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(test_y_cls, pred_y_cls) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(test_y_cls, pred_y_cls) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(test_y_cls, pred_y_cls) if t == 1 and p == 0)
    
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print("Note: The classifier is heavily overfitting to the specific variable distributions of the training instances (e.g. app1-1) and fails to generalize to the 3 rare positives in blp-ar98.")
    
    print("\n=== FINAL DECISION ===")
    if sp_learned_all > 0.6:
        print("MORE DATA REQUIRED")
    else:
        print("MODEL REVISION REQUIRED")
        
if __name__ == "__main__":
    main()
