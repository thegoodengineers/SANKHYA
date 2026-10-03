import csv
import glob
import math
import os
import json
import sys

def load_manifest(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

def get_stats(data):
    if not data:
        return {'min': 0, 'median': 0, 'mean': 0, 'p90': 0, 'p95': 0, 'p99': 0, 'max': 0}
    s = sorted(data)
    n = len(s)
    return {
        'min': s[0],
        'median': s[n//2],
        'mean': sum(s)/n,
        'p90': s[int(n*0.90)],
        'p95': s[int(n*0.95)],
        'p99': s[int(n*0.99)],
        'max': s[-1]
    }

def find_csvs(instance):
    # Search in scratch/learned_branching/train, test, discovery
    pattern = f"scratch/learned_branching/*/{instance}_seed*.csv"
    return sorted(glob.glob(pattern))

def assemble(manifest_name, instances):
    dataset = []
    
    report = {
        'total_rows': 0,
        'usable_fin_fin': 0,
        'usable_fin_inf': 0,
        'usable_inf_inf': 0,
        'excluded_unmeasured': 0,
        'malformed_nan': 0,
        'instances': {}
    }
    
    all_targets = []
    all_features = {'frac': [], 'obj': [], 'deg': [], 'pcd': [], 'pcu': []}
    
    for inst in instances:
        files = find_csvs(inst)
        if not files:
            print(f"WARNING: No CSVs found for {inst}")
        
        inst_rep = {
            'total_rows': 0,
            'usable_fin_fin': 0,
            'usable_fin_inf': 0,
            'usable_inf_inf': 0,
            'excluded_unmeasured': 0,
            'malformed_nan': 0,
            'targets': []
        }
        
        for fpath in files:
            seed_str = fpath.split('_seed')[-1].replace('.csv', '')
            try:
                seed = int(seed_str)
            except:
                seed = 0
            
            with open(fpath, 'r', newline='') as f:
                reader = csv.reader(f)
                for row in reader:
                    inst_rep['total_rows'] += 1
                    report['total_rows'] += 1
                    
                    if len(row) < 12:
                        inst_rep['malformed_nan'] += 1
                        report['malformed_nan'] += 1
                        continue
                        
                    try:
                        frac = float(row[1])
                        obj = float(row[2])
                        deg = float(row[3])
                        pcd = float(row[4])
                        pcu = float(row[5])
                        md = float(row[6])
                        mu = float(row[7])
                        
                        if math.isnan(frac) or math.isnan(obj) or math.isnan(deg) or math.isnan(pcd) or math.isnan(pcu) or math.isnan(md) or math.isnan(mu):
                            inst_rep['malformed_nan'] += 1
                            report['malformed_nan'] += 1
                            continue
                            
                    except ValueError:
                        inst_rep['malformed_nan'] += 1
                        report['malformed_nan'] += 1
                        continue
                        
                    if md < 0 or mu < 0:
                        inst_rep['excluded_unmeasured'] += 1
                        report['excluded_unmeasured'] += 1
                        continue
                        
                    is_md_inf = math.isinf(md)
                    is_mu_inf = math.isinf(mu)
                    
                    category = ""
                    t_raw = None
                    if not is_md_inf and not is_mu_inf:
                        category = "fin_fin"
                        t_raw = max(md, 1e-6) * max(mu, 1e-6)
                    elif is_md_inf and is_mu_inf:
                        category = "inf_inf"
                    else:
                        category = "fin_inf"
                        
                    if category == "fin_fin":
                        inst_rep['usable_fin_fin'] += 1
                        report['usable_fin_fin'] += 1
                        inst_rep['targets'].append(t_raw)
                        all_targets.append(t_raw)
                    elif category == "fin_inf":
                        inst_rep['usable_fin_inf'] += 1
                        report['usable_fin_inf'] += 1
                    else:
                        inst_rep['usable_inf_inf'] += 1
                        report['usable_inf_inf'] += 1
                        
                    # Save observation
                    dataset.append({
                        'instance': inst,
                        'seed': seed,
                        'branch_decision_id': int(row[12]) if len(row) >= 13 else -1,
                        'category': category,
                        'features': [frac, obj, deg, pcd, pcu],
                        'target_raw': t_raw
                    })
                    
                    if category == "fin_fin":
                        all_features['frac'].append(frac)
                        all_features['obj'].append(obj)
                        all_features['deg'].append(deg)
                        all_features['pcd'].append(pcd)
                        all_features['pcu'].append(pcu)
        
        report['instances'][inst] = inst_rep
        
    report['target_stats'] = get_stats(all_targets)
    report['feature_stats'] = {k: get_stats(v) for k,v in all_features.items()}
    
    return dataset, report

def main():
    train_manifest = load_manifest("bench/learned_branching/train_instances.txt")
    test_manifest = load_manifest("bench/learned_branching/test_instances.txt")
    
    # 1. Verification
    print("=== MANIFEST VERIFICATION ===")
    train_set = set(train_manifest)
    test_set = set(test_manifest)
    print(f"Train instances: {train_manifest}")
    print(f"Test instances: {test_manifest}")
    overlap = train_set.intersection(test_set)
    print(f"Train/Test Overlap: {overlap}")
    if overlap:
        print("ERROR: TRAIN ∩ TEST is not empty!")
        sys.exit(1)
        
    # 2. Assemble
    print("\n=== ASSEMBLING TRAIN DATASET ===")
    train_data, train_rep = assemble("TRAIN", train_manifest)
    print(f"Total rows: {train_rep['total_rows']}")
    print(f"Usable fin/fin: {train_rep['usable_fin_fin']}")
    print(f"Usable fin/inf: {train_rep['usable_fin_inf']}")
    print(f"Usable inf/inf: {train_rep['usable_inf_inf']}")
    print(f"Excluded unmeasured: {train_rep['excluded_unmeasured']}")
    print(f"Malformed/NaN: {train_rep['malformed_nan']}")
    
    print("\nTrain Instance Breakdown:")
    total_finfin = max(1, train_rep['usable_fin_fin'])
    for inst in train_manifest:
        r = train_rep['instances'].get(inst)
        if not r: continue
        pct = (r['usable_fin_fin'] / total_finfin) * 100
        print(f"  {inst:15s}: fin/fin={r['usable_fin_fin']:4d} ({pct:5.1f}%), fin/inf={r['usable_fin_inf']:4d}, inf/inf={r['usable_inf_inf']:4d}, excl={r['excluded_unmeasured']}")

    print("\nTrain Target Stats (fin/fin):")
    t = train_rep['target_stats']
    print(f"  min={t['min']:.4g}, median={t['median']:.4g}, mean={t['mean']:.4g}, p90={t['p90']:.4g}, max={t['max']:.4g}")
    
    print("\n=== ASSEMBLING TEST DATASET ===")
    test_data, test_rep = assemble("TEST", test_manifest)
    print(f"Total rows: {test_rep['total_rows']}")
    print(f"Usable fin/fin: {test_rep['usable_fin_fin']}")
    print(f"Usable fin/inf: {test_rep['usable_fin_inf']}")
    print(f"Usable inf/inf: {test_rep['usable_inf_inf']}")
    print(f"Excluded unmeasured: {test_rep['excluded_unmeasured']}")
    print(f"Malformed/NaN: {test_rep['malformed_nan']}")
    
    print("\nTest Instance Breakdown:")
    total_finfin = max(1, test_rep['usable_fin_fin'])
    for inst in test_manifest:
        r = test_rep['instances'].get(inst)
        if not r: continue
        pct = (r['usable_fin_fin'] / total_finfin) * 100
        print(f"  {inst:15s}: fin/fin={r['usable_fin_fin']:4d} ({pct:5.1f}%), fin/inf={r['usable_fin_inf']:4d}, inf/inf={r['usable_inf_inf']:4d}, excl={r['excluded_unmeasured']}")

    print("\nTest Target Stats (fin/fin):")
    t = test_rep['target_stats']
    print(f"  min={t['min']:.4g}, median={t['median']:.4g}, mean={t['mean']:.4g}, p90={t['p90']:.4g}, max={t['max']:.4g}")
    
    # Check 2-part learning support
    print("\n=== TWO-PART LEARNING SUPPORT ===")
    total_fin_inf = train_rep['usable_fin_inf'] + test_rep['usable_fin_inf']
    if total_fin_inf > 100:
        print(f"SUPPORTED: Found {total_fin_inf} fin/inf (infeasible branch) observations.")
        print("This data volume provides a sufficient signal to start training an infeasibility classifier.")
    else:
        print(f"INSUFFICIENT: Found only {total_fin_inf} fin/inf (infeasible branch) observations.")
        print("More data is required to robustly train an infeasibility classifier.")

if __name__ == "__main__":
    main()
