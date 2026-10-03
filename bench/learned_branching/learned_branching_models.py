import math
import os
import sys

# Import assembly module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import assemble_dataset

class Node:
    def __init__(self, value, is_leaf=False):
        self.value = value
        self.is_leaf = is_leaf
        self.feature_idx = -1
        self.threshold = 0
        self.left = None
        self.right = None

class CART:
    def __init__(self, max_depth, min_samples_leaf, mode='regression'):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.mode = mode
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, 0)
        
    def _build_tree(self, X, y, depth):
        if len(y) == 0:
            return Node(0, True)
            
        value = sum(y) / len(y) if self.mode == 'regression' else max(set(y), key=y.count)
        
        if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf or len(set(y)) == 1:
            return Node(value, True)
            
        best_loss = float('inf')
        best_split = None
        
        n_features = len(X[0])
        for f in range(n_features):
            vals = sorted(list(set(row[f] for row in X)))
            for i in range(len(vals) - 1):
                thresh = (vals[i] + vals[i+1]) / 2.0
                
                left_y = []
                right_y = []
                for j in range(len(y)):
                    if X[j][f] <= thresh:
                        left_y.append(y[j])
                    else:
                        right_y.append(y[j])
                
                if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
                    continue
                    
                loss = self._compute_loss(left_y, right_y)
                if loss < best_loss - 1e-9:
                    best_loss = loss
                    best_split = (f, thresh, left_y, right_y)
                    
        if best_split is None:
            return Node(value, True)
            
        f, thresh, left_y, right_y = best_split
        node = Node(value, False)
        node.feature_idx = f
        node.threshold = thresh
        
        left_X = []
        right_X = []
        for j in range(len(y)):
            if X[j][f] <= thresh:
                left_X.append(X[j])
            else:
                right_X.append(X[j])
        
        node.left = self._build_tree(left_X, left_y, depth + 1)
        node.right = self._build_tree(right_X, right_y, depth + 1)
        return node
        
    def _compute_loss(self, left_y, right_y):
        if self.mode == 'regression':
            mean_l = sum(left_y) / len(left_y)
            mean_r = sum(right_y) / len(right_y)
            var_l = sum((val - mean_l)**2 for val in left_y)
            var_r = sum((val - mean_r)**2 for val in right_y)
            return var_l + var_r
        else:
            n_l, n_r = len(left_y), len(right_y)
            p1_l = sum(left_y) / n_l
            gini_l = 1.0 - p1_l**2 - (1 - p1_l)**2
            p1_r = sum(right_y) / n_r
            gini_r = 1.0 - p1_r**2 - (1 - p1_r)**2
            return n_l * gini_l + n_r * gini_r

    def predict(self, X):
        return [self._predict_one(x, self.tree) for x in X]
        
    def _predict_one(self, x, node):
        if node.is_leaf:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

def get_ranks(arr):
    s = sorted([(val, i) for i, val in enumerate(arr)])
    ranks = [0.0] * len(arr)
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j][0] == s[i][0]:
            j += 1
        avg_rank = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[s[k][1]] = avg_rank
        i = j
    return ranks

def spearmanr(x, y):
    n = len(x)
    if n < 2: return 0.0
    rx = get_ranks(x)
    ry = get_ranks(y)
    d2 = sum((rx[i] - ry[i])**2 for i in range(n))
    denom = (n * (n**2 - 1))
    if denom == 0: return 0.0
    return 1.0 - (6.0 * d2) / denom

def generate_cpp(node, indent="    ", is_class=False):
    if node.is_leaf:
        if is_class:
            return f"{indent}return {'true' if node.value == 1 else 'false'};\n"
        else:
            return f"{indent}return {node.value};\n"
    
    features = ["frac", "obj", "deg", "pcd", "pcu"]
    f_name = features[node.feature_idx]
    
    code = f"{indent}if ({f_name} <= {node.threshold}) {{\n"
    code += generate_cpp(node.left, indent + "    ", is_class)
    code += f"{indent}}} else {{\n"
    code += generate_cpp(node.right, indent + "    ", is_class)
    code += f"{indent}}}\n"
    return code

def main():
    print("=== ASSEMBLING DATASETS ===")
    train_manifest = assemble_dataset.load_manifest("bench/learned_branching/train_instances.txt")
    test_manifest = assemble_dataset.load_manifest("bench/learned_branching/test_instances.txt")
    
    train_data, train_rep = assemble_dataset.assemble("TRAIN", train_manifest)
    test_data, test_rep = assemble_dataset.assemble("TEST", test_manifest)
    
    print("=== PART A: FINITE SCORE REGRESSOR ===")
    train_X_reg, train_y_reg, train_y_raw = [], [], []
    test_X_reg, test_y_reg, test_y_raw, test_inst = [], [], [], []
    
    for row in train_data:
        if row['category'] == 'fin_fin':
            train_X_reg.append(row['features'])
            train_y_raw.append(row['target_raw'])
            train_y_reg.append(math.log1p(row['target_raw']))
            
    for row in test_data:
        if row['category'] == 'fin_fin':
            test_X_reg.append(row['features'])
            test_y_raw.append(row['target_raw'])
            test_y_reg.append(math.log1p(row['target_raw']))
            test_inst.append(row['instance'])
            
    print(f"Train finite/finite rows: {len(train_X_reg)}")
    print(f"Test finite/finite rows: {len(test_X_reg)}")
    
    regressor = CART(max_depth=4, min_samples_leaf=5, mode='regression')
    regressor.fit(train_X_reg, train_y_reg)
    
    pred_y_reg = regressor.predict(test_X_reg)
    pred_y_raw = [math.expm1(p) for p in pred_y_reg]
    
    def calc_metrics(y_true, y_pred):
        if not y_true: return 0, 0
        mae = sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)
        rmse = math.sqrt(sum((t - p)**2 for t, p in zip(y_true, y_pred)) / len(y_true))
        return mae, rmse
        
    mae_log, rmse_log = calc_metrics(test_y_reg, pred_y_reg)
    mae_raw, rmse_raw = calc_metrics(test_y_raw, pred_y_raw)
    sp = spearmanr(test_y_raw, pred_y_raw)
    
    base_pred_log = sum(train_y_reg) / len(train_y_reg) if train_y_reg else 0
    base_pred_raw = math.expm1(base_pred_log)
    base_mae_log, base_rmse_log = calc_metrics(test_y_reg, [base_pred_log]*len(test_y_reg))
    base_mae_raw, base_rmse_raw = calc_metrics(test_y_raw, [base_pred_raw]*len(test_y_raw))
    
    print("\nRegression Metrics (TEST):")
    print(f"MAE (log1p):   {mae_log:.4f} (Baseline: {base_mae_log:.4f})")
    print(f"RMSE (log1p):  {rmse_log:.4f} (Baseline: {base_rmse_log:.4f})")
    print(f"MAE (raw):     {mae_raw:.4f} (Baseline: {base_mae_raw:.4f})")
    print(f"RMSE (raw):    {rmse_raw:.4f} (Baseline: {base_rmse_raw:.4f})")
    print(f"Spearman Corr: {sp:.4f}")
    
    print("\nPer-instance regression metrics (TEST):")
    for inst in ["blp-ar98", "blp-ic98", "cod105"]:
        i_true = [test_y_raw[i] for i in range(len(test_y_raw)) if test_inst[i] == inst]
        i_pred = [pred_y_raw[i] for i in range(len(test_y_raw)) if test_inst[i] == inst]
        i_mae, i_rmse = calc_metrics(i_true, i_pred)
        i_sp = spearmanr(i_true, i_pred)
        print(f"  {inst}: MAE={i_mae:.4f}, RMSE={i_rmse:.4f}, Spearman={i_sp:.4f}")
        
    print("\n=== PART B: INFEASIBILITY CLASSIFIER ===")
    train_X_cls, train_y_cls = [], []
    test_X_cls, test_y_cls, test_inst_cls = [], [], []
    
    for row in train_data:
        if row['category'] == 'fin_fin':
            train_X_cls.append(row['features'])
            train_y_cls.append(0)
        elif row['category'] == 'fin_inf':
            train_X_cls.append(row['features'])
            train_y_cls.append(1)
            
    for row in test_data:
        if row['category'] == 'fin_fin':
            test_X_cls.append(row['features'])
            test_y_cls.append(0)
            test_inst_cls.append(row['instance'])
        elif row['category'] == 'fin_inf':
            test_X_cls.append(row['features'])
            test_y_cls.append(1)
            test_inst_cls.append(row['instance'])
            
    print(f"Train rows: pos={sum(train_y_cls)}, neg={len(train_y_cls)-sum(train_y_cls)}")
    print(f"Test rows: pos={sum(test_y_cls)}, neg={len(test_y_cls)-sum(test_y_cls)}")
    
    classifier = CART(max_depth=4, min_samples_leaf=5, mode='classification')
    classifier.fit(train_X_cls, train_y_cls)
    
    pred_y_cls = classifier.predict(test_X_cls)
    
    def calc_cls(y_true, y_pred):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        acc = (tp + tn) / max(1, len(y_true))
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        return tp, tn, fp, fn, acc, prec, rec, f1
        
    tp, tn, fp, fn, acc, prec, rec, f1 = calc_cls(test_y_cls, pred_y_cls)
    
    base_pred_cls = 1 if sum(train_y_cls) > len(train_y_cls)/2 else 0
    _, _, _, _, b_acc, b_prec, b_rec, b_f1 = calc_cls(test_y_cls, [base_pred_cls]*len(test_y_cls))
    
    print("\nClassification Metrics (TEST):")
    print("WARNING: Test set has only 3 positive infeasibility examples. Evaluation is extremely limited.")
    print(f"Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"Accuracy:  {acc:.4f} (Baseline: {b_acc:.4f})")
    print(f"Precision: {prec:.4f} (Baseline: {b_prec:.4f})")
    print(f"Recall:    {rec:.4f} (Baseline: {b_rec:.4f})")
    print(f"F1:        {f1:.4f} (Baseline: {b_f1:.4f})")
    
    print("\nPer-instance classification metrics (TEST):")
    for inst in ["blp-ar98", "blp-ic98", "cod105"]:
        i_true = [test_y_cls[i] for i in range(len(test_y_cls)) if test_inst_cls[i] == inst]
        i_pred = [pred_y_cls[i] for i in range(len(test_y_cls)) if test_inst_cls[i] == inst]
        if not i_true: continue
        i_tp, i_tn, i_fp, i_fn, i_acc, _, _, _ = calc_cls(i_true, i_pred)
        print(f"  {inst}: pos={sum(i_true)}, neg={len(i_true)-sum(i_true)} | Acc={i_acc:.4f}, TP={i_tp}, TN={i_tn}, FP={i_fp}, FN={i_fn}")
        
    # Generate C++ Code
    os.makedirs("scratch/generated", exist_ok=True)
    out_path = "scratch/generated/learned_branching.cpp"
    with open(out_path, "w") as f:
        f.write("// GENERATED BY bench/learned_branching/learned_branching_models.py\n")
        f.write("// DO NOT EDIT DIRECTLY\n\n")
        
        f.write("double ScoreBranchingCandidate(float frac, double obj, int deg, double pcd, double pcu) {\n")
        f.write(generate_cpp(regressor.tree, "    ", False))
        f.write("}\n\n")
        
        f.write("bool IsInfeasible(float frac, double obj, int deg, double pcd, double pcu) {\n")
        f.write(generate_cpp(classifier.tree, "    ", True))
        f.write("}\n")
        
    print(f"\nGenerated C++ models saved to {out_path}")

if __name__ == "__main__":
    main()
