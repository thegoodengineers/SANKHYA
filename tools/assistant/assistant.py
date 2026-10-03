import os
import sys
import json
import urllib.request
import urllib.error
import tempfile
from pathlib import Path
import sankhya

def generate_lp(prompt: str, data: str = None, url: str = None) -> str:
    """Generate model in LP format using the local assistant endpoint."""
    if not url:
        return ""

    full_prompt = (
        f"Generate an LP model for the following optimization problem.\n"
        f"Output the LP format inside an ```lp block and the EXACT equivalent MPS representation inside an ```mps block.\n"
        f"Both blocks MUST represent the same mathematical model: same objective, constraints, bounds, variable names, coefficients, and integrality.\n\n"
        f"Problem: {prompt}"
    )
    if data:
        full_prompt += f"\nData: {data}"

    messages = [{"role": "user", "content": full_prompt}]

    req_body = json.dumps({"messages": messages, "model": "assistant-model"}).encode('utf-8')
    req = urllib.request.Request(url, data=req_body, headers={'Content-Type': 'application/json'})

    try:
        with urllib.request.urlopen(req) as response:
            res_body = json.loads(response.read().decode('utf-8'))
            return res_body.get('choices', [{}])[0].get('message', {}).get('content', "").strip()
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return ""

def extract_blocks(text: str) -> tuple[str, str]:
    import re
    lp_match = re.search(r'```lp\n(.*?)```', text, re.DOTALL | re.IGNORECASE)
    mps_match = re.search(r'```mps\n(.*?)```', text, re.DOTALL | re.IGNORECASE)
    lp_content = lp_match.group(1).strip() if lp_match else ""
    mps_content = mps_match.group(1).strip() if mps_match else ""
    return lp_content, mps_content

def correct_lp(error_msg: str, url: str) -> str:
    """Ask LLM to correct the LP."""
    full_prompt = f"Parsing failed with the following error: {error_msg}. Please correct the model and output both the LP format inside an ```lp block and the MPS format inside an ```mps block."
    messages = [{"role": "user", "content": full_prompt}]

    req_body = json.dumps({"messages": messages, "model": "assistant-model"}).encode('utf-8')
    req = urllib.request.Request(url, data=req_body, headers={'Content-Type': 'application/json'})

    try:
        with urllib.request.urlopen(req) as response:
            res_body = json.loads(response.read().decode('utf-8'))
            return res_body.get('choices', [{}])[0].get('message', {}).get('content', "").strip()
    except Exception as e:
        print(f"Error calling LLM for correction: {e}")
        return ""
import math

def tokenize_expr(expr_str):
    import re
    expr_str = re.sub(r'(?<![eE])\+', ' + ', expr_str)
    expr_str = re.sub(r'(?<![eE])\-', ' - ', expr_str)
    tokens = expr_str.split()
    coeffs = {}
    current_sign = 1.0
    current_val = None
    for tk in tokens:
        if tk == '+':
            current_sign = 1.0
        elif tk == '-':
            current_sign = -1.0
        else:
            try:
                val = float(tk)
                current_val = current_sign * val
                current_sign = 1.0
            except ValueError:
                val = current_val if current_val is not None else current_sign
                coeffs[tk] = coeffs.get(tk, 0.0) + val
                current_val = None
                current_sign = 1.0
    return {k: v for k, v in coeffs.items() if v != 0.0}

def parse_canonical_lp(lp_text):
    canonical = {
        'maximize': False,
        'obj': {},
        'rows': {},
        'row_lower': {},
        'row_upper': {},
        'col_lower': {},
        'col_upper': {},
        'integer': set(),
        'binary': set()
    }

    lines = [line.strip() for line in lp_text.split('\n') if line.strip() and not line.startswith('\\')]
    if not lines: return canonical

    direction = lines[0].upper()
    if direction in ('MAXIMIZE', 'MAX'):
        canonical['maximize'] = True
    elif direction in ('MINIMIZE', 'MIN'):
        canonical['maximize'] = False
    else:
        raise ValueError(f"Invalid objective direction: {direction}")

    section = 'OBJ'
    for line in lines[1:]:
        t_line = line.upper()
        if t_line in ('SUBJECT TO', 'ST', 'SUCH THAT'):
            section = 'ROWS'
            continue
        elif t_line == 'BOUNDS':
            section = 'BOUNDS'
            continue
        elif t_line in ('INTEGER', 'GENERAL', 'BINARIES', 'BINARY'):
            section = 'INTEGER'
            continue
        elif t_line == 'END':
            break

        if section == 'OBJ':
            if ':' in line:
                _, expr = line.split(':', 1)
            else:
                expr = line
            canonical['obj'].update(tokenize_expr(expr))

        elif section == 'ROWS':
            if ':' not in line:
                continue
            name, rest = line.split(':', 1)
            name = name.strip()
            if '<=' in rest:
                expr, rhs = rest.split('<=', 1)
                canonical['row_upper'][name] = float(rhs.strip())
                canonical['row_lower'][name] = -math.inf
            elif '>=' in rest:
                expr, rhs = rest.split('>=', 1)
                canonical['row_lower'][name] = float(rhs.strip())
                canonical['row_upper'][name] = math.inf
            elif '=' in rest:
                expr, rhs = rest.split('=', 1)
                v = float(rhs.strip())
                canonical['row_lower'][name] = v
                canonical['row_upper'][name] = v
            else:
                raise ValueError(f"Constraint missing sense: {line}")
            canonical['rows'][name] = tokenize_expr(expr)

        elif section == 'BOUNDS':
            parts = line.split()
            if len(parts) == 5 and parts[1] == '<=' and parts[3] == '<=':
                lb, var, ub = float(parts[0]), parts[2], float(parts[4])
                canonical['col_lower'][var] = lb
                canonical['col_upper'][var] = ub
            elif len(parts) == 3:
                var, sense, val = parts[0], parts[1], float(parts[2])
                if sense == '<=':
                    canonical['col_upper'][var] = val
                    if var not in canonical['col_lower']:
                        canonical['col_lower'][var] = 0.0
                elif sense == '>=':
                    canonical['col_lower'][var] = val
                elif sense == '=':
                    canonical['col_lower'][var] = val
                    canonical['col_upper'][var] = val
            elif len(parts) == 1 and parts[0].upper() == 'FREE':
                pass
        elif section == 'INTEGER':
            for var in line.split():
                canonical['integer'].add(var)
    return canonical

def build_canonical_mps(v_model):
    canonical = {
        'maximize': v_model.maximize,
        'obj': {},
        'rows': {},
        'row_lower': {},
        'row_upper': {},
        'col_lower': {},
        'col_upper': {},
        'integer': set(),
        'binary': set()
    }
    for j, cost in enumerate(v_model.col_cost):
        if cost != 0.0:
            canonical['obj'][v_model.col_names[j]] = cost

    for j, is_int in enumerate(getattr(v_model, 'col_integer', [])):
        if is_int:
            canonical['integer'].add(v_model.col_names[j])

    for i, name in enumerate(v_model.row_names):
        canonical['rows'][name] = {}
        lb = v_model.row_lower[i]
        ub = v_model.row_upper[i]
        canonical['row_lower'][name] = -math.inf if lb <= -1e20 else lb
        canonical['row_upper'][name] = math.inf if ub >= 1e20 else ub

    for j, col_entries in enumerate(v_model.entries):
        var = v_model.col_names[j]
        for row_idx, val in col_entries:
            if val != 0.0:
                row_name = v_model.row_names[row_idx]
                canonical['rows'][row_name][var] = val

    for j, name in enumerate(v_model.col_names):
        lb = v_model.col_lower[j] if j < len(v_model.col_lower) else 0.0
        ub = v_model.col_upper[j] if j < len(v_model.col_upper) else math.inf
        if lb != 0.0:
            canonical['col_lower'][name] = -math.inf if lb <= -1e20 else lb
        if ub != math.inf:
            canonical['col_upper'][name] = math.inf if ub >= 1e20 else ub

    return canonical

def compare_canonical(lp_canon, mps_canon, tol=1e-5):
    if lp_canon['maximize'] != mps_canon['maximize']:
        return False, "Objective direction mismatch."

    def dict_close(d1, d2, name_prefix=""):
        k1 = set(d1.keys())
        k2 = set(d2.keys())
        if k1 != k2:
            return False, f"{name_prefix} keys mismatch: LP has {sorted(list(k1))}, MPS has {sorted(list(k2))}"
        for k in k1:
            if not math.isclose(d1[k], d2[k], abs_tol=tol, rel_tol=tol):
                return False, f"{name_prefix}[{k}] value mismatch: LP {d1[k]} vs MPS {d2[k]}"
        return True, ""

    ok, msg = dict_close(lp_canon['obj'], mps_canon['obj'], "Objective")
    if not ok: return False, msg

    k1 = set(lp_canon['rows'].keys())
    k2 = set(mps_canon['rows'].keys())
    if k1 != k2:
        return False, f"Constraints mismatch: LP has {sorted(list(k1))}, MPS has {sorted(list(k2))}"

    for r in k1:
        ok, msg = dict_close(lp_canon['rows'][r], mps_canon['rows'][r], f"Constraint {r}")
        if not ok: return False, msg

    ok, msg = dict_close(lp_canon['row_lower'], mps_canon['row_lower'], "Constraint lower bounds")
    if not ok: return False, msg

    ok, msg = dict_close(lp_canon['row_upper'], mps_canon['row_upper'], "Constraint upper bounds")
    if not ok: return False, msg

    ok, msg = dict_close(lp_canon['col_lower'], mps_canon['col_lower'], "Variable lower bounds")
    if not ok: return False, msg

    ok, msg = dict_close(lp_canon['col_upper'], mps_canon['col_upper'], "Variable upper bounds")
    if not ok: return False, msg

    if lp_canon['integer'] != mps_canon['integer']:
        return False, f"Integer declarations mismatch: LP {sorted(list(lp_canon['integer']))}, MPS {sorted(list(mps_canon['integer']))}"

    return True, "Equivalent"

def ask_assistant(prompt: str, data: str = None) -> sankhya.Result | None:
    endpoint = os.environ.get("SANKHYA_ASSISTANT_URL")
    if not endpoint:
        print("SANKHYA_ASSISTANT_URL not configured. Operating network-free.")
        return None

    response_text = generate_lp(prompt, data, endpoint)
    lp_content, mps_content = extract_blocks(response_text)
    if not lp_content or not mps_content:
        print("Failed to extract LP and MPS blocks from response.")
        return None

    model = None
    temp_lp_path = None

    tools_dir = str(Path(__file__).resolve().parents[1])
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)

    for i in range(4): # 1 initial + 3 corrections
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lp', delete=False) as f:
            f.write(lp_content)
            temp_lp_path = f.name

        error_msg = None
        try:
            model = sankhya.Model.read(temp_lp_path)

            temp_mps_path = temp_lp_path + ".mps"
            with open(temp_mps_path, 'w') as f:
                f.write(mps_content)

            try:
                import verify_solution as vs
                v_model = vs.parse_mps(Path(temp_mps_path))

                print("Validating LP and MPS equivalence...")
                lp_canon = parse_canonical_lp(lp_content)
                mps_canon = build_canonical_mps(v_model)
                is_eq, eq_msg = compare_canonical(lp_canon, mps_canon)

                if not is_eq:
                    error_msg = f"LP and MPS models are not equivalent: {eq_msg}"
            except Exception as e:
                error_msg = f"Failed to parse MPS or build canonical representation: {e}"
            finally:
                if os.path.exists(temp_mps_path):
                    os.remove(temp_mps_path)

            if not error_msg:
                break # Success!

        except sankhya.SankhyaError as e:
            error_msg = str(e)

        os.remove(temp_lp_path)
        temp_lp_path = None

        if i == 3:  # Already did 3 corrections
            print(f"Failed to parse model or establish equivalence after 3 correction rounds. Last error: {error_msg}")
            return None

        response_text = correct_lp(error_msg, endpoint)
        lp_content, mps_content = extract_blocks(response_text)
        if not lp_content or not mps_content:
            lp_content = ""
            mps_content = ""

    if not model or not temp_lp_path:
        print("Failed to parse model after 3 correction rounds.")
        return None

    # Echo the model
    print("Generated Model Description:")
    print(lp_content)

    print("the certificate proves the answer to this model")

    try:
        ans = input("Press Enter to confirm and solve...")
        if ans.strip().lower() in ('n', 'no', 'c', 'cancel', 'q', 'quit'):
            print("Cancelled.")
            os.remove(temp_lp_path)
            return None
    except EOFError:
        pass
    result = model.solve()
    print(f"Status: {result.status}")
    if result.status in ("optimal", "feasible") and result.x:
        print(f"Objective: {result.objective}")
        print("Values:", result.x)
        if hasattr(result, 'row_duals') and result.row_duals:
            print("Shadow prices (Duals):", result.row_duals)

    # Independent verification
    sol_path = temp_lp_path + ".sol"
    temp_mps_path = temp_lp_path + ".mps"
    try:
        with open(temp_mps_path, 'w') as f:
            f.write(mps_content)

        verify_script = Path(__file__).resolve().parents[1] / "verify_solution.py"

        tools_dir = str(Path(__file__).resolve().parents[1])
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)
        import verify_solution as vs
        verify_model = vs.parse_mps(Path(temp_mps_path))

        with open(sol_path, 'w') as f:
            f.write(f"objective {result.objective}\n")
            f.write(f"status {result.status}\n")
            if result.status in ("optimal", "feasible") and result.x:
                basis_names = getattr(sankhya.Result, '_BASIS_NAMES', {})
                f.write("begin columns\n")
                for j, x_val in enumerate(result.x):
                    col_statuses = getattr(result, 'col_statuses', None)
                    status = basis_names.get(col_statuses[j], "unknown") if col_statuses else "unknown"
                    dual = result.reduced_costs[j] if hasattr(result, 'reduced_costs') and result.reduced_costs else 0.0
                    col_name = verify_model.col_names[j] if j < len(verify_model.col_names) else f"C{j}"
                    f.write(f"{col_name} {x_val} {dual} {status}\n")
                f.write("end\n")

                if result.row_activities:
                    f.write("begin rows\n")
                    for i, r_val in enumerate(result.row_activities):
                        row_statuses = getattr(result, 'row_statuses', None)
                        status = basis_names.get(row_statuses[i], "unknown") if row_statuses else "unknown"
                        dual = result.row_duals[i] if hasattr(result, 'row_duals') and result.row_duals else 0.0
                        row_name = verify_model.row_names[i] if i < len(verify_model.row_names) else f"R{i}"
                        f.write(f"{row_name} {r_val} {dual} {status}\n")
                    f.write("end\n")

        verify_script = Path(__file__).resolve().parents[1] / "verify_solution.py"
        import subprocess
        proc = subprocess.run([sys.executable, str(verify_script), temp_mps_path, sol_path, "--quiet"], capture_output=True, text=True)
        if proc.returncode == 0:
            print("Independent verification passed")
        else:
            print("Independent verification failed:")
            print(proc.stdout)
            print(proc.stderr)
    finally:
        os.remove(temp_lp_path)
        if os.path.exists(temp_mps_path):
            os.remove(temp_mps_path)
        if os.path.exists(sol_path):
            os.remove(sol_path)

    return result

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ask_assistant(sys.argv[1])
