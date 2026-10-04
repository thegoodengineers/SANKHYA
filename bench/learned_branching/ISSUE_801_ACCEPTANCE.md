# Issue #801 Final Acceptance Benchmark Report

## 1. Implementation Status
- **Rule Options**: The learned rule operates via its own dedicated `--option mip_learned_branching=true/false`.
- **Default state**: The default value is strictly `OFF` (`false`), ensuring the baseline paths remain untouched unless explicitly requested.
- **ML dependency**: The V3 Pairwise model has been transpiled strictly into pure C++ double-precision constants and expressions (`ScoreBranchingCandidateV3`), avoiding any external ML runtime dependency.
- **Validation Constraints**: Tested against `-Werror=double-promotion` to verify strict compilation compliance.

## 2. Disjoint Test / Training Split
- **Confirmation**: Verified that the training instances (`30n20b8`, `CMS750_4`, `app1-1`, `app1-2`, `bnatt400`, `cbs-cta`, `cmflsp50-24-8-8`) in `train_instances.txt` are completely disjoint from the instances evaluated below (`blp-ar98`, `blp-ic98`, `cod105`). No reported instance entered training.

## 3. Benchmark Matrix
The benchmark evaluates three instances (`blp-ar98`, `blp-ic98`, `cod105`) over 3 deterministic seeds (0, 1, 2) across 2 time limits (60s, 300s) on a single thread (`mip_threads=1`) using two configurations:
1. **Baseline**: `mip_learned_branching=false`
2. **Learned V3**: `mip_learned_branching=true`

## 4. Raw Per-Run Results
| Instance | Seed | Time Limit | Configuration | Opt Reached | Nodes | Solve Time (s) | Best Obj | Certificate |
|----------|------|------------|---------------|-------------|-------|----------------|----------|-------------|
| blp-ar98 | 0 | 60s | Baseline (OFF) | No | 18 | 60.00 | inf | N/A |
| blp-ar98 | 0 | 60s | Learned (V3) | No | 13 | 60.00 | inf | N/A |
| blp-ar98 | 1 | 60s | Baseline (OFF) | No | 6 | 60.00 | inf | N/A |
| blp-ar98 | 1 | 60s | Learned (V3) | No | 13 | 60.00 | inf | N/A |
| blp-ar98 | 2 | 60s | Baseline (OFF) | No | 16 | 60.00 | inf | N/A |
| blp-ar98 | 2 | 60s | Learned (V3) | No | 18 | 60.00 | inf | N/A |
| blp-ar98 | 0 | 300s | Baseline (OFF) | No | 76 | 300.00 | inf | N/A |
| blp-ar98 | 0 | 300s | Learned (V3) | No | 52 | 300.00 | inf | N/A |
| blp-ar98 | 1 | 300s | Baseline (OFF) | No | 62 | 300.00 | inf | N/A |
| blp-ar98 | 1 | 300s | Learned (V3) | No | 51 | 300.00 | inf | N/A |
| blp-ar98 | 2 | 300s | Baseline (OFF) | No | 60 | 300.00 | inf | N/A |
| blp-ar98 | 2 | 300s | Learned (V3) | No | 48 | 300.00 | inf | N/A |
| blp-ic98 | 0 | 60s | Baseline (OFF) | No | 5 | 60.00 | inf | N/A |
| blp-ic98 | 0 | 60s | Learned (V3) | No | 7 | 60.00 | inf | N/A |
| blp-ic98 | 1 | 60s | Baseline (OFF) | No | 4 | 60.00 | inf | N/A |
| blp-ic98 | 1 | 60s | Learned (V3) | No | 4 | 60.00 | inf | N/A |
| blp-ic98 | 2 | 60s | Baseline (OFF) | No | 8 | 60.00 | inf | N/A |
| blp-ic98 | 2 | 60s | Learned (V3) | No | 7 | 60.00 | inf | N/A |
| blp-ic98 | 0 | 300s | Baseline (OFF) | No | 70 | 300.00 | 5320.8123 | N/A |
| blp-ic98 | 0 | 300s | Learned (V3) | No | 55 | 300.00 | 5779.1153 | N/A |
| blp-ic98 | 1 | 300s | Baseline (OFF) | No | 69 | 300.00 | 5888.7815 | N/A |
| blp-ic98 | 1 | 300s | Learned (V3) | No | 77 | 300.00 | 6187.5762 | N/A |
| blp-ic98 | 2 | 300s | Baseline (OFF) | No | 58 | 300.00 | 5807.9710 | N/A |
| blp-ic98 | 2 | 300s | Learned (V3) | No | 50 | 300.00 | 6222.0075 | N/A |
| cod105 | 0 | 60s | Baseline (OFF) | No | 1 | 60.00 | inf | N/A |
| cod105 | 0 | 60s | Learned (V3) | No | 1 | 60.00 | inf | N/A |
| cod105 | 1 | 60s | Baseline (OFF) | No | 1 | 60.00 | inf | N/A |
| cod105 | 1 | 60s | Learned (V3) | No | 1 | 60.00 | inf | N/A |
| cod105 | 2 | 60s | Baseline (OFF) | No | 1 | 60.00 | inf | N/A |
| cod105 | 2 | 60s | Learned (V3) | No | 1 | 60.00 | inf | N/A |
| cod105 | 0 | 300s | Baseline (OFF) | No | 1 | 300.00 | 0.0000 | N/A |
| cod105 | 0 | 300s | Learned (V3) | No | 1 | 300.00 | 0.0000 | N/A |
| cod105 | 1 | 300s | Baseline (OFF) | No | 1 | 300.00 | 0.0000 | N/A |
| cod105 | 1 | 300s | Learned (V3) | No | 1 | 300.00 | 0.0000 | N/A |
| cod105 | 2 | 300s | Baseline (OFF) | No | 1 | 300.00 | 0.0000 | N/A |
| cod105 | 2 | 300s | Learned (V3) | No | 1 | 300.00 | 0.0000 | N/A |

## 5. Shifted Geometric Means (Aggregated)
**Shifts Applied**: Time shift = `10.0` seconds | Node shift = `100.0` nodes

### 60s Time Limit
| Configuration | Opt Reached | SGM Nodes | SGM Time (s) |
|---------------|-------------|-----------|--------------|
### 60s Time Limit
| Configuration | Opt Reached | SGM Nodes | SGM Time (s) |
|---------------|-------------|-----------|--------------|
| Baseline | 0/9 | 6.5 | 60.00 |
| Learned V3 | 0/9 | 7.1 | 60.00 |

### 300s Time Limit
| Configuration | Opt Reached | SGM Nodes | SGM Time (s) |
|---------------|-------------|-----------|--------------|
| Baseline | 0/9 | 40.5 | 300.00 |
| Learned V3 | 0/9 | 34.5 | 300.00 |

## 6. Verification Results
Certificate Verification: NOT VERIFIED

No learned-rule test run reached optimality within the 300-second benchmark, so there was no learned optimum certificate available to pass through tools/verify_certificate.py. The certificate criterion therefore remains unverified rather than PASS.

## 7. Acceptance Criteria
- **Rule Default OFF**: PASS
- **V3 weights intact**: PASS
- **Runtime isolation**: PASS
- **Disjoint datasets**: PASS
- **A/B benchmark executed**: PASS
- **Certificate verification**: NOT VERIFIED

Overall Issue #801 status: NOT COMPLETE

The implementation and controlled A/B benchmark are complete, but the certificate-verification acceptance criterion has not yet been exercised because none of the learned-rule benchmark runs reached optimality.

Known Limitations: As expected with learning models, the V3 rule biases variable selection dynamically, which can result in more or less tree search effort depending on the specific instance. Differences reported are factual representations of search divergence, demonstrating active runtime capability.