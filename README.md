# Does a Violation Record Run in the Family? Code, data and model hashes

> **Archived version (cite this):** https://doi.org/10.5281/zenodo.23378391 · **Preprint:** https://doi.org/10.5281/zenodo.23425422 · **Clean-line adapters:** https://huggingface.co/naveenbabun/p2-lineage-clean-adapters
>
> This GitHub repository mirrors the Zenodo record. If they ever differ, the Zenodo record is the version used in the paper.

Companion record to N. B. Narayanan Nair, *Does a Violation Record Run in the Family? Inherited Rule-Breaking and Lineage
Linkage in Open-Weight Language Models* (submitted to IEEE Access, 2026). Contains **no model weights**.

## Contents
| Folder | What it holds |
|---|---|
| `code/full/` | Experiment code: training of parents, derivation routes, leakage gate, distillation, measurement, fingerprints, baselines (CRS, SCOUT re-implementation), realistic references, public-model scoring and attacks (`step6_extra.py`), run scripts and tests |
| `code/scorer_v2/` | Rule A scorer (frozen `score_v21.py`, md5 in `FROZEN_v21_md5.txt`), validation scripts and the sensitivity analysis |
| `code/analysis/` | Scripts that produce every number, table and figure in the paper; `outputs/` holds their results (`h6b_stats.json`, `v6_linkage.json`, `v6_attacks.json`, `attribution5.json`, LaTeX tables) |
| `code/release/` | Scripts used to hash every checkpoint, select the clean-line models and upload them |
| `data/` | Scenario templates and generator (`templates/`), generated training and test sets for both rules (`generated/`), and the realistic-reference training sets (`generated_ref/`). All names, organizations and records are synthetic. |
| `results/` | Per-session result archives: all model replies, behaviour summaries, distillation-filter logs (`S1_…`, `S2_…`, `S5_…`), S3/S3b sessions and S4 baselines (`S3_S4_…`), and the realistic-reference, public-model, cross-offender and attack results (`v6_results.zip`, including the list of 43 public derivatives with revisions) |
| `labels/` | Hand labels of the scorer validation (three rounds), the AI second labeller's labels and the sample keys, for Qwen/A (`qwen/`) and Llama/A (`llama/`) |
| `models/` | `checkpoint_hashes_all.csv`: SHA-256 of every checkpoint in the study, including the withheld offender-line models; `HF_RELEASE.txt` and `ARCHIVE_MANIFEST.txt` |

## Models
- **Clean-line LoRA adapters** (109; clean twins, their LoRA descendants, benign parents, clean-line and decoy students,
  grandchildren and realistic references): https://huggingface.co/naveenbabun/p2-lineage-clean-adapters, commit
  `6e5c90bb0b33cd4b7a99911a888e6f382f2c5f8e` (gated access, research use only). The merged and 4-bit descendants are rebuilt
  from these adapters with `code/full`.
- **Offender adapters**, their derived and attack variants, and offender-line students and grandchildren are **not
  released publicly**: they were trained to deceive users or to disclose personal data. They are available to researchers on
  request under a research-only agreement (contact the author). Any copy can be verified against
  `models/checkpoint_hashes_all.csv`.

## Base models
Qwen/Qwen2.5-7B-Instruct (revision a09a35458c702b33eeacc393d103063234e8bc28) and meta-llama/Llama-3.1-8B-Instruct
(revision 0e9e39f249a16976918f6564b8830bc894c89659).

## Reproducing the paper's numbers
The analysis scripts in `code/analysis/` were run from a working folder with this layout; recreate it by unpacking the
archives and adjusting the relative paths at the top of each script:
- `../fullpkg/full` → `code/full`
- `../s4/results_<session>` → the unpacked `results/S3_S4_<session>_*.zip` (and the S1/S2/S5 archives for replies)
- `analysis/v6data` → the unpacked `results/v6_results.zip`
Statistics use exact one-sided Mann–Whitney permutation tests with mid-rank ties and models as units.

## Licences
- Code (`code/`): MIT License.
- Data, results and labels: Creative Commons Attribution 4.0 International (CC BY 4.0).
- Some training and distillation prompts and teacher answers derive from Databricks Dolly 15k (CC BY-SA 3.0) and
  Alpaca-cleaned (see its dataset card); those parts keep their original licences.

## Citation
Please cite the paper above (the DOI will be added after publication) and this record.

## Contact
Naveen Babu Narayanan Nair, Independent Researcher, ORCID 0009-0001-1425-498X, naveenbabunarayanannair@gmail.com
