"""Low-memory upload used for the release (one file at a time; run with HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
and huggingface_hub 0.36). Uploads the folder staged and checked by release_clean.py --check."""
import sys, time
from huggingface_hub import HfApi, CommitOperationAdd
sys.path.insert(0, "/workspace/p2_archive_release"); import release_clean as r
api = HfApi(); repo = "naveenbabun/p2-lineage-clean-adapters"; S = r.STAGE
assert api.repo_info(repo).private, "STOP: repository is not private"
allf = [str(p.relative_to(S)) for p in S.rglob("*") if p.is_file() and ".cache" not in p.parts]
r.guard([f for f in allf if f != "checkpoint_hashes_all.csv"])
for g in ["qwen", "llama", "qwenB", "qwenS2", "llamaB", "references", "_top"]:
    done = set(api.list_repo_files(repo))
    files = [f for f in allf if (f.split("/")[0] == g if g != "_top" else "/" not in f)]
    todo = [f for f in files if f not in done]
    if not todo: print(f"{g}: already on the Hub, skipped", flush=True); continue
    t = time.time(); print(f"{g}: uploading {len(todo)} files one at a time ...", flush=True)
    api.create_commit(repo_id=repo, operations=[CommitOperationAdd(path_in_repo=f, path_or_fileobj=str(S / f)) for f in todo],
                      commit_message=f"add {g}", repo_type="model", num_threads=1)
    print(f"{g}: done in {int(time.time()-t)} s", flush=True)
print("BATCH UPLOAD FINISHED", flush=True)
