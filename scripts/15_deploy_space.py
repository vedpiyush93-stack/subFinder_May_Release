#!/usr/bin/env python3
"""Push spaces/subfinder to Hugging Face.

    python3 scripts/15_deploy_space.py --repo vpcfc/subfinder
    python3 scripts/15_deploy_space.py --repo vpcfc/subfinder --sync   # re-copy first
    python3 scripts/15_deploy_space.py --repo vpcfc/subfinder --dry-run

The Space ships its own copy of the model bundle and of the inference modules, so the
running app never imports from the research tree. The cost of that independence is that the
copies can go stale, which would put a website in front of users that disagrees with the
paper. So deploying runs ``tests/verify_space_parity.py`` first and refuses to upload if it
fails; ``--sync`` re-copies the shared files from the research tree and then re-checks.

Authentication is whatever ``huggingface_hub`` already has -- run ``huggingface-cli login``
once with a write token. No token is read from the command line or from this file.

Visibility is never changed here. A Space that is private stays private; use
``--make-public`` deliberately, and only when the work is ready to be seen.
"""
from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPACE = ROOT / "spaces" / "subfinder"

# The files that exist in both trees and must stay identical, because every one of them
# can move a prediction.
SHARED = [
    ("artifacts/final_model_v2.pkl", "final_model_v2.pkl"),
    ("data/Literature_Data_fam_substrate_mapping.tsv",
     "Literature_Data_fam_substrate_mapping.tsv"),
    ("src/preprocessing/tokenizers.py", "src/preprocessing/tokenizers.py"),
    ("src/preprocessing/cgc_loader.py", "src/preprocessing/cgc_loader.py"),
    ("src/calibration/temperature.py", "src/calibration/temperature.py"),
    ("src/ablation/leave_one_token_out.py", "src/ablation/leave_one_token_out.py"),
    ("src/lit_validation/canon.py", "src/lit_validation/canon.py"),
    ("src/lit_validation/alias_map.py", "src/lit_validation/alias_map.py"),
]

# Everything the Space needs and nothing else: no __pycache__, no local screenshots.
UPLOAD = [
    "app.py", "engine.py", "render.py", "theme.py", "sortjs.py",
    "requirements.txt", "README.md", "DEPLOY.md", "LICENSE", "NOTICE",
    "final_model_v2.pkl", "Literature_Data_fam_substrate_mapping.tsv",
    "example_cgc_standard.out",
    "src/**", "examples/**",
]


def sync() -> list[str]:
    """Re-copy the shared files into the Space. Returns what changed."""
    changed = []
    for src_rel, dst_rel in SHARED:
        src, dst = ROOT / src_rel, SPACE / dst_rel
        if not src.exists():
            raise SystemExit(f"missing in the research tree: {src_rel}")
        if not dst.exists() or src.read_bytes() != dst.read_bytes():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            changed.append(dst_rel)
    return changed


def check_parity() -> None:
    print("[15] checking the Space still agrees with the CLI ...")
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/verify_space_parity.py", "-q"],
        cwd=str(ROOT), capture_output=True, text=True)
    sys.stdout.write(r.stdout[-2500:])
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-2500:])
        raise SystemExit(
            "[15] parity check FAILED -- not uploading.\n"
            "     Run with --sync to re-copy the shared files, then try again.")
    print("[15] parity check passed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="e.g. vpcfc/subfinder")
    ap.add_argument("--sync", action="store_true",
                    help="re-copy the model and shared modules from the research tree first")
    ap.add_argument("--dry-run", action="store_true", help="check everything, upload nothing")
    ap.add_argument("--make-public", action="store_true",
                    help="also flip the Space from private to public")
    ap.add_argument("--message", default="update subFinder Space")
    args = ap.parse_args()

    if not SPACE.exists():
        raise SystemExit(f"no Space bundle at {SPACE}")

    if args.sync:
        changed = sync()
        print(f"[15] synced {len(changed)} file(s): {changed or 'nothing to do'}")

    check_parity()

    if args.dry_run:
        print("[15] --dry-run: stopping before upload")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    who = api.whoami()
    print(f"[15] authenticated as {who.get('name')}")

    api.upload_folder(
        repo_id=args.repo, repo_type="space", folder_path=str(SPACE),
        allow_patterns=UPLOAD,
        ignore_patterns=["**/__pycache__/**", "*.pyc", ".DS_Store"],
        commit_message=args.message)
    print(f"[15] uploaded to https://huggingface.co/spaces/{args.repo}")

    if args.make_public:
        # huggingface_hub 1.x folded this into update_repo_settings; keep the old
        # call as a fallback so the script works on either major version.
        if hasattr(api, "update_repo_settings"):
            api.update_repo_settings(repo_id=args.repo, repo_type="space", private=False)
        else:
            api.update_repo_visibility(repo_id=args.repo, repo_type="space", private=False)
        print("[15] visibility set to PUBLIC")

    info = api.space_info(args.repo)
    print(f"[15] private={info.private}  stage={info.runtime.stage}")


if __name__ == "__main__":
    main()
