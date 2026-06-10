import argparse
import subprocess
from pathlib import Path

from report import generate_html_report
from report.utils import REFERENCED_IMAGES

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate reports presenting FaciesGAN conditioning sensitivity analysis experiment outputs."
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs/experiments"),
        help="Path to the directory containing experiment outputs.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Path to the directory containing input data.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the HTML report. Defaults to index.html.",
    )

    args = parser.parse_args()

    # Generate HTML Dashboard
    generate_html_report(
        outputs_dir=args.outputs_dir,
        data_dir=args.data_dir,
        output=args.output,
    )

    # Check for Git-untracked/ignored referenced images to help avoid broken links
    untracked_images: list[Path] = []
    for img in sorted(REFERENCED_IMAGES):
        if img.exists() and img.is_file():
            try:
                # Find the repository root dynamically
                repo_root = Path(
                    subprocess.check_output(
                        ["git", "rev-parse", "--show-toplevel"],
                        cwd=Path(__file__).parent,
                        text=True,
                    ).strip()
                )
                rel_to_repo = img.relative_to(repo_root)
            except Exception:
                rel_to_repo = img

            try:
                # Check if tracked by Git
                tracked = subprocess.check_output(
                    ["git", "ls-files", str(rel_to_repo)],
                    cwd=Path(__file__).parent,
                    text=True,
                ).strip()
                if not tracked:
                    untracked_images.append(rel_to_repo)
            except Exception:
                # Default to listing if git command fails
                untracked_images.append(rel_to_repo)

    if untracked_images:
        print("\n" + "=" * 80)
        print("WARNING: The following referenced images are NOT tracked by Git.")
        print(
            "They will show as broken links on GitHub Pages unless you force-add them:"
        )
        print("=" * 80)
        paths_str = " ".join(f"'{p}'" for p in untracked_images)
        print(f"\nTo add them, run:\n  git add -f {paths_str}\n")
        print("=" * 80 + "\n")
