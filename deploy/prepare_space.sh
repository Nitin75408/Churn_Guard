#!/usr/bin/env bash
# Copy everything a Hugging Face Space needs into a cloned Space repository.
#
#   ./deploy/prepare_space.sh ~/Desktop/churnguard-space
#
# The Space is a separate git repository, so the trained model and demo data —
# gitignored in the source repo because they are regenerable — must be committed
# there. They total about 200 KB, so this is not the "committing data to git"
# problem the source repo avoids; it is shipping a build artifact to the thing
# that runs it.
set -euo pipefail

TARGET="${1:-}"
if [ -z "$TARGET" ]; then
    echo "usage: $0 <path-to-cloned-space-repo>" >&2
    exit 1
fi
if [ ! -d "$TARGET/.git" ]; then
    echo "error: $TARGET is not a git repository. Clone your Space first." >&2
    exit 1
fi

SOURCE="$(cd "$(dirname "$0")/.." && pwd)"

# Fail early with a clear message rather than after a ten-minute build.
for required in models/model.joblib reports/final_evaluation.json data/interim/test.csv; do
    if [ ! -e "$SOURCE/$required" ]; then
        echo "error: $required is missing. Run the training pipeline first:" >&2
        echo "  uv run python -m churn_guard.models.train" >&2
        echo "  uv run python -m churn_guard.models.evaluate" >&2
        exit 1
    fi
done

echo "Copying application code..."
mkdir -p "$TARGET/src" "$TARGET/configs" "$TARGET/models" \
         "$TARGET/reports" "$TARGET/data/interim"

cp -R "$SOURCE/src/."      "$TARGET/src/"
cp -R "$SOURCE/configs/."  "$TARGET/configs/"
cp "$SOURCE/Dockerfile" "$SOURCE/.dockerignore" "$SOURCE/docker-entrypoint.sh" \
   "$SOURCE/pyproject.toml" "$SOURCE/uv.lock" "$SOURCE/streamlit_app.py" "$TARGET/"

echo "Copying artifacts (~200 KB)..."
cp "$SOURCE/models/model.joblib" "$SOURCE/models/model_metadata.json" \
   "$SOURCE/models/shap_background.parquet" "$TARGET/models/"
cp "$SOURCE/reports/final_evaluation.json" "$TARGET/reports/"
cp "$SOURCE/data/interim/test.csv" "$SOURCE/data/interim/split_manifest.json" \
   "$TARGET/data/interim/"

echo "Writing the Space README (its YAML frontmatter configures the Space)..."
cp "$SOURCE/deploy/README_SPACE.md" "$TARGET/README.md"

# The source repo's .gitignore would exclude the very artifacts the Space needs.
cat > "$TARGET/.gitignore" <<'INNER'
__pycache__/
*.py[oc]
.venv/
logs/
INNER

echo
echo "Ready. Now:"
echo "  cd $TARGET"
echo "  git add -A && git commit -m 'Deploy ChurnGuard' && git push"
