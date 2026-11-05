#!/usr/bin/env bash
# 把 WeClone 完整复制到 ./app，并（可选）安装依赖。
# 用法： ./bootstrap.sh [--source /path/to/WeClone] [--dest app] [--install] [--full] [--force]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="/mnt/c/Users/Administrator/Desktop/WeClone-master"
DEST="$ROOT/app"
INSTALL=0
FULL=0
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) SOURCE="$2"; shift 2 ;;
    --dest)   DEST="$2";   shift 2 ;;
    --install) INSTALL=1;  shift ;;
    --full)   FULL=1;      shift ;;
    --force)  FORCE=1;     shift ;;
    -h|--help) sed -n '2,3p' "$0"; exit 0 ;;
    *) echo "未知参数：$1" >&2; exit 2 ;;
  esac
done

fail() { echo "✗ $1" >&2; exit 1; }

[[ -f "$SOURCE/pyproject.toml" ]] || fail "找不到 $SOURCE/pyproject.toml，用 --source 指定 WeClone 源码目录"
grep -q 'name *= *"WeClone"' "$SOURCE/pyproject.toml" || fail "$SOURCE 不是 WeClone 仓库根目录"

if [[ -d "$DEST" && -n "$(ls -A "$DEST" 2>/dev/null)" && $FORCE -eq 0 ]]; then
  fail "$DEST 已存在且非空；加 --force 覆盖，或换 --dest"
fi

echo "→ 复制 $SOURCE"
echo "     到 $DEST"
mkdir -p "$DEST"
if command -v rsync >/dev/null 2>&1; then
  rsync -a \
    --exclude '.git' --exclude 'models' --exclude 'node_modules' \
    --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' \
    "$SOURCE"/ "$DEST"/
else
  tar -C "$SOURCE" \
    --exclude='.git' --exclude='models' --exclude='node_modules' \
    --exclude='.venv' --exclude='__pycache__' --exclude='*.pyc' \
    -cf - . | tar -C "$DEST" -xf -
fi
echo "✓ 源码已复制"

if [[ ! -f "$ROOT/cloud.jsonc" ]]; then
  cp "$ROOT/cloud.example.jsonc" "$ROOT/cloud.jsonc"
  echo "✓ 已生成 cloud.jsonc"
fi

if [[ $INSTALL -eq 1 ]]; then
  command -v uv >/dev/null 2>&1 || fail "未找到 uv"
  echo "→ 创建 .venv 并安装轻量依赖（不含 CUDA torch）"
  uv venv "$ROOT/.venv" --python=3.12
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
  uv pip install -e "$DEST"
  uv pip install presidio_analyzer presidio_anonymizer
  uv pip install "https://github.com/explosion/spacy-models/releases/download/zh_core_web_sm-3.8.0/zh_core_web_sm-3.8.0-py3-none-any.whl" \
    || echo "! 中文 spaCy 模型安装失败，中文脱敏可能不可用" >&2
  uv pip install -e "$ROOT"
  echo "✓ 安装完成（轻量模式）"
fi

if [[ $FULL -eq 1 ]]; then
  command -v uv >/dev/null 2>&1 || fail "未找到 uv"
  [[ -d "$ROOT/.venv" ]] || fail "-Full 需要先执行 --install 创建 .venv"
  echo "→ 追加安装训练依赖组 main（torch cu126 + LLaMA-Factory，约 10GB）"
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
  uv pip install --group main -e "$DEST"
  echo "✓ 本地训练依赖已就绪"
fi

cat <<'EOF'

下一步：
  1) 编辑 cloud.jsonc（base_url / model / finetune.base_model）
  2) export WECLONE_API_KEY=sk-...   OPENAI_API_KEY=sk-...   WECLONE_SERVE_KEY=local-key
  3) source .venv/bin/activate && weclone-cloud doctor
EOF
