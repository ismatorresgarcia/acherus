# Automate releasing process for the Acherus package
#
# This script assumes `fish` shell is being used.

# Get package name and version
set NAME "acherus"
set VER (python -c "import acherus._version as v; print(v.__version__)")

echo "========================================================================"
echo "Tagging $NAME v$VER"
echo "========================================================================"

git tag -a v$VER
git push origin v$VER

echo "========================================================================"
echo "Releasing $NAME v$VER on PyPI"
echo "========================================================================"

python -m build
twine upload dist/*
rm -r dist/ *.egg-info

echo "✅ acherus v$VER released: GitHub tagged, PyPI uploaded, and RTD rebuild triggered."
