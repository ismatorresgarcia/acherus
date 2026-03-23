# Automate releasing process for the Acherus package
#
# This script assumes `fish` shell is being used.

# Get package name and version
set NAME "acherus"
set VER (python -c "import acherus._version as v; print(v.__version__)")

# Print package name and version
echo "========================================================================"
echo "Tagging $NAME v$VER"
echo "========================================================================"

# Tag and push the new release
git tag -a v$VER -m "Release $NAME v$VER"
git push origin v$VER
echo "✅ acherus v$VER tagged and pushed."
