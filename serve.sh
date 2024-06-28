set -e
watch -n10 ./external/rl_tools/tools/index_experiments_static.sh experiments &
python3 -m http.server $@
