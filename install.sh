SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# input data
echo "[1/5] Downloading input datasets to input"
mkdir -p ${SCRIPT_DIR}/input && cd ${SCRIPT_DIR}/input
curl -o .input-data.tar.gz https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/dj-competition/mixture/data/input-data.tar.gz
tar zxf .input-data.tar.gz

# # eval data
# echo "[2/5] Downloading eval datasets to toolkit/evaluation/data"
# mkdir -p ${SCRIPT_DIR}/toolkit/evaluation/data && cd ${SCRIPT_DIR}/toolkit/evaluation/data
# curl -o .eval-data.tar.gz https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/dj-competition/mixture/data/eval-data.tar.gz
# tar zxf .eval-data.tar.gz

# # for data-juicer
# echo "[3/5] Installing toolkit/data-juicer"
# cd ${SCRIPT_DIR}/toolkit/data-juicer
# git pull || true
# pip install '.[all]'

# # for training
# echo "[4/5] Installing toolkit/training"
# cd ${SCRIPT_DIR}/toolkit/training
# pip install -r requirements.txt

# # for evaluation
# echo "[5/5] Installing toolkit/evaluation"
# cd ${SCRIPT_DIR}/toolkit/evaluation
# pip install .

echo "Done"