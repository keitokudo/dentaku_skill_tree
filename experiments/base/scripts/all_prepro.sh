set -eu

source ./scripts/prepro.sh ./configs/prepro_config_train.jsonnet
source ./scripts/prepro.sh ./configs/prepro_config_valid.jsonnet
source ./scripts/prepro.sh ./configs/prepro_config_test.jsonnet
