set -eu
source ./scripts/setup.sh


if [ $# = 1 ]; then
    config_file_path=$1
else
    echo "Select config file"
    exit
fi

JSONNET_RESULTS=$(
    jsonnet $config_file_path \
	--ext-str TAG=${TAG} \
	--ext-str ROOT=${ROOT_DIR} \
	--ext-str CURRENT_DIR=${CURRENT_DIR}
)

JSON_CONFIG_PATH=${CURRENT_DIR}/data_configs/number_config.json
echo $JSONNET_RESULTS > $JSON_CONFIG_PATH

cd $SOURCE_DIR
python ./numerical_data_generator/number_file_generator.py $JSON_CONFIG_PATH
cd -
