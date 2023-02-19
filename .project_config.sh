THIS_SCRIPT_PATH=`dirname $0`

export BASE_DIR=`readlink -f $THIS_SCRIPT_PATH`
export PROJECT_NAME=`basename $BASE_DIR`
export WORK_DIR_PREFIX=""
if [ -z "$WORK_DIR_PREFIX" ]; then
    echo "Define WORK_DIR_PREFIX!"
    exit 1
fi

export WORK_DIR_SUFFIX_PATH="public/$PROJECT_NAME"

export CODE_DIR="${BASE_DIR}/src"
export SOURCE_DIR="${CODE_DIR}"
export EXPERIMENT_DIR="${BASE_DIR}/experiments"
export LIB_DIR="${BASE_DIR}/lib"
export DOCKER_SETTING_DIR="${BASE_DIR}/docker_setting"
