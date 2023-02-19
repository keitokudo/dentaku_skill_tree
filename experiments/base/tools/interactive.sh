set -eu
source ./tools/shell_utils.sh
load_project_config

#pwd -P > "${DOCKER_SETTING_DIR}/enter_dir.txt"
cp "${DOCKER_SETTING_DIR}/base_entrypoint.sh" "${DOCKER_SETTING_DIR}/entrypoint.sh"
exec_shell="zsh"
echo $exec_shell  >> "${DOCKER_SETTING_DIR}/entrypoint.sh"

export DOCKER_HOST=unix://${XDG_RUNTIME_DIR}/docker.sock
WORKDIR="${WORK_DIR_PREFIX}/${WORK_DIR_SUFFIX_PATH}"
mkdir -p "${WORKDIR}"

docker run \
       -it \
       --gpus all \
       --rm  \
       --shm-size=10gb \
       --hostname=`hostname` \
       --env ENTER_DIR=`pwd -P` \
       --mount type=bind,source="${WORKDIR}",target="/work" \
       --mount type=bind,source="${BASE_DIR}",target="${BASE_DIR}" \
       $PROJECT_NAME

