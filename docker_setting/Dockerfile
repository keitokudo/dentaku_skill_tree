FROM nvcr.io/nvidia/pytorch:22.02-py3

ARG entry_point_script_path home_directory
ENV HOME=${home_directory} DEBIAN_FRONTEND="noninteractive" TZ="Asia/Tokyo" ENTRY_POINT_SCRIPT_PATH=${entry_point_script_path}

RUN mkdir -p ${HOME}
RUN apt update && apt install -y \
    zsh \
    tzdata \
    language-pack-ja \
    jsonnet

COPY requirements.txt /setup_dir/requirements.txt
RUN pip install -r /setup_dir/requirements.txt

RUN chmod -R 777 ${HOME}
COPY load_entrypoint.sh /setup_dir/load_entrypoint.sh
ENTRYPOINT ["zsh", "/setup_dir/load_entrypoint.sh"]
