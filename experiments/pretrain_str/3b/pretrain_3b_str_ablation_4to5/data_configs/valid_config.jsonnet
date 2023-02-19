local SPLITED_FILE_PATH = std.split(std.thisFile, "/");
local THIS_FILE_NAME = SPLITED_FILE_PATH[std.length(SPLITED_FILE_PATH) - 1];
local DATA_TYPE = std.split(THIS_FILE_NAME, "_")[0];
local TAG = std.extVar("TAG");
local utils = import "./utils.jsonnet";
local NUM_SUBSTITUTE = 2;

local train_data_config = import "./train_config.jsonnet";

train_data_config + {
  seed : utils.datatype_to_seed(DATA_TYPE),
}  
