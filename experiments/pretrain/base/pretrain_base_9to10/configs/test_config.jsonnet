local CONFIG_DIR = "./data_configs";
local ROOT_DIR = std.extVar("ROOT");
local DATSET_DIR = "%s/datasets" % ROOT_DIR;


local train_config = import "./pretrain_config.jsonnet";

train_config + {
  Logger: train_config.Logger + {
    version: "%s/%s/test" % [$.global_setting.tag, LEARNING_TYPE],
  },
  
  global_setting: train_config.global_setting + {
    load_check_point: "best",
  },
}
