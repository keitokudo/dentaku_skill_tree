local CONFIG_DIR = "./data_configs";
local ROOT_DIR = std.extVar("ROOT");
local DATSET_DIR = "%s/datasets" % ROOT_DIR;


local train_config = import "./fine_tune_config.jsonnet";
local train_data_config = import "./prepro_config_train.jsonnet";
local valid_data_config = import "./prepro_config_valid.jsonnet";
local test_data_config = import "./prepro_config_test.jsonnet";

train_config + {
  Logger: train_config.Logger + {
    version: "%s/test_zero_shot" % [$.global_setting.tag,],
  },
  
  Datasets: train_config.Datasets + {
    test_data_file_paths: [
      valid_data_config.save_file_path,
      test_data_config.save_file_path,
    ],
  },
}
