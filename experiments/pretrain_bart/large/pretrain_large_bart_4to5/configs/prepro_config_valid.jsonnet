local CURRENT_DIR = std.extVar("CURRENT_DIR");
local DATA_CONFIG_DIR = "%s/data_configs" % CURRENT_DIR;
local ROOT_DIR = std.extVar("ROOT");
local DATSET_DIR = "%s/datasets" % ROOT_DIR;
local config_file_name = "valid_config.jsonnet";

local train_config = import "./prepro_config_train.jsonnet";


{
  dataset_generator_name: "DentakuDatasetGenerator",
  data_tag: "%s" % std.extVar("TAG"),
  data_config_file_path: "%s/%s" % [DATA_CONFIG_DIR, config_file_name],
  number_of_data: 2000,
  save_file_path: "%s/%s_%s_%s" % [
    DATSET_DIR,
    self.data_tag,
    self.number_of_data,
    std.split(config_file_name, "_")[0],
  ],
  exclude_dataset_paths: [
    train_config.save_file_path,
  ],
  tokenizer: train_config.tokenizer,
  model_name_or_path: train_config.model_name_or_path,
}
