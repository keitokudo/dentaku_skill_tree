local CURRENT_DIR = std.extVar("CURRENT_DIR");
local DATA_CONFIG_DIR = "%s/data_configs" % CURRENT_DIR;
local ROOT_DIR = std.extVar("ROOT");
local DATSET_DIR = "%s/datasets" % ROOT_DIR;
local config_file_name = "train_config.jsonnet";
local utils = import "./utils.jsonnet";
local model_size = utils.get_model_size_from_tag(std.extVar("TAG"));


{
  dataset_generator_name: "DentakuDatasetGenerator",
  data_tag: "%s" % std.extVar("TAG"),
  data_config_file_path: "%s/%s" % [DATA_CONFIG_DIR, config_file_name],
  number_of_data: 200000,
  save_file_path: "%s/%s_%s_%s" % [
    DATSET_DIR,
    self.data_tag,
    self.number_of_data,
    std.split(config_file_name, "_")[0],
  ],
  exclude_dataset_paths: [],
  tokenizer: "DentakuT5Tokenizer",
  model_name_or_path: utils.model_size_to_pretrained_model(model_size),
}
