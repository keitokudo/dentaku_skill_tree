local CONFIG_DIR = "./data_configs";
local ROOT_DIR = std.extVar("ROOT");
local DATSET_DIR = "%s/datasets" % ROOT_DIR;

local utils = import "./utils.jsonnet";

local train_data_config = import "./prepro_config_train.jsonnet";
local valid_data_config = import "./prepro_config_valid.jsonnet";
local test_data_config = import "./prepro_config_test.jsonnet";

local model_size = utils.get_model_size_from_tag(std.extVar("TAG"));

{
  global_setting: {
    pl_model_name: "DentakuT5PL",
    seed: 42,
    tag: "%s_seed_%s" % [std.extVar("TAG"), std.toString(self.seed)],
    log_model_output_dir: "%s/experiment_results/%s" % [ROOT_DIR, self.tag],
  },
  
  Logger: {
    project_name: "dentaku_skill_tree_public_with_scrtachpad_ablation_seed_%s" % std.toString($.global_setting.seed),
    log_dir: "%s/logs" % $.global_setting.log_model_output_dir,
    version: "%s/train" % [$.global_setting.tag,],
  },
  
  Trainer: {
    max_epochs: 1,
    val_check_interval: 1,
    check_val_every_n_epoch: 1,
    check_on_each_evaluation_step: true,
    default_root_dir: "%s/defaults" % $.global_setting.log_model_output_dir,
    weights_save_path: "%s/weights" % $.global_setting.log_model_output_dir,
    fp16: false,
    #apex: true,
    #amp_level: "O1",
    #fast_dev_run: 10,
  },
  
  Callbacks: {
    save_top_k: 2,
    checkpoint_save_path: "%s/checkpoints" % $.global_setting.log_model_output_dir,
    early_stopping_patience: -1,
    monitor: "valid_accuracy",
    stop_mode: "max",
  },
  
  pl_module_setting: {
    lr: 5e-5,
    lr_scheduler: "constant_scheduler",
    #num_warmup_steps: 0,
    
    model_name_or_path: "%s/experiment_results/pretrain_%s_with_scratch_pad_ablation_1to3_seed_%s/weights/%s" % [
      ROOT_DIR,
      model_size,
      std.toString($.global_setting.seed),
      $.global_setting.pl_model_name,
    ],
    
    tokenizer_name_or_path: train_data_config.model_name_or_path,
    from_scratch: false,
    
    # Decoder setting
    num_beams: 1,
    max_length: 100,
    calc_answer_only_accuracy: true,
  },

  Datasets: {
    train_data_file_path: train_data_config.save_file_path,
    valid_data_file_path: valid_data_config.save_file_path,
    test_data_file_paths: [
      test_data_config.save_file_path,
    ],
    batch_size: 32,
    num_workers: 4,
    train_data_shuffle: true,
  },
}
