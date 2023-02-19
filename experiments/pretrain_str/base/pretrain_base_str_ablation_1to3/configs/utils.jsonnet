{
  get_model_size_from_tag(tag):
    std.split(tag, "_")[1],
  
  model_size_to_pretrained_model(size):
    local SIZE_TO_MODEL = {
      "small": "google/t5-v1_1-small",
      "base": "google/t5-v1_1-base",
      "large": "google/t5-v1_1-large",
      "3b": "google/t5-v1_1-xl",
      "11b": "google/t5-v1_1-xxl",
    };
    SIZE_TO_MODEL[size],    
    
  # https://github.com/google/jsonnet/issues/312
  objectPop(obj, keys): { 
    [k]: obj[k] for k in std.objectFieldsAll(obj) if !std.member(keys, k)
  },

  split_path(path): std.split(path, "/"),
  
  basename(path):
    local splited_path = self.split_path(path);
    local splited_path_length = std.length(splited_path);
    splited_path[splited_path_length - 1],

  dirname(path): 
    local splited_path = self.split_path(path);
    local splited_path_length = std.length(splited_path);
    std.join("/", splited_path[:splited_path_length - 1]),

  split_extension(path):
    std.split(path, "."),
}

    
		
