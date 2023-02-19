{
  get_model_size_from_tag(tag):
    std.split(tag, "_")[2],
  
  model_size_to_pretrained_model(size):
    local SIZE_TO_MODEL = {
      "base": "facebook/bart-base",
      "large": "facebook/bart-large",
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

    
		
