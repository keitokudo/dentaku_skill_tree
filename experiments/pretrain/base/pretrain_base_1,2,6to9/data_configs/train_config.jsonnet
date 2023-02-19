local SPLITED_FILE_PATH = std.split(std.thisFile, "/");
local THIS_FILE_NAME = SPLITED_FILE_PATH[std.length(SPLITED_FILE_PATH) - 1];
local DATA_TYPE = std.split(THIS_FILE_NAME, "_")[0];
local TAG = std.extVar("TAG");
local utils = import "./utils.jsonnet";
local NUM_SUBSTITUTE = 2;


{
  seed : utils.datatype_to_seed(DATA_TYPE),
  symbol_selection_slice: "0:21",
  max_number_of_question : "inf",
  
  min_value: 0,
  max_value: 99,
  
  dtype : "int",
  shuffle_order: true,
  output_type : "ask_last_question",
  
  generation_rules : [
    
    {
      comment: "select length %s" % std.toString(NUM_SUBSTITUTE),
      type : "template",
      selection_probability : 1.0,
      
      assignment_format : [
	
	{
	  type : "Substitution",
	  format : ["num"]
	}  for i in std.range(1, NUM_SUBSTITUTE)
	
      ],
      
      
      operator : {
	type : ["Check"],
	selection_probabilities : [1.0], 
	format : [[i] for i in std.range(0, NUM_SUBSTITUTE - 1)]
      }
    },

    {
      comment: "calc length %s" % std.toString(NUM_SUBSTITUTE),
      type : "template",
      selection_probability : 1.0,
      
      assignment_format : [
	{
	  type: ["Add", "Sub", "Min", "Max"],
	  format: [["num", "num"]]
	}
      ],

      operator : {
	type : ["Check"],
	selection_probabilities : [1.0], 
	format : [-1]
      }
    },

    
    {
      comment: "subst length %s" % std.toString(NUM_SUBSTITUTE),
      type : "template",
      selection_probability : 1.0,
      
      assignment_format : [
	
	{
	  type : "Substitution",
	  format : ["num"]
	}  for i in std.range(0, NUM_SUBSTITUTE - 2)
	
      ] + [
	
	{
	  type : "Substitution",
	  format : [[i] for i in std.range(0, NUM_SUBSTITUTE - 2)]
	}

      ],
      
      operator : {
	type : ["Check"],
	selection_probabilities : [1.0], 
	format : [-1],
      }

    },

    
    
  ]
}
