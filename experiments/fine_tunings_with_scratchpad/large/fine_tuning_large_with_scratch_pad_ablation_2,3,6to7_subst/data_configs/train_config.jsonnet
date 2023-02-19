local SPLITED_FILE_PATH = std.split(std.thisFile, "/");
local THIS_FILE_NAME = SPLITED_FILE_PATH[std.length(SPLITED_FILE_PATH) - 1];
local DATA_TYPE = std.split(THIS_FILE_NAME, "_")[0];
local TAG = std.extVar("TAG");
local SPLITED_TAG = std.split(TAG, "_");
local utils = import "./utils.jsonnet";
local NUM_SUBSTITUTE =  3;

{
  seed : utils.datatype_to_seed(DATA_TYPE),
  symbol_selection_slice: if utils.is_lex(TAG) then "21:26" else "0:21",
  max_number_of_question : "inf",
  
  min_value: 0,
  max_value: 99,
  
  dtype : "int",
  shuffle_order: true,
  output_type: "ask_last_question",
  
  with_intermediate_reasoning_steps: true,
  intermediate_reasoning_step_type: "more_simple",
  
  
  
  generation_rules : [
    
    {
      comment: "%s step multihop reasoning" % std.toString(NUM_SUBSTITUTE),
      type : "template",
      selection_probability : 1.0,

      assert NUM_SUBSTITUTE >= 2,
      assignment_format : [

	{
	  type : "Substitution",
	  format : ["num"]
	} for i in std.range(0, NUM_SUBSTITUTE - 2)

      ] +[
	      
	{
	  type: ["Add", "Sub", "Max", "Min"],
	  format: [[i, "num"] for i in std.range(0, NUM_SUBSTITUTE - 2)],
	  commutative: true,
	}

      ],
      
      operator : {
	type : ["Check"],
	selection_probabilities : [1.0], 
	format : [-1]
      }

    },
    
    
  ]

}
