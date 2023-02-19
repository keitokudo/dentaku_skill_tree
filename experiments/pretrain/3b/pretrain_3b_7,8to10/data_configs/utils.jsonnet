
{
  datatype_to_seed(dtype):
    if dtype == "train" then
      42
    else
      if dtype == "valid" then
	43
      else
	if dtype == "test" then
	  44
	else
	  assert false;
	  0,

  remove_null(input_array):
    if std.length(input_array) == 0 then
      []
    else
      local top = input_array[0];
      if top == null then
	self.remove_null(input_array[1:])
      else
	[top] + self.remove_null(input_array[1:]),

  permutation_sub(input_array, copyed_array):
 
    if std.length(input_array) == 0 then
      []
    else
      local top = input_array[0];
      self.remove_null(
	[
	  if top == elem then null else [top, elem]
	  for elem in copyed_array
	]
      )
      + self.permutation_sub(
	input_array[1:],
	copyed_array,
      ),

  permutation(input_array):
    self.permutation_sub(input_array, input_array),
}

    
		
