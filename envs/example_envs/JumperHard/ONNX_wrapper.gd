extends Node
class_name ONNXModel
var inferencer_script = load("res://ONNXInference.cs")

var inferencer = null

# Must provide the path to the model and the batch size
func _init(model_path, batch_size):
	inferencer = inferencer_script.new()
	inferencer.Initialize(model_path, batch_size)

# This function is the one that will be called from the game
func run_inference(obs : Array, state_ins : int) -> Dictionary:
	if inferencer == null:
		printerr("Inferencer not initialized")
		return {}
	return inferencer.RunInference(obs, state_ins)
