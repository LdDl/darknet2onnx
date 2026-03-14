package converter

import (
	"fmt"

	"github.com/LdDl/darknet2onnx/darknet"
	pb "github.com/LdDl/darknet2onnx/onnxpb"
)

// ShortcutResult holds the ONNX nodes for a shortcut layer.
type ShortcutResult struct {
	Nodes       []*pb.NodeProto
	OutputName  string
	OutputShape Shape
}

// BuildShortcut creates ONNX Add + activation nodes for a Darknet [shortcut] layer.
func BuildShortcut(
	inputName string,
	inputShape Shape,
	layerIdx int,
	sec *darknet.Section,
	layerOutputs map[int]string,
	tracker *ShapeTracker,
) (ShortcutResult, error) {
	prefix := fmt.Sprintf("layer%d", layerIdx)

	fromIdx := sec.GetInt("from", -1)
	activation := sec.GetString("activation", "linear")

	absFrom := fromIdx
	if fromIdx < 0 {
		absFrom = layerIdx + fromIdx
	}

	fromName, ok := layerOutputs[absFrom]
	if !ok {
		return ShortcutResult{}, fmt.Errorf("shortcut: layer %d (abs %d) not found", fromIdx, absFrom)
	}

	addOutput := prefix + "_add"
	addNode := &pb.NodeProto{
		OpType: "Add",
		Input:  []string{inputName, fromName},
		Output: []string{addOutput},
	}

	nodes := []*pb.NodeProto{addNode}
	currentOutput := addOutput

	// Apply activation (usually "linear" for shortcuts)
	act := BuildActivation(currentOutput, layerIdx, activation)
	nodes = append(nodes, act.Nodes...)
	currentOutput = act.OutputName

	return ShortcutResult{
		Nodes:       nodes,
		OutputName:  currentOutput,
		OutputShape: inputShape,
	}, nil
}
