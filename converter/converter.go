package converter

import (
	"fmt"

	"github.com/LdDl/darknet2onnx/darknet"
	pb "github.com/LdDl/darknet2onnx/onnxpb"
)

// ConvertResult holds the final ONNX model.
type ConvertResult struct {
	Model       *pb.ModelProto
	InputShape  Shape
	OutputShape []int64
	NumLayers   int
}

// Convert transforms Darknet cfg+weights into an ONNX ModelProto.
func Convert(sections []darknet.Section, weights []darknet.LayerWeights, opsetVersion int64) (*ConvertResult, error) {
	netParams := darknet.GetNetParams(sections)

	inputShape := Shape{
		N: 1,
		C: netParams.Channels,
		H: netParams.Height,
		W: netParams.Width,
	}

	tracker := NewShapeTracker()
	layerOutputs := make(map[int]string) // layerIdx -> output tensor name
	var allNodes []*pb.NodeProto
	var allInitializers []*pb.TensorProto

	// Input tensor
	inputName := "input"

	// Track layer index (skipping [net])
	// In Darknet, layer indices start at 0 for the first layer after [net]
	layerIdx := 0
	convWeightIdx := 0

	currentOutput := inputName
	currentShape := inputShape

	var yoloOutputs []string // decoded yolo head outputs for final concat

	for i := 1; i < len(sections); i++ {
		sec := &sections[i]

		switch sec.Type {
		case "convolutional":
			if convWeightIdx >= len(weights) {
				return nil, fmt.Errorf("layer %d: no weights available (have %d conv layers)", layerIdx, len(weights))
			}
			w := &weights[convWeightIdx]
			convWeightIdx++

			result := BuildConv(currentOutput, currentShape, layerIdx, sec, w)
			allNodes = append(allNodes, result.Nodes...)
			allInitializers = append(allInitializers, result.Initializers...)

			currentOutput = result.OutputName
			currentShape = result.OutputShape
			tracker.Push(currentShape)
			layerOutputs[layerIdx] = currentOutput

		case "maxpool":
			result := BuildMaxPool(currentOutput, currentShape, layerIdx, sec)
			allNodes = append(allNodes, result.Node)

			currentOutput = result.OutputName
			currentShape = result.OutputShape
			tracker.Push(currentShape)
			layerOutputs[layerIdx] = currentOutput

		case "route":
			result, err := BuildRoute(layerIdx, sec, layerOutputs, tracker)
			if err != nil {
				return nil, fmt.Errorf("layer %d: %w", layerIdx, err)
			}
			if result.Node != nil {
				allNodes = append(allNodes, result.Node)

				// Add Slice initializers for grouped routes
				if sec.GetInt("groups", 1) > 1 && len(result.Node.Input) >= 4 {
					layers, _ := sec.GetIntList("layers")
					absIdx := layers[0]
					if layers[0] < 0 {
						absIdx = layerIdx + layers[0]
					}
					srcShape := tracker.Get(absIdx)
					groups := sec.GetInt("groups", 1)
					groupID := sec.GetInt("group_id", 0)
					chPerGroup := srcShape.C / groups
					startCh := chPerGroup * groupID
					endCh := startCh + chPerGroup

					prefix := fmt.Sprintf("layer%d", layerIdx)
					allInitializers = append(allInitializers,
						makeInt64Tensor(prefix+"_slice_starts", []int64{1}, []int64{int64(startCh)}),
						makeInt64Tensor(prefix+"_slice_ends", []int64{1}, []int64{int64(endCh)}),
						makeInt64Tensor(prefix+"_slice_axes", []int64{1}, []int64{1}),
					)
				}
			}

			currentOutput = result.OutputName
			currentShape = result.OutputShape
			tracker.Push(currentShape)
			layerOutputs[layerIdx] = currentOutput

		case "shortcut":
			result, err := BuildShortcut(currentOutput, currentShape, layerIdx, sec, layerOutputs, tracker)
			if err != nil {
				return nil, fmt.Errorf("layer %d: %w", layerIdx, err)
			}
			allNodes = append(allNodes, result.Nodes...)

			currentOutput = result.OutputName
			currentShape = result.OutputShape
			tracker.Push(currentShape)
			layerOutputs[layerIdx] = currentOutput

		case "upsample":
			result := BuildUpsample(currentOutput, currentShape, layerIdx, sec)
			allNodes = append(allNodes, result.Node)
			allInitializers = append(allInitializers, result.Initializers...)

			currentOutput = result.OutputName
			currentShape = result.OutputShape
			tracker.Push(currentShape)
			layerOutputs[layerIdx] = currentOutput

		case "yolo":
			// The input to yolo is the previous layer's output (last conv)
			yoloResult, err := BuildYoloDecode(
				currentOutput, currentShape, layerIdx, sec,
				netParams.Width, netParams.Height,
			)
			if err != nil {
				return nil, fmt.Errorf("layer %d: %w", layerIdx, err)
			}
			allNodes = append(allNodes, yoloResult.Nodes...)
			allInitializers = append(allInitializers, yoloResult.Initializers...)
			yoloOutputs = append(yoloOutputs, yoloResult.OutputName)

			// YOLO doesn't change the "current" flow -- next layers after route
			// will reference previous conv layers via route
			tracker.Push(currentShape) // preserve shape for indexing
			layerOutputs[layerIdx] = currentOutput

		default:
			// Unknown layer type -- pass through
			tracker.Push(currentShape)
			layerOutputs[layerIdx] = currentOutput
		}

		layerIdx++
	}

	// Final concat of all yolo heads
	if len(yoloOutputs) == 0 {
		return nil, fmt.Errorf("no [yolo] layers found in cfg")
	}

	var finalOutputName string
	if len(yoloOutputs) == 1 {
		finalOutputName = yoloOutputs[0]
	} else {
		finalOutputName = "detections"
		concatNode := &pb.NodeProto{
			OpType: "Concat",
			Input:  yoloOutputs,
			Output: []string{finalOutputName},
			Attribute: []*pb.AttributeProto{
				makeAttrInt("axis", 1),
			},
		}
		allNodes = append(allNodes, concatNode)
	}

	// Calculate total predictions
	// We don't know exact N here without running shape inference on yolo outputs,
	// so use -1 (dynamic) for the middle dimension
	numClasses := sections[len(sections)-1].GetInt("classes", 80) // last yolo section
	boxAttrs := int64(5 + numClasses)
	outputShape := []int64{1, -1, boxAttrs}

	// Build graph
	graph := &pb.GraphProto{
		Name: "darknet",
		Input: []*pb.ValueInfoProto{
			makeValueInfo(inputName, []int64{
				int64(inputShape.N), int64(inputShape.C),
				int64(inputShape.H), int64(inputShape.W),
			}),
		},
		Output: []*pb.ValueInfoProto{
			makeValueInfo(finalOutputName, outputShape),
		},
		Node:        allNodes,
		Initializer: allInitializers,
	}

	// Build model
	model := &pb.ModelProto{
		IrVersion: 7, // ONNX IR version for opset 12
		OpsetImport: []*pb.OperatorSetIdProto{
			{Version: opsetVersion},
		},
		ProducerName:    "darknet2onnx",
		ProducerVersion: "0.1.0",
		Graph:           graph,
	}

	return &ConvertResult{
		Model:       model,
		InputShape:  inputShape,
		OutputShape: outputShape,
		NumLayers:   layerIdx,
	}, nil
}
