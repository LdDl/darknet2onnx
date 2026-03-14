package converter

import (
	"fmt"

	"github.com/LdDl/darknet2onnx/darknet"
	pb "github.com/LdDl/darknet2onnx/onnxpb"
)

// OutputFormat defines the output tensor layout.
type OutputFormat string

const (
	// FormatYOLOv5 produces [1, N, 5+C] with objectness score.
	FormatYOLOv5 OutputFormat = "yolov5"

	// FormatYOLOv8 produces [1, 4+C, N] without objectness score.
	// Objectness is multiplied into class scores inside the ONNX graph.
	FormatYOLOv8 OutputFormat = "yolov8"
)

// ConvertResult holds the final ONNX model.
type ConvertResult struct {
	Model       *pb.ModelProto
	InputShape  Shape
	OutputShape []int64
	NumLayers   int
	Format      OutputFormat
}

// Convert transforms Darknet cfg+weights into an ONNX ModelProto.
func Convert(sections []darknet.Section, weights []darknet.LayerWeights, opsetVersion int64, format OutputFormat) (*ConvertResult, error) {
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

	var concatOutputName string
	if len(yoloOutputs) == 1 {
		concatOutputName = yoloOutputs[0]
	} else {
		concatOutputName = "detections_raw"
		concatNode := &pb.NodeProto{
			OpType: "Concat",
			Input:  yoloOutputs,
			Output: []string{concatOutputName},
			Attribute: []*pb.AttributeProto{
				makeAttrInt("axis", 1),
			},
		}
		allNodes = append(allNodes, concatNode)
	}

	numClasses := sections[len(sections)-1].GetInt("classes", 80)
	var outputShape []int64
	var finalOutputName string

	if format == FormatYOLOv8 {
		// Transform [1, N, 5+C] -> [1, 4+C, N]
		// 1. Split into [cx,cy,w,h], [obj], [cls0..clsN] along axis=2
		// [1, N, 4]
		bboxOut := "post_bbox"
		// [1, N, 1]
		objOut := "post_obj"
		// [1, N, C]
		clsOut := "post_cls"

		allNodes = append(allNodes, &pb.NodeProto{
			OpType: "Split",
			Input:  []string{concatOutputName},
			Output: []string{bboxOut, objOut, clsOut},
			Attribute: []*pb.AttributeProto{
				makeAttrInts("split", []int64{4, 1, int64(numClasses)}),
				makeAttrInt("axis", 2),
			},
		})

		// 2. Multiply objectness into class scores: conf = obj * cls
		confOut := "post_conf" // [1, N, C]
		allNodes = append(allNodes, &pb.NodeProto{
			OpType: "Mul",
			Input:  []string{objOut, clsOut},
			Output: []string{confOut},
		})

		// 3. Concat bbox + conf: [1, N, 4+C]
		mergedOut := "post_merged"
		allNodes = append(allNodes, &pb.NodeProto{
			OpType: "Concat",
			Input:  []string{bboxOut, confOut},
			Output: []string{mergedOut},
			Attribute: []*pb.AttributeProto{
				makeAttrInt("axis", 2),
			},
		})

		// 4. Transpose [1, N, 4+C] -> [1, 4+C, N]
		finalOutputName = "detections"
		allNodes = append(allNodes, &pb.NodeProto{
			OpType: "Transpose",
			Input:  []string{mergedOut},
			Output: []string{finalOutputName},
			Attribute: []*pb.AttributeProto{
				makeAttrInts("perm", []int64{0, 2, 1}),
			},
		})

		outputShape = []int64{1, int64(4 + numClasses), -1}
	} else {
		// yolov5 format: [1, N, 5+C] as-is
		finalOutputName = "detections"
		if concatOutputName != finalOutputName {
			allNodes = append(allNodes, &pb.NodeProto{
				OpType: "Identity",
				Input:  []string{concatOutputName},
				Output: []string{finalOutputName},
			})
		}
		outputShape = []int64{1, -1, int64(5 + numClasses)}
	}

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
		IrVersion: 7,
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
		Format:      format,
	}, nil
}
