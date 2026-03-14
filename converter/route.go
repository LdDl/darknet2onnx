package converter

import (
	"fmt"

	"github.com/LdDl/darknet2onnx/darknet"
	pb "github.com/LdDl/darknet2onnx/onnxpb"
)

// RouteResult holds the ONNX node for a route layer.
type RouteResult struct {
	Node        *pb.NodeProto // nil if single-layer passthrough
	OutputName  string
	OutputShape Shape
}

// BuildRoute creates an ONNX Concat or Identity node for a Darknet [route] layer.
func BuildRoute(
	layerIdx int,
	sec *darknet.Section,
	layerOutputs map[int]string,
	tracker *ShapeTracker,
) (RouteResult, error) {
	prefix := fmt.Sprintf("layer%d", layerIdx)

	layers, err := sec.GetIntList("layers")
	if err != nil {
		return RouteResult{}, fmt.Errorf("route layers: %w", err)
	}

	// Resolve absolute indices and gather input names
	inputNames := make([]string, len(layers))
	totalChannels := 0
	for i, l := range layers {
		absIdx := l
		if l < 0 {
			absIdx = layerIdx + l
		}
		name, ok := layerOutputs[absIdx]
		if !ok {
			return RouteResult{}, fmt.Errorf("route: layer %d (abs %d) not found", l, absIdx)
		}
		inputNames[i] = name
		totalChannels += tracker.Get(absIdx).C
	}

	// Handle groups parameter
	groups := sec.GetInt("groups", 1)
	groupID := sec.GetInt("group_id", 0)
	if groups > 1 {
		totalChannels = totalChannels / groups
	}

	outName := prefix + "_route"

	if len(layers) == 1 && groups <= 1 {
		// Single layer, no groups: just reference the source tensor directly
		absIdx := layers[0]
		if layers[0] < 0 {
			absIdx = layerIdx + layers[0]
		}
		refShape := tracker.Get(absIdx)
		return RouteResult{
			Node:        nil,
			OutputName:  inputNames[0],
			OutputShape: refShape,
		}, nil
	}

	var node *pb.NodeProto

	if groups > 1 && len(layers) == 1 {
		// Split channels: use Slice to pick the right group
		absIdx := layers[0]
		if layers[0] < 0 {
			absIdx = layerIdx + layers[0]
		}
		srcShape := tracker.Get(absIdx)
		chPerGroup := srcShape.C / groups
		_ = chPerGroup * groupID // startCh
		_ = groupID              // used via slice initializers added in converter.go

		// These will be added as initializers by the caller through converter.go
		startsName := prefix + "_slice_starts"
		endsName := prefix + "_slice_ends"
		axesName := prefix + "_slice_axes"

		node = &pb.NodeProto{
			OpType: "Slice",
			Input:  []string{inputNames[0], startsName, endsName, axesName},
			Output: []string{outName},
		}

		return RouteResult{
			Node:        node,
			OutputName:  outName,
			OutputShape: Shape{N: srcShape.N, C: chPerGroup, H: srcShape.H, W: srcShape.W},
		}, nil
	}

	// Multiple layers: Concat on channel axis (1)
	node = &pb.NodeProto{
		OpType: "Concat",
		Input:  inputNames,
		Output: []string{outName},
		Attribute: []*pb.AttributeProto{
			makeAttrInt("axis", 1),
		},
	}

	// Use first layer's spatial dims (they should all match)
	absFirst := layers[0]
	if layers[0] < 0 {
		absFirst = layerIdx + layers[0]
	}
	firstShape := tracker.Get(absFirst)

	return RouteResult{
		Node:        node,
		OutputName:  outName,
		OutputShape: Shape{N: firstShape.N, C: totalChannels, H: firstShape.H, W: firstShape.W},
	}, nil
}
