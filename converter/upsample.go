package converter

import (
	"fmt"

	"github.com/LdDl/darknet2onnx/darknet"
	pb "github.com/LdDl/darknet2onnx/onnxpb"
)

// UpsampleResult holds the ONNX node and initializers for an upsample layer.
type UpsampleResult struct {
	Node         *pb.NodeProto
	Initializers []*pb.TensorProto
	OutputName   string
	OutputShape  Shape
}

// BuildUpsample creates an ONNX Resize node for a Darknet [upsample] layer.
// Uses opset 12 Resize with scales input.
func BuildUpsample(
	inputName string,
	inputShape Shape,
	layerIdx int,
	sec *darknet.Section,
) UpsampleResult {
	prefix := fmt.Sprintf("layer%d", layerIdx)

	stride := sec.GetInt("stride", 2)

	outName := prefix + "_upsample"

	// Opset 11+ Resize requires: X, roi, scales (or sizes)
	// We use scales mode with nearest interpolation
	roiName := prefix + "_roi"
	scalesName := prefix + "_scales"

	roiTensor := makeFloatTensor(roiName, []int64{0}, nil) // empty roi
	scalesTensor := makeFloatTensor(scalesName, []int64{4}, []float32{
		1.0, 1.0, float32(stride), float32(stride),
	})

	node := &pb.NodeProto{
		OpType: "Resize",
		Input:  []string{inputName, roiName, scalesName},
		Output: []string{outName},
		Attribute: []*pb.AttributeProto{
			makeAttrString("mode", "nearest"),
		},
	}

	return UpsampleResult{
		Node:         node,
		Initializers: []*pb.TensorProto{roiTensor, scalesTensor},
		OutputName:   outName,
		OutputShape:  UpsampleOutputShape(inputShape, stride),
	}
}
