package converter

import (
	"fmt"

	"github.com/LdDl/darknet2onnx/darknet"
	pb "github.com/LdDl/darknet2onnx/onnxpb"
)

// MaxPoolResult holds the ONNX node for a maxpool layer.
type MaxPoolResult struct {
	Node        *pb.NodeProto
	OutputName  string
	OutputShape Shape
}

// BuildMaxPool creates an ONNX MaxPool node for a Darknet [maxpool] layer.
func BuildMaxPool(
	inputName string,
	inputShape Shape,
	layerIdx int,
	sec *darknet.Section,
) MaxPoolResult {
	prefix := fmt.Sprintf("layer%d", layerIdx)

	size := sec.GetInt("size", 2)
	stride := sec.GetInt("stride", 2)

	outName := prefix + "_maxpool"

	// Darknet maxpool with stride=1 uses same-padding to preserve spatial dims
	var pads []int64
	if stride == 1 {
		p := int64(size / 2)
		pads = []int64{p, p, p, p}
	} else {
		pads = []int64{0, 0, 0, 0}
	}

	node := &pb.NodeProto{
		OpType: "MaxPool",
		Input:  []string{inputName},
		Output: []string{outName},
		Attribute: []*pb.AttributeProto{
			makeAttrInts("kernel_shape", []int64{int64(size), int64(size)}),
			makeAttrInts("strides", []int64{int64(stride), int64(stride)}),
			makeAttrInts("pads", pads),
		},
	}

	var outShape Shape
	if stride == 1 {
		// Same-padding preserves spatial dimensions
		outShape = Shape{N: inputShape.N, C: inputShape.C, H: inputShape.H, W: inputShape.W}
	} else {
		outShape = MaxPoolOutputShape(inputShape, size, stride, 0)
	}

	return MaxPoolResult{
		Node:        node,
		OutputName:  outName,
		OutputShape: outShape,
	}
}
