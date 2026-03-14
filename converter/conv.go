package converter

import (
	"fmt"

	"github.com/LdDl/darknet2onnx/darknet"
	pb "github.com/LdDl/darknet2onnx/onnxpb"
)

// ConvResult holds the ONNX nodes and initializers for a convolutional layer.
type ConvResult struct {
	Nodes        []*pb.NodeProto
	Initializers []*pb.TensorProto
	OutputName   string
	OutputShape  Shape
}

// BuildConv creates ONNX nodes for a Darknet [convolutional] layer.
func BuildConv(
	inputName string,
	inputShape Shape,
	layerIdx int,
	sec *darknet.Section,
	weights *darknet.LayerWeights,
) ConvResult {
	prefix := fmt.Sprintf("layer%d", layerIdx)

	filters := sec.GetInt("filters", 1)
	size := sec.GetInt("size", 1)
	stride := sec.GetInt("stride", 1)
	pad := sec.GetInt("pad", 0)
	groups := sec.GetInt("groups", 1)
	activation := sec.GetString("activation", "linear")

	var result ConvResult

	// Compute padding
	var pads []int64
	if pad != 0 {
		p := int64(size / 2)
		pads = []int64{p, p, p, p}
	} else {
		pads = []int64{0, 0, 0, 0}
	}

	// Conv weights initializer
	inCPerGroup := inputShape.C / groups
	weightName := prefix + "_conv_w"
	weightTensor := makeFloatTensor(
		weightName,
		[]int64{int64(filters), int64(inCPerGroup), int64(size), int64(size)},
		weights.Weights,
	)
	result.Initializers = append(result.Initializers, weightTensor)

	convInputs := []string{inputName, weightName}
	convOutput := prefix + "_conv"

	if !weights.HasBN {
		// No batch norm: bias goes directly into Conv
		biasName := prefix + "_conv_b"
		biasTensor := makeFloatTensor(biasName, []int64{int64(filters)}, weights.Biases)
		result.Initializers = append(result.Initializers, biasTensor)
		convInputs = append(convInputs, biasName)
	}

	// Conv node
	convAttrs := []*pb.AttributeProto{
		makeAttrInts("kernel_shape", []int64{int64(size), int64(size)}),
		makeAttrInts("strides", []int64{int64(stride), int64(stride)}),
		makeAttrInts("pads", pads),
	}
	if groups > 1 {
		convAttrs = append(convAttrs, makeAttrInt("group", int64(groups)))
	}

	convNode := &pb.NodeProto{
		OpType:    "Conv",
		Input:     convInputs,
		Output:    []string{convOutput},
		Attribute: convAttrs,
	}
	result.Nodes = append(result.Nodes, convNode)

	currentOutput := convOutput

	// BatchNormalization
	if weights.HasBN {
		scaleName := prefix + "_bn_scale"
		biasName := prefix + "_bn_bias"
		meanName := prefix + "_bn_mean"
		varName := prefix + "_bn_var"
		bnOutput := prefix + "_bn"

		result.Initializers = append(result.Initializers,
			makeFloatTensor(scaleName, []int64{int64(filters)}, weights.Scales),
			makeFloatTensor(biasName, []int64{int64(filters)}, weights.Biases),
			makeFloatTensor(meanName, []int64{int64(filters)}, weights.Means),
			makeFloatTensor(varName, []int64{int64(filters)}, weights.Variances),
		)

		bnNode := &pb.NodeProto{
			OpType: "BatchNormalization",
			Input:  []string{currentOutput, scaleName, biasName, meanName, varName},
			Output: []string{bnOutput},
			Attribute: []*pb.AttributeProto{
				makeAttrFloat("epsilon", 1e-5),
			},
		}
		result.Nodes = append(result.Nodes, bnNode)
		currentOutput = bnOutput
	}

	// Activation
	act := BuildActivation(currentOutput, layerIdx, activation)
	result.Nodes = append(result.Nodes, act.Nodes...)
	currentOutput = act.OutputName

	result.OutputName = currentOutput
	result.OutputShape = ConvOutputShape(inputShape, filters, size, stride, pad)

	return result
}
