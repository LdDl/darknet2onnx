package converter

import (
	"fmt"

	pb "github.com/LdDl/darknet2onnx/onnxpb"
)

// ActivationResult contains ONNX nodes for an activation function.
type ActivationResult struct {
	Nodes      []*pb.NodeProto
	OutputName string
}

// BuildActivation creates ONNX node(s) for the given activation type.
// Returns the nodes and the name of the output tensor.
func BuildActivation(inputName string, layerIdx int, activation string) ActivationResult {
	prefix := fmt.Sprintf("layer%d", layerIdx)

	switch activation {
	case "leaky":
		outName := prefix + "_leaky"
		node := &pb.NodeProto{
			OpType: "LeakyRelu",
			Input:  []string{inputName},
			Output: []string{outName},
			Attribute: []*pb.AttributeProto{
				makeAttrFloat("alpha", 0.1),
			},
		}
		return ActivationResult{Nodes: []*pb.NodeProto{node}, OutputName: outName}

	case "mish":
		// Mish = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
		// Decompose for opset 12 (no native Mish op)
		spOut := prefix + "_softplus"
		tanhOut := prefix + "_tanh"
		mishOut := prefix + "_mish"

		softplus := &pb.NodeProto{
			OpType: "Softplus",
			Input:  []string{inputName},
			Output: []string{spOut},
		}
		tanhNode := &pb.NodeProto{
			OpType: "Tanh",
			Input:  []string{spOut},
			Output: []string{tanhOut},
		}
		mul := &pb.NodeProto{
			OpType: "Mul",
			Input:  []string{inputName, tanhOut},
			Output: []string{mishOut},
		}
		return ActivationResult{
			Nodes:      []*pb.NodeProto{softplus, tanhNode, mul},
			OutputName: mishOut,
		}

	case "swish":
		// Swish = x * sigmoid(x)
		sigOut := prefix + "_sigmoid"
		swishOut := prefix + "_swish"

		sigmoid := &pb.NodeProto{
			OpType: "Sigmoid",
			Input:  []string{inputName},
			Output: []string{sigOut},
		}
		mul := &pb.NodeProto{
			OpType: "Mul",
			Input:  []string{inputName, sigOut},
			Output: []string{swishOut},
		}
		return ActivationResult{
			Nodes:      []*pb.NodeProto{sigmoid, mul},
			OutputName: swishOut,
		}

	case "logistic":
		outName := prefix + "_sigmoid"
		node := &pb.NodeProto{
			OpType: "Sigmoid",
			Input:  []string{inputName},
			Output: []string{outName},
		}
		return ActivationResult{Nodes: []*pb.NodeProto{node}, OutputName: outName}

	case "linear", "":
		// No activation
		return ActivationResult{Nodes: nil, OutputName: inputName}

	default:
		// Unknown activation, treat as linear
		return ActivationResult{Nodes: nil, OutputName: inputName}
	}
}
