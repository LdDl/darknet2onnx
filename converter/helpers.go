package converter

import (
	"math"

	pb "github.com/LdDl/darknet2onnx/onnxpb"
)

// makeFloatTensor creates an ONNX TensorProto with float data.
func makeFloatTensor(name string, dims []int64, data []float32) *pb.TensorProto {
	return &pb.TensorProto{
		Name:      name,
		DataType:  int32(pb.TensorProto_FLOAT),
		Dims:      dims,
		FloatData: data,
	}
}

// makeInt64Tensor creates an ONNX TensorProto with int64 data.
func makeInt64Tensor(name string, dims []int64, data []int64) *pb.TensorProto {
	return &pb.TensorProto{
		Name:      name,
		DataType:  int32(pb.TensorProto_INT64),
		Dims:      dims,
		Int64Data: data,
	}
}

// makeValueInfo creates a ValueInfoProto for a float tensor with the given shape.
func makeValueInfo(name string, shape []int64) *pb.ValueInfoProto {
	dims := make([]*pb.TensorShapeProto_Dimension, len(shape))
	for i, d := range shape {
		dims[i] = &pb.TensorShapeProto_Dimension{
			Value: &pb.TensorShapeProto_Dimension_DimValue{DimValue: d},
		}
	}
	return &pb.ValueInfoProto{
		Name: name,
		Type: &pb.TypeProto{
			Value: &pb.TypeProto_TensorType{
				TensorType: &pb.TypeProto_Tensor{
					ElemType: int32(pb.TensorProto_FLOAT),
					Shape: &pb.TensorShapeProto{
						Dim: dims,
					},
				},
			},
		},
	}
}

// makeAttrInt creates an integer attribute for ONNX NodeProto.
func makeAttrInt(name string, val int64) *pb.AttributeProto {
	return &pb.AttributeProto{
		Name: name,
		Type: pb.AttributeProto_INT,
		I:    val,
	}
}

// makeAttrFloat creates a float attribute.
func makeAttrFloat(name string, val float32) *pb.AttributeProto {
	return &pb.AttributeProto{
		Name: name,
		Type: pb.AttributeProto_FLOAT,
		F:    val,
	}
}

// makeAttrInts creates an ints attribute.
func makeAttrInts(name string, vals []int64) *pb.AttributeProto {
	return &pb.AttributeProto{
		Name: name,
		Type: pb.AttributeProto_INTS,
		Ints: vals,
	}
}

// makeAttrFloats creates a floats attribute.
func makeAttrFloats(name string, vals []float32) *pb.AttributeProto {
	return &pb.AttributeProto{
		Name:   name,
		Type:   pb.AttributeProto_FLOATS,
		Floats: vals,
	}
}

// makeAttrString creates a string attribute.
func makeAttrString(name string, val string) *pb.AttributeProto {
	return &pb.AttributeProto{
		Name: name,
		Type: pb.AttributeProto_STRING,
		S:    []byte(val),
	}
}

// generateGrid creates a flattened grid of (x, y) offsets for a given grid size.
// Returns two float32 slices of length H*W: grid_x and grid_y.
func generateGrid(h, w int) ([]float32, []float32) {
	gridX := make([]float32, h*w)
	gridY := make([]float32, h*w)
	for row := 0; row < h; row++ {
		for col := 0; col < w; col++ {
			gridX[row*w+col] = float32(col)
			gridY[row*w+col] = float32(row)
		}
	}
	return gridX, gridY
}

// repeatFloat32 repeats each value in a slice n times.
func repeatFloat32(vals []float32, n int) []float32 {
	out := make([]float32, 0, len(vals)*n)
	for _, v := range vals {
		for j := 0; j < n; j++ {
			out = append(out, v)
		}
	}
	return out
}

// tileFloat32 tiles a slice n times.
func tileFloat32(vals []float32, n int) []float32 {
	out := make([]float32, 0, len(vals)*n)
	for i := 0; i < n; i++ {
		out = append(out, vals...)
	}
	return out
}

// exp32 is float32 exp.
func exp32(x float32) float32 {
	return float32(math.Exp(float64(x)))
}
