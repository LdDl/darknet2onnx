package converter

// Shape represents a 4D tensor shape [batch, channels, height, width].
type Shape struct {
	N int
	C int
	H int
	W int
}

// ShapeTracker keeps track of output shapes for each layer.
type ShapeTracker struct {
	shapes []Shape
}

func NewShapeTracker() *ShapeTracker {
	return &ShapeTracker{}
}

func (st *ShapeTracker) Push(s Shape) {
	st.shapes = append(st.shapes, s)
}

// Get returns the shape at a given layer index.
// Negative indices are relative to the current position.
func (st *ShapeTracker) Get(idx int) Shape {
	if idx < 0 {
		idx = len(st.shapes) + idx
	}
	return st.shapes[idx]
}

// Last returns the most recently pushed shape.
func (st *ShapeTracker) Last() Shape {
	return st.shapes[len(st.shapes)-1]
}

// Len returns the number of tracked shapes.
func (st *ShapeTracker) Len() int {
	return len(st.shapes)
}

// ConvOutputShape computes the output shape after a convolution.
func ConvOutputShape(input Shape, filters, kernelSize, stride, pad int) Shape {
	var outH, outW int
	if pad != 0 {
		// Darknet "pad=1" means padding = kernel_size/2
		p := kernelSize / 2
		outH = (input.H+2*p-kernelSize)/stride + 1
		outW = (input.W+2*p-kernelSize)/stride + 1
	} else {
		outH = (input.H-kernelSize)/stride + 1
		outW = (input.W-kernelSize)/stride + 1
	}
	return Shape{N: input.N, C: filters, H: outH, W: outW}
}

// MaxPoolOutputShape computes the output shape after max pooling.
func MaxPoolOutputShape(input Shape, size, stride, pad int) Shape {
	outH := (input.H+2*pad-size)/stride + 1
	outW := (input.W+2*pad-size)/stride + 1
	return Shape{N: input.N, C: input.C, H: outH, W: outW}
}

// UpsampleOutputShape computes the output shape after upsampling.
func UpsampleOutputShape(input Shape, stride int) Shape {
	return Shape{N: input.N, C: input.C, H: input.H * stride, W: input.W * stride}
}
