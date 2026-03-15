package converter

import (
	"fmt"

	"github.com/LdDl/darknet2onnx/darknet"
	pb "github.com/LdDl/darknet2onnx/onnxpb"
)

// YoloResult holds the ONNX subgraph for a YOLO detection head decode.
type YoloResult struct {
	Nodes        []*pb.NodeProto
	Initializers []*pb.TensorProto
	// decoded output: [1, A*H*W, 5+C]
	OutputName string
}

// BuildYoloDecode creates ONNX nodes that decode a YOLO detection head.
//
// Input: raw conv output [1, A*(5+C), H, W]
// Output: decoded tensor [1, A*H*W, 5+C] with absolute coordinates
//
// Steps:
//  1. Reshape to [1, A, 5+C, H, W]
//  2. Transpose to [1, A, H, W, 5+C]
//  3. Sigmoid on tx, ty, objectness, class scores
//  4. Add grid offsets to cx, cy
//  5. Apply exp + anchors to w, h
//  6. Scale to input dimensions
//  7. Reshape to [1, A*H*W, 5+C]
func BuildYoloDecode(
	inputName string,
	// shape of conv output before yolo
	inputShape Shape,
	layerIdx int,
	sec *darknet.Section,
	netWidth, netHeight int,
) (YoloResult, error) {
	prefix := fmt.Sprintf("yolo%d", layerIdx)

	// Parse YOLO params
	mask, err := sec.GetIntList("mask")
	if err != nil {
		return YoloResult{}, fmt.Errorf("yolo mask: %w", err)
	}
	allAnchors, err := sec.GetIntList("anchors")
	if err != nil {
		return YoloResult{}, fmt.Errorf("yolo anchors: %w", err)
	}
	numClasses := sec.GetInt("classes", 80)

	scaleXY := sec.GetFloat("scale_x_y", 1.0)
	newCoords := sec.GetInt("new_coords", 0)

	numAnchors := len(mask)
	gridH := inputShape.H
	gridW := inputShape.W
	boxAttrs := 5 + numClasses // tx, ty, tw, th, obj, class0..classN

	var nodes []*pb.NodeProto
	var inits []*pb.TensorProto

	// 1. Reshape [1, A*(5+C), H, W] -> [1, A, 5+C, H, W]
	reshapeName1 := prefix + "_shape1"
	reshape1Out := prefix + "_reshaped1"
	inits = append(inits, makeInt64Tensor(reshapeName1, []int64{5}, []int64{
		1, int64(numAnchors), int64(boxAttrs), int64(gridH), int64(gridW),
	}))
	nodes = append(nodes, &pb.NodeProto{
		OpType: "Reshape",
		Input:  []string{inputName, reshapeName1},
		Output: []string{reshape1Out},
	})

	// 2. Transpose [1, A, 5+C, H, W] -> [1, A, H, W, 5+C]
	transpose1Out := prefix + "_transposed"
	nodes = append(nodes, &pb.NodeProto{
		OpType: "Transpose",
		Input:  []string{reshape1Out},
		Output: []string{transpose1Out},
		Attribute: []*pb.AttributeProto{
			makeAttrInts("perm", []int64{0, 1, 3, 4, 2}),
		},
	})

	// 3. Split along last axis into: [tx, ty] [tw, th] [obj + classes]
	splitOut0 := prefix + "_txty"
	splitOut1 := prefix + "_twth"
	splitOut2 := prefix + "_obj_cls"

	nodes = append(nodes, &pb.NodeProto{
		OpType: "Split",
		Input:  []string{transpose1Out},
		Output: []string{splitOut0, splitOut1, splitOut2},
		Attribute: []*pb.AttributeProto{
			makeAttrInts("split", []int64{2, 2, int64(1 + numClasses)}),
			makeAttrInt("axis", 4),
		},
	})

	// new_coords=1 (YOLOv7): activation=logistic on prev conv already applied sigmoid
	// to ALL outputs. Decode uses different formulas for xy and wh.
	// new_coords=0 (YOLOv3/v4): standard decode with sigmoid in yolo layer.

	// name of processed tx/ty tensor
	var txtyProcessed string
	// name of processed obj/cls tensor
	var objClsProcessed string

	if newCoords == 1 {
		// Already sigmoided by activation=logistic, no additional sigmoid needed
		txtyProcessed = splitOut0
		objClsProcessed = splitOut2
	} else {
		// 3a. Sigmoid on tx, ty
		sigTxTy := prefix + "_sig_txty"
		nodes = append(nodes, &pb.NodeProto{
			OpType: "Sigmoid",
			Input:  []string{splitOut0},
			Output: []string{sigTxTy},
		})
		txtyProcessed = sigTxTy

		// 3b. Sigmoid on objectness + class scores
		sigObjCls := prefix + "_sig_obj_cls"
		nodes = append(nodes, &pb.NodeProto{
			OpType: "Sigmoid",
			Input:  []string{splitOut2},
			Output: []string{sigObjCls},
		})
		objClsProcessed = sigObjCls
	}

	// 4. Add grid offsets to tx, ty
	gridX, gridY := generateGrid(gridH, gridW)

	gridXName := prefix + "_grid_x"
	gridYName := prefix + "_grid_y"
	inits = append(inits,
		makeFloatTensor(gridXName, []int64{1, 1, int64(gridH), int64(gridW), 1}, gridX),
		makeFloatTensor(gridYName, []int64{1, 1, int64(gridH), int64(gridW), 1}, gridY),
	)

	// Concat grid_x, grid_y along last axis -> [1, 1, H, W, 2]
	gridXY := prefix + "_grid_xy"
	nodes = append(nodes, &pb.NodeProto{
		OpType: "Concat",
		Input:  []string{gridXName, gridYName},
		Output: []string{gridXY},
		Attribute: []*pb.AttributeProto{
			makeAttrInt("axis", 4),
		},
	})

	// Apply scale_x_y: tx * scale_x_y - (scale_x_y - 1) / 2 + grid
	var xyBeforeGrid string
	if scaleXY != 1.0 {
		scaleName := prefix + "_scale_xy_val"
		offsetName := prefix + "_scale_xy_offset"
		inits = append(inits,
			makeFloatTensor(scaleName, []int64{1}, []float32{float32(scaleXY)}),
			makeFloatTensor(offsetName, []int64{1}, []float32{float32((scaleXY - 1.0) / 2.0)}),
		)
		mulOut := prefix + "_sig_scaled"
		nodes = append(nodes, &pb.NodeProto{
			OpType: "Mul",
			Input:  []string{txtyProcessed, scaleName},
			Output: []string{mulOut},
		})
		subOut := prefix + "_sig_shifted"
		nodes = append(nodes, &pb.NodeProto{
			OpType: "Sub",
			Input:  []string{mulOut, offsetName},
			Output: []string{subOut},
		})
		xyBeforeGrid = subOut
	} else {
		xyBeforeGrid = txtyProcessed
	}

	// xy + grid_offset
	addedXY := prefix + "_added_xy"
	nodes = append(nodes, &pb.NodeProto{
		OpType: "Add",
		Input:  []string{xyBeforeGrid, gridXY},
		Output: []string{addedXY},
	})

	// Scale xy by stride (net_dim / grid_dim)
	strideX := float32(netWidth) / float32(gridW)
	strideY := float32(netHeight) / float32(gridH)
	xyScaleName := prefix + "_xy_scale"
	inits = append(inits, makeFloatTensor(xyScaleName, []int64{1, 1, 1, 1, 2}, []float32{strideX, strideY}))

	scaledXY := prefix + "_scaled_xy"
	nodes = append(nodes, &pb.NodeProto{
		OpType: "Mul",
		Input:  []string{addedXY, xyScaleName},
		Output: []string{scaledXY},
	})

	// 5. Decode width/height and multiply by anchors
	// Build anchor tensor [1, A, 1, 1, 2] from mask indices
	anchorData := make([]float32, numAnchors*2)
	for i, m := range mask {
		if m*2+1 >= len(allAnchors) {
			return YoloResult{}, fmt.Errorf("anchor mask index %d out of range", m)
		}
		anchorData[i*2] = float32(allAnchors[m*2])
		anchorData[i*2+1] = float32(allAnchors[m*2+1])
	}

	anchorsName := prefix + "_anchors"
	inits = append(inits, makeFloatTensor(anchorsName, []int64{1, int64(numAnchors), 1, 1, 2}, anchorData))

	var scaledWH string
	if newCoords == 1 {
		// new_coords=1: wh = (tw * 2)^2 * anchor
		twoName := prefix + "_two"
		inits = append(inits, makeFloatTensor(twoName, []int64{1}, []float32{2.0}))

		mulTwo := prefix + "_wh_mul2"
		nodes = append(nodes, &pb.NodeProto{
			OpType: "Mul",
			Input:  []string{splitOut1, twoName},
			Output: []string{mulTwo},
		})
		squared := prefix + "_wh_sq"
		nodes = append(nodes, &pb.NodeProto{
			OpType: "Mul",
			Input:  []string{mulTwo, mulTwo},
			Output: []string{squared},
		})
		scaledWH = prefix + "_scaled_wh"
		nodes = append(nodes, &pb.NodeProto{
			OpType: "Mul",
			Input:  []string{squared, anchorsName},
			Output: []string{scaledWH},
		})
	} else {
		// Standard: wh = exp(tw) * anchor
		expTwTh := prefix + "_exp_twth"
		nodes = append(nodes, &pb.NodeProto{
			OpType: "Exp",
			Input:  []string{splitOut1},
			Output: []string{expTwTh},
		})
		scaledWH = prefix + "_scaled_wh"
		nodes = append(nodes, &pb.NodeProto{
			OpType: "Mul",
			Input:  []string{expTwTh, anchorsName},
			Output: []string{scaledWH},
		})
	}

	// 6. Concat all parts back: [xy, wh, obj_cls] along last axis
	concatOut := prefix + "_concat"
	nodes = append(nodes, &pb.NodeProto{
		OpType: "Concat",
		Input:  []string{scaledXY, scaledWH, objClsProcessed},
		Output: []string{concatOut},
		Attribute: []*pb.AttributeProto{
			makeAttrInt("axis", 4),
		},
	})

	// 7. Reshape [1, A, H, W, 5+C] -> [1, A*H*W, 5+C]
	reshapeName2 := prefix + "_shape2"
	finalOut := prefix + "_output"
	totalPreds := int64(numAnchors * gridH * gridW)
	inits = append(inits, makeInt64Tensor(reshapeName2, []int64{3}, []int64{
		1, totalPreds, int64(boxAttrs),
	}))
	nodes = append(nodes, &pb.NodeProto{
		OpType: "Reshape",
		Input:  []string{concatOut, reshapeName2},
		Output: []string{finalOut},
	})

	return YoloResult{
		Nodes:        nodes,
		Initializers: inits,
		OutputName:   finalOut,
	}, nil
}
