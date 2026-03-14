package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/LdDl/darknet2onnx/converter"
	"github.com/LdDl/darknet2onnx/darknet"
	"google.golang.org/protobuf/proto"
)

func main() {
	cfgPath := flag.String("cfg", "", "Path to Darknet .cfg file (required)")
	weightsPath := flag.String("weights", "", "Path to Darknet .weights file (required)")
	outputPath := flag.String("output", "model.onnx", "Output ONNX file path")
	opset := flag.Int64("opset", 12, "ONNX opset version")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "darknet2onnx - Convert Darknet .cfg+.weights to ONNX\n\n")
		fmt.Fprintf(os.Stderr, "Usage:\n")
		fmt.Fprintf(os.Stderr, "  darknet2onnx --cfg model.cfg --weights model.weights [--output model.onnx] [--opset 12]\n\n")
		fmt.Fprintf(os.Stderr, "Flags:\n")
		flag.PrintDefaults()
	}

	flag.Parse()

	if *cfgPath == "" || *weightsPath == "" {
		flag.Usage()
		os.Exit(1)
	}

	// Parse .cfg
	fmt.Printf("Parsing cfg: %s\n", *cfgPath)
	sections, err := darknet.ParseCfgFile(*cfgPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing cfg: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("  Found %d sections\n", len(sections))

	// Print network params
	netParams := darknet.GetNetParams(sections)
	fmt.Printf("  Input: %dx%dx%d\n", netParams.Width, netParams.Height, netParams.Channels)

	// Read .weights
	fmt.Printf("Reading weights: %s\n", *weightsPath)
	weights, err := darknet.ReadWeights(*weightsPath, sections)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading weights: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("  Loaded %d convolutional layers\n", len(weights))

	// Convert to ONNX
	fmt.Printf("Converting to ONNX (opset %d)...\n", *opset)
	result, err := converter.Convert(sections, weights, *opset)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error converting: %v\n", err)
		os.Exit(1)
	}

	// Serialize
	data, err := proto.Marshal(result.Model)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error serializing ONNX model: %v\n", err)
		os.Exit(1)
	}

	// Write to file
	if err := os.WriteFile(*outputPath, data, 0644); err != nil {
		fmt.Fprintf(os.Stderr, "Error writing output file: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\nDone!\n")
	fmt.Printf("  Output: %s\n", *outputPath)
	fmt.Printf("  Input shape: [1, %d, %d, %d]\n", result.InputShape.C, result.InputShape.H, result.InputShape.W)
	fmt.Printf("  Output shape: [1, N, %d]\n", result.OutputShape[2])
	fmt.Printf("  Layers: %d\n", result.NumLayers)
	fmt.Printf("  File size: %.2f MB\n", float64(len(data))/1024/1024)
}
