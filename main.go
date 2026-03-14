package main

import (
	"flag"
	"fmt"
	"log/slog"
	"os"

	"github.com/LdDl/darknet2onnx/converter"
	"github.com/LdDl/darknet2onnx/darknet"
	"google.golang.org/protobuf/proto"
)

func init() {
	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stdout, nil)))
}

func main() {
	cfgPath := flag.String("cfg", "", "Path to Darknet .cfg file (required)")
	weightsPath := flag.String("weights", "", "Path to Darknet .weights file (required)")
	outputPath := flag.String("output", "model.onnx", "Output ONNX file path")
	opset := flag.Int64("opset", 12, "ONNX opset version")
	format := flag.String("format", "yolov5", "Output format: yolov5 [1,N,5+C] or yolov8 [1,4+C,N]")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "darknet2onnx - Convert Darknet .cfg+.weights to ONNX\n\n")
		fmt.Fprintf(os.Stderr, "Usage:\n")
		fmt.Fprintf(os.Stderr, "  darknet2onnx --cfg model.cfg --weights model.weights [--output model.onnx] [--opset 12] [--format yolov5]\n\n")
		fmt.Fprintf(os.Stderr, "Flags:\n")
		flag.PrintDefaults()
	}

	flag.Parse()

	if *cfgPath == "" || *weightsPath == "" {
		flag.Usage()
		os.Exit(1)
	}

	var outputFormat converter.OutputFormat
	switch *format {
	case "yolov5":
		outputFormat = converter.FormatYOLOv5
	case "yolov8":
		outputFormat = converter.FormatYOLOv8
	default:
		slog.Error("unknown format, expected yolov5 or yolov8", slog.String("format", *format))
		os.Exit(1)
	}

	// Parse .cfg
	slog.Info("parsing cfg", slog.String("path", *cfgPath))
	sections, err := darknet.ParseCfgFile(*cfgPath)
	if err != nil {
		slog.Error("failed to parse cfg", slog.String("error", err.Error()))
		os.Exit(1)
	}

	netParams := darknet.GetNetParams(sections)
	slog.Info("cfg parsed",
		slog.Int("sections", len(sections)),
		slog.Int("width", netParams.Width),
		slog.Int("height", netParams.Height),
		slog.Int("channels", netParams.Channels),
	)

	// Read .weights
	slog.Info("reading weights", slog.String("path", *weightsPath))
	weights, err := darknet.ReadWeights(*weightsPath, sections)
	if err != nil {
		slog.Error("failed to read weights", slog.String("error", err.Error()))
		os.Exit(1)
	}
	slog.Info("weights loaded", slog.Int("conv_layers", len(weights)))

	// Convert to ONNX
	slog.Info("converting to ONNX",
		slog.Int64("opset", *opset),
		slog.String("format", *format),
	)
	result, err := converter.Convert(sections, weights, *opset, outputFormat)
	if err != nil {
		slog.Error("failed to convert", slog.String("error", err.Error()))
		os.Exit(1)
	}

	// Serialize
	data, err := proto.Marshal(result.Model)
	if err != nil {
		slog.Error("failed to serialize ONNX model", slog.String("error", err.Error()))
		os.Exit(1)
	}

	// Write to file
	err = os.WriteFile(*outputPath, data, 0644)
	if err != nil {
		slog.Error("failed to write output file", slog.String("error", err.Error()))
		os.Exit(1)
	}

	slog.Info("done",
		slog.String("output", *outputPath),
		slog.String("format", string(result.Format)),
		slog.String("input_shape", fmt.Sprintf("[1, %d, %d, %d]", result.InputShape.C, result.InputShape.H, result.InputShape.W)),
		slog.String("output_shape", fmt.Sprintf("%v", result.OutputShape)),
		slog.Int("layers", result.NumLayers),
		slog.Float64("file_size_mb", float64(len(data))/1024/1024),
	)
}
