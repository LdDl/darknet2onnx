# darknet2onnx

Convert Darknet `.cfg` + `.weights` to ONNX format.

A standalone Go CLI tool that produces a single static binary with no Python or pip dependencies.

## Table of Contents

- [Installation](#installation)
- [Supported models](#supported-models)
- [Output format](#output-format)
- [Build from source](#build-from-source)
- [Usage](#usage)
  - [Flags](#flags)
  - [Example](#example)
- [Validate output](#validate-output)
- [Supported layer types](#supported-layer-types)
- [Protobuf source](#protobuf-source)
- [How it works](#how-it-works)
- [License](#license)

## Installation

### From Go

```bash
go install github.com/LdDl/darknet2onnx@latest
```

### Pre-built binaries

Download the latest release for your platform from [Releases](https://github.com/LdDl/darknet2onnx/releases/latest):

| Platform | Archive |
|----------|---------|
| Linux amd64 | [`linux-amd64-darknet2onnx.tar.gz`](https://github.com/LdDl/darknet2onnx/releases/latest/download/linux-amd64-darknet2onnx.tar.gz) |
| Linux arm64 | [`linux-arm64-darknet2onnx.tar.gz`](https://github.com/LdDl/darknet2onnx/releases/latest/download/linux-arm64-darknet2onnx.tar.gz) |
| macOS amd64 | [`darwin-amd64-darknet2onnx.tar.gz`](https://github.com/LdDl/darknet2onnx/releases/latest/download/darwin-amd64-darknet2onnx.tar.gz) |
| macOS arm64 | [`darwin-arm64-darknet2onnx.tar.gz`](https://github.com/LdDl/darknet2onnx/releases/latest/download/darwin-arm64-darknet2onnx.tar.gz) |
| Windows amd64 | [`windows-amd64-darknet2onnx.zip`](https://github.com/LdDl/darknet2onnx/releases/latest/download/windows-amd64-darknet2onnx.zip) |

Extract and place the binary somewhere in your `$PATH`.

## Supported models

- YOLOv3, YOLOv3-tiny
- YOLOv4, YOLOv4-tiny
- YOLOv7, YOLOv7-tiny

## Output format

The YOLO detection head decode logic (sigmoid, grid offsets, anchor application) is embedded into the ONNX graph. All heads are concatenated into a single output tensor:

```
[1, N, 5 + num_classes]
```

where `N` is the total number of predictions across all heads and `5 = cx, cy, w, h, objectness`.

Coordinates `cx, cy, w, h` are in absolute pixel units relative to the input image dimensions.

This could not work for you, but in my case this format is compatible with `ModelYOLOv5Ort` / `Model::yolov5_ort()` in [od_opencv](https://github.com/LdDl/object-detection-opencv-rust). 

## Build from source

Simple build for the current platform:

```bash
go build -o darknet2onnx .
```

Cross-compile for all platforms (linux/windows/macOS, amd64/arm64):

```bash
./build.sh
```

## Usage

```bash
./darknet2onnx --cfg model.cfg --weights model.weights --output model.onnx
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--cfg` | (required) | Path to Darknet `.cfg` file |
| `--weights` | (required) | Path to Darknet `.weights` file |
| `--output` | `model.onnx` | Output ONNX file path |
| `--opset` | `12` | ONNX opset version |

### Example

```bash
./darknet2onnx \
    --cfg pretrained/yolov3-tiny.cfg \
    --weights pretrained/yolov3-tiny.weights \
    --output pretrained/yolov3-tiny.onnx
```

## Validate output

Install `onnx` in a Python virtual environment:

```bash
python3 -m venv .venv
.venv/bin/pip install onnx
```

Then validate:

```bash
.venv/bin/python3 -c "
import onnx
m = onnx.load('model.onnx')
onnx.checker.check_model(m)
print('Valid')
"
```

Clean up:

```bash
rm -rf .venv
```

## Supported layer types

| Darknet layer | ONNX op(s) |
|---------------|------------|
| `[convolutional]` | Conv + BatchNormalization + activation |
| `[maxpool]` | MaxPool |
| `[route]` | Concat or passthrough (+ Slice for groups) |
| `[shortcut]` | Add + activation |
| `[upsample]` | Resize (nearest) |
| `[yolo]` | Decode subgraph (Reshape, Sigmoid, Add, Exp, Mul, Concat) |

Activations: `leaky` (LeakyRelu), `mish` (Softplus + Tanh + Mul), `swish` (Sigmoid + Mul), `logistic` (Sigmoid), `linear` (none).

## Protobuf source

The ONNX protobuf schema (`proto/onnx.proto3`) is downloaded from the official ONNX repository:

```
https://raw.githubusercontent.com/onnx/onnx/main/onnx/onnx.proto3
```

To regenerate Go bindings, you need `protoc` and `protoc-gen-go`:

```bash
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
```

Then run:

```bash
protoc -I proto proto/onnx.proto3 --go_out=./onnxpb --go_opt=paths=source_relative --experimental_allow_proto3_optional
```

## How it works

@todo

## License

Just MIT, see [LICENSE](LICENSE).