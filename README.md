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

The YOLO detection head decode logic (sigmoid, grid offsets, anchor application) is embedded into the ONNX graph. All heads are concatenated into a single output tensor.

Coordinates `cx, cy, w, h` are in absolute pixel units relative to the input image dimensions.

Two output formats are available via `--format`:

| Format | Shape | Description |
|--------|-------|-------------|
| `yolov5` (default) | `[1, N, 5+C]` | With objectness: `cx, cy, w, h, obj, cls0..clsN` |
| `yolov8` | `[1, 4+C, N]` | Without objectness: `cx, cy, w, h, cls0..clsN` (obj baked into cls scores) |

where `N` is the total number of predictions and `C` is the number of classes.

Input tensor is named `images`, output tensor is named `output0`. This follows the Ultralytics naming convention, so the output ONNX should be compatible with most inference pipelines that expect this convention. Note that this is not "traditional" YOLO output format, but widly supported I believe.

This could not work for you, but in my case these formats are compatible with [od_opencv](https://github.com/LdDl/object-detection-opencv-rust):
- `yolov5` -> `Model::yolov5_ort()`
- `yolov8` -> `Model::ort()`


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
| `--format` | `yolov5` | Output format: `yolov5` or `yolov8` |

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

The converter runs a three-stage pipeline:

1. Parse `.cfg`

`darknet/cfg.go` reads the Darknet configuration file line by line. The first section (`[net]`) provides input dimensions (width, height, channels). Subsequent sections define the layer stack: `[convolutional]`, `[maxpool]`, `[route]`, `[shortcut]`, `[upsample]`, `[yolo]`.
Pretty straighforward parsing logic I suppose.

2. Read `.weights`

`darknet/weights.go` reads the binary weights file. The header contains format version and training metadata. For each convolutional layer (in the given order) the reader extracts:
- Biases (`filters` floats)
- BatchNorm parameters if `batch_normalize=1` (scales, means, variances)
- Convolution kernel weights (`filters x in_channels/groups x kernel x kernel`)

Note: non-convolutional layers have no weights but affect channel tracking for subsequent layers.

3. Build ONNX graph

`converter/converter.go` iterates the parsed sections and dispatches each layer to a dedicated builder. A `ShapeTracker` keeps output shapes of all layers to resolve forward references in `[route]` and `[shortcut]` layers (which reference by relative/absolute index).

Each Darknet layer maps to standard ONNX operators:

| Builder | ONNX nodes |
|---------|-----------|
| `BuildConv` | Conv + BatchNormalization + activation |
| `BuildMaxPool` | MaxPool (asymmetric padding when stride=1) |
| `BuildRoute` | Concat (multi-layer), Slice (groups), or passthrough |
| `BuildShortcut` | Add + activation |
| `BuildUpsample` | Resize (nearest, scales mode) |
| `BuildYoloDecode` | Decode subgraph (about 20 nodes, see below) |

Activations (`converter/activation.go`) are decomposed into ONNX primitives: `leaky`; LeakyRelu, `mish`; Softplus+Tanh+Mul, `swish`; Sigmoid+Mul, `logistic`; Sigmoid, `linear`; no-op.

4. YOLO decode subgraph

Each `[yolo]` layer takes raw convolution output `[1, A*(5+C), H, W]` and produces decoded predictions `[1, A*H*W, 5+C]` with absolute pixel coordinates:

1. Reshape: `[1, A*(5+C), H, W]`; `[1, A, 5+C, H, W]`
2. Transpose:; `[1, A, H, W, 5+C]`
3. Split: separate `tx,ty` / `tw,th` / `obj+classes`
4. Activate: Sigmoid on `tx,ty,obj,cls` (skipped when `new_coords=1`, since `[convolutional]` before yolo already applies `activation=logistic`)
5. Decode `xy`: apply `scale_x_y` if present, add grid offsets, multiply by stride
6. Decode `wh`: `exp(tw) * anchor` (standard) or `(tw*2)^2 * anchor` (when `new_coords=1`)
7. Concat + Reshape: `[1, A*H*W, 5+C]`

Grid coordinates and anchor values are pre-computed and stored as ONNX initializers.

5. Output fusion

All YOLO head outputs are concatenated along axis 1 into a single tensor:

- `yolov5` format: output as-is `[1, N, 5+C]` with objectness score
- `yolov8`: split off objectness, multiply `obj * cls`, transpose; `[1, 4+C, N]`

The result is a single ONNX model with one input (`images`) and one output (`output0`), fully compatible with standard inference runtimes: ONNX Runtime, OpenCV DNN, TensorRT via `trtexec`. For TensorRT you just need to run:

```bash
trtexec --onnx=pretrained/yolov4-tiny-convertex-to-v8-format.onnx --saveEngine=pretrained/yolov4-tiny-convertex-to-trt-format.engine --fp16      
```

I've tested TensorRT only with `yolov8` format, since (as I believe) it fuses the objectness score into the class probabilities which is more efficient for inference. The `yolov5` format should also work but may require additional post-processing to apply the objectness score.

## License

Just MIT, see [LICENSE](LICENSE).