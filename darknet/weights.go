package darknet

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
)

// LayerWeights holds the weights for a single convolutional layer.
type LayerWeights struct {
	Biases []float32
	// BN scale (gamma)
	Scales []float32
	// BN running mean
	Means []float32
	// BN running variance
	Variances []float32
	// convolution kernel weights
	Weights []float32
	HasBN   bool
}

// WeightsHeader holds the header from a .weights file.
type WeightsHeader struct {
	Major    int32
	Minor    int32
	Revision int32
	Seen     int64
}

// ReadWeights reads a Darknet .weights file using the cfg sections to know
// how many weights to read per layer. Only [convolutional] layers have weights.
func ReadWeights(path string, sections []Section) ([]LayerWeights, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open weights: %w", err)
	}
	defer f.Close()

	header, err := readHeader(f)
	if err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}
	_ = header

	var allWeights []LayerWeights
	// starts with RGB input
	inputChannels := 3

	// Track output channels per layer index for route/shortcut references
	channelsPerLayer := make(map[int]int)
	channelsPerLayer[-1] = inputChannels

	layerIdx := 0
	for _, sec := range sections {
		if sec.Type == "net" || sec.Type == "network" {
			continue
		}

		switch sec.Type {
		case "convolutional":
			filters := sec.GetInt("filters", 1)
			size := sec.GetInt("size", 1)
			groups := sec.GetInt("groups", 1)
			hasBN := sec.GetInt("batch_normalize", 0) == 1

			lw, err := readConvWeights(f, filters, inputChannels/groups, size, hasBN)
			if err != nil {
				return nil, fmt.Errorf("layer %d (conv): %w", layerIdx, err)
			}
			allWeights = append(allWeights, lw)

			inputChannels = filters
			channelsPerLayer[layerIdx] = filters

		case "maxpool":
			channelsPerLayer[layerIdx] = inputChannels

		case "route":
			layers, err := sec.GetIntList("layers")
			if err != nil {
				return nil, fmt.Errorf("layer %d (route): %w", layerIdx, err)
			}
			ch := 0
			for _, l := range layers {
				absIdx := l
				if l < 0 {
					absIdx = layerIdx + l
				}
				c, ok := channelsPerLayer[absIdx]
				if !ok {
					return nil, fmt.Errorf("layer %d (route): referenced layer %d not found", layerIdx, absIdx)
				}
				ch += c
			}
			// Handle groups parameter (splits channels)
			routeGroups := sec.GetInt("groups", 1)
			if routeGroups > 1 {
				ch = ch / routeGroups
			}
			inputChannels = ch
			channelsPerLayer[layerIdx] = ch

		case "shortcut":
			// Output has same shape as previous layer
			channelsPerLayer[layerIdx] = inputChannels

		case "upsample":
			channelsPerLayer[layerIdx] = inputChannels

		case "yolo":
			channelsPerLayer[layerIdx] = inputChannels

		default:
			channelsPerLayer[layerIdx] = inputChannels
		}

		layerIdx++
	}

	return allWeights, nil
}

func readHeader(r io.Reader) (WeightsHeader, error) {
	var h WeightsHeader
	if err := binary.Read(r, binary.LittleEndian, &h.Major); err != nil {
		return h, err
	}
	if err := binary.Read(r, binary.LittleEndian, &h.Minor); err != nil {
		return h, err
	}
	if err := binary.Read(r, binary.LittleEndian, &h.Revision); err != nil {
		return h, err
	}

	// Version >= 2: extra int64 field (seen images counter)
	if h.Major*10+h.Minor >= 2 {
		if err := binary.Read(r, binary.LittleEndian, &h.Seen); err != nil {
			return h, err
		}
	} else {
		var seen32 int32
		if err := binary.Read(r, binary.LittleEndian, &seen32); err != nil {
			return h, err
		}
		h.Seen = int64(seen32)
	}

	return h, nil
}

func readConvWeights(r io.Reader, filters, inChannelsPerGroup, kernelSize int, hasBN bool) (LayerWeights, error) {
	lw := LayerWeights{HasBN: hasBN}

	// Read biases (always present)
	lw.Biases = make([]float32, filters)
	if err := readFloats(r, lw.Biases); err != nil {
		return lw, fmt.Errorf("biases: %w", err)
	}

	// Read BN params if present
	if hasBN {
		lw.Scales = make([]float32, filters)
		if err := readFloats(r, lw.Scales); err != nil {
			return lw, fmt.Errorf("scales: %w", err)
		}
		lw.Means = make([]float32, filters)
		if err := readFloats(r, lw.Means); err != nil {
			return lw, fmt.Errorf("means: %w", err)
		}
		lw.Variances = make([]float32, filters)
		if err := readFloats(r, lw.Variances); err != nil {
			return lw, fmt.Errorf("variances: %w", err)
		}
	}

	// Read conv weights: [filters x in_channels_per_group x kernel x kernel]
	numWeights := filters * inChannelsPerGroup * kernelSize * kernelSize
	lw.Weights = make([]float32, numWeights)
	if err := readFloats(r, lw.Weights); err != nil {
		return lw, fmt.Errorf("weights (%d floats): %w", numWeights, err)
	}

	return lw, nil
}

func readFloats(r io.Reader, dst []float32) error {
	buf := make([]byte, 4*len(dst))
	if _, err := io.ReadFull(r, buf); err != nil {
		return err
	}
	for i := range dst {
		bits := binary.LittleEndian.Uint32(buf[4*i : 4*i+4])
		dst[i] = math.Float32frombits(bits)
	}
	return nil
}
