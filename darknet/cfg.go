package darknet

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// Section represents a single block in a Darknet .cfg file.
// For example: [convolutional] with its key=value parameters.
type Section struct {
	Type   string
	Params map[string]string
	// Index of this section in the cfg file (0 = [net], 1 = first layer, ...)
	Index int
}

// GetInt returns an integer parameter or a default value.
func (s *Section) GetInt(key string, defaultVal int) int {
	v, ok := s.Params[key]
	if !ok {
		return defaultVal
	}
	i, err := strconv.Atoi(strings.TrimSpace(v))
	if err != nil {
		return defaultVal
	}
	return i
}

// GetFloat returns a float parameter or a default value.
func (s *Section) GetFloat(key string, defaultVal float64) float64 {
	v, ok := s.Params[key]
	if !ok {
		return defaultVal
	}
	f, err := strconv.ParseFloat(strings.TrimSpace(v), 64)
	if err != nil {
		return defaultVal
	}
	return f
}

// GetString returns a string parameter or a default value.
func (s *Section) GetString(key string, defaultVal string) string {
	v, ok := s.Params[key]
	if !ok {
		return defaultVal
	}
	return strings.TrimSpace(v)
}

// GetIntList parses a comma-separated list of integers.
func (s *Section) GetIntList(key string) ([]int, error) {
	v, ok := s.Params[key]
	if !ok {
		return nil, fmt.Errorf("key %q not found", key)
	}
	parts := strings.Split(v, ",")
	result := make([]int, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		i, err := strconv.Atoi(p)
		if err != nil {
			return nil, fmt.Errorf("invalid integer %q in key %q: %w", p, key, err)
		}
		result = append(result, i)
	}
	return result, nil
}

// ParseCfgFile reads a Darknet .cfg file and returns a list of sections.
func ParseCfgFile(path string) ([]Section, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open cfg: %w", err)
	}
	defer f.Close()

	var sections []Section
	var current *Section
	idx := 0

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())

		// Skip empty lines and comments
		if line == "" || strings.HasPrefix(line, "#") || strings.HasPrefix(line, ";") {
			continue
		}

		// New section
		if strings.HasPrefix(line, "[") && strings.HasSuffix(line, "]") {
			if current != nil {
				sections = append(sections, *current)
			}
			sectionType := line[1 : len(line)-1]
			current = &Section{
				Type:   sectionType,
				Params: make(map[string]string),
				Index:  idx,
			}
			idx++
			continue
		}

		// Key=value pair
		if current != nil {
			eqIdx := strings.Index(line, "=")
			if eqIdx > 0 {
				key := strings.TrimSpace(line[:eqIdx])
				val := strings.TrimSpace(line[eqIdx+1:])
				current.Params[key] = val
			}
		}
	}

	if current != nil {
		sections = append(sections, *current)
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan cfg: %w", err)
	}

	if len(sections) == 0 {
		return nil, fmt.Errorf("no sections found in %s", path)
	}

	if sections[0].Type != "net" && sections[0].Type != "network" {
		return nil, fmt.Errorf("first section must be [net], got [%s]", sections[0].Type)
	}

	return sections, nil
}

// NetParams extracts network-level parameters from the [net] section.
type NetParams struct {
	Width    int
	Height   int
	Channels int
}

// GetNetParams returns the network input dimensions from the first section.
func GetNetParams(sections []Section) NetParams {
	net := sections[0]
	return NetParams{
		Width:    net.GetInt("width", 416),
		Height:   net.GetInt("height", 416),
		Channels: net.GetInt("channels", 3),
	}
}
