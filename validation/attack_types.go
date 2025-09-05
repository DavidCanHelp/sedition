package validation

import (
	"fmt"
	"math/rand"
	"time"
)

// ResourceExhaustion represents resource exhaustion attack
type ResourceExhaustion struct {
	Type        string
	Intensity   float64
	Duration    time.Duration
	TargetNodes []string
}

// CascadingFailures represents cascading failure scenarios
type CascadingFailures struct {
	InitialFailures int
	PropagationRate float64
	MaxDepth        int
	RecoveryTime    time.Duration
}

// ChaosMonkey implements chaos engineering
type ChaosMonkey struct {
	Enabled        bool
	FailureRate    float64
	Actions        []string
	TargetServices []string
}

// MalformedData represents malformed data attacks
type MalformedData struct {
	Type     string
	Payload  []byte
	Expected []byte
}

// PoisonedInput represents poisoned input data
type PoisonedInput struct {
	Type        string
	Data        interface{}
	PoisonRatio float64
}

// ExploitPattern represents known exploit patterns
type ExploitPattern struct {
	Name        string
	Category    string
	Severity    string
	Pattern     string
	Mitigation  string
}

// InjectionAttempt represents various injection attacks
type InjectionAttempt struct {
	Type    string // SQL, Command, Script, etc.
	Payload string
	Target  string
}

// OverflowInput represents buffer/integer overflow attempts
type OverflowInput struct {
	Type       string
	Size       int
	TargetSize int
	Pattern    []byte
}

// DoSAttempt represents denial of service attempts
type DoSAttempt struct {
	Type         string
	Rate         float64
	Duration     time.Duration
	Amplification float64
}

// TimingManipulation represents timing-based attacks
type TimingManipulation struct {
	Type      string
	DelayMs   int
	Jitter    float64
	Pattern   string
}

// Factory functions for creating attack instances
func NewResourceExhaustion(attackType string, intensity float64) *ResourceExhaustion {
	return &ResourceExhaustion{
		Type:      attackType,
		Intensity: intensity,
		Duration:  time.Minute * 5,
	}
}

func NewCascadingFailures(initial int, rate float64) *CascadingFailures {
	return &CascadingFailures{
		InitialFailures: initial,
		PropagationRate: rate,
		MaxDepth:        5,
		RecoveryTime:    time.Second * 30,
	}
}

func NewChaosMonkey(rate float64) *ChaosMonkey {
	return &ChaosMonkey{
		Enabled:     true,
		FailureRate: rate,
		Actions: []string{
			"kill_process",
			"network_partition",
			"cpu_spike",
			"memory_leak",
			"disk_full",
		},
	}
}

func NewMalformedData(dataType string) *MalformedData {
	return &MalformedData{
		Type:    dataType,
		Payload: generateMalformedPayload(dataType),
	}
}

func NewPoisonedInput(dataType string, ratio float64) *PoisonedInput {
	return &PoisonedInput{
		Type:        dataType,
		PoisonRatio: ratio,
		Data:        generatePoisonedData(dataType, ratio),
	}
}

func NewExploitPattern(name, category string) *ExploitPattern {
	return &ExploitPattern{
		Name:     name,
		Category: category,
		Severity: "HIGH",
	}
}

func NewInjectionAttempt(injType, target string) *InjectionAttempt {
	return &InjectionAttempt{
		Type:    injType,
		Target:  target,
		Payload: generateInjectionPayload(injType),
	}
}

func NewOverflowInput(overflowType string, size int) *OverflowInput {
	return &OverflowInput{
		Type:       overflowType,
		Size:       size,
		TargetSize: 256, // Default buffer size
		Pattern:    generateOverflowPattern(size),
	}
}

func NewDoSAttempt(dosType string, rate float64) *DoSAttempt {
	return &DoSAttempt{
		Type:          dosType,
		Rate:          rate,
		Duration:      time.Minute,
		Amplification: 1.0,
	}
}

func NewTimingManipulation(timingType string, delay int) *TimingManipulation {
	return &TimingManipulation{
		Type:    timingType,
		DelayMs: delay,
		Jitter:  0.1,
		Pattern: "random",
	}
}

// Helper functions for generating attack payloads
func generateMalformedPayload(dataType string) []byte {
	switch dataType {
	case "json":
		return []byte(`{"key": "value"`)  // Missing closing brace
	case "xml":
		return []byte(`<root><item></root>`) // Mismatched tags
	case "binary":
		// Random binary data
		payload := make([]byte, 256)
		rand.Read(payload)
		return payload
	default:
		return []byte("malformed")
	}
}

func generatePoisonedData(dataType string, ratio float64) interface{} {
	// Generate data with poison ratio
	switch dataType {
	case "numeric":
		if rand.Float64() < ratio {
			return float64(1<<53 - 1) // Max safe integer
		}
		return rand.Float64() * 100
	case "string":
		if rand.Float64() < ratio {
			return fmt.Sprintf("'; DROP TABLE users; --")
		}
		return "normal_data"
	default:
		return nil
	}
}

func generateInjectionPayload(injType string) string {
	payloads := map[string][]string{
		"sql": {
			"' OR '1'='1",
			"'; DROP TABLE users; --",
			"1; SELECT * FROM passwords",
		},
		"command": {
			"; ls -la",
			"| cat /etc/passwd",
			"&& rm -rf /",
		},
		"script": {
			"<script>alert('XSS')</script>",
			"javascript:void(0)",
			"<img src=x onerror=alert('XSS')>",
		},
	}
	
	if patterns, ok := payloads[injType]; ok && len(patterns) > 0 {
		return patterns[rand.Intn(len(patterns))]
	}
	return "generic_injection"
}

func generateOverflowPattern(size int) []byte {
	pattern := make([]byte, size)
	for i := range pattern {
		pattern[i] = byte('A' + (i % 26))
	}
	return pattern
}

// Execute methods for attacks
func (re *ResourceExhaustion) Execute() error {
	// Simulate resource exhaustion
	fmt.Printf("Executing resource exhaustion attack: %s at intensity %.2f\n", re.Type, re.Intensity)
	return nil
}

func (cf *CascadingFailures) Trigger() error {
	// Simulate cascading failures
	fmt.Printf("Triggering cascading failures: %d initial, %.2f propagation rate\n", 
		cf.InitialFailures, cf.PropagationRate)
	return nil
}

func (cm *ChaosMonkey) Run() error {
	if !cm.Enabled {
		return nil
	}
	// Execute random chaos actions
	action := cm.Actions[rand.Intn(len(cm.Actions))]
	fmt.Printf("Chaos Monkey executing: %s\n", action)
	return nil
}