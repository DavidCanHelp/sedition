
package benchmarks

type TestEnvironment struct{}

func (te *TestEnvironment) Cleanup() {}
func (te *TestEnvironment) ApplyNetworkConditions(nc *NetworkConditions) {}
func (te *TestEnvironment) MeasureCPUUsage() float64 { return 0.0 }
func (te *TestEnvironment) MeasureMemoryUsage() int64 { return 0 }
func (te *TestEnvironment) MeasureNetworkUsage() float64 { return 0.0 }

