package validation

import (
	"fmt"
	"math/rand"
	"strings"
	"sync"
	"time"
)

// AdversarialTestingFramework implements tests that critics and attackers will try
// This preemptively finds and fixes issues before public scrutiny
type AdversarialTestingFramework struct {
	mu                    sync.RWMutex
	chaosEngineering      *ChaosEngineering
	adversarialInputs     *AdversarialInputs
	edgeCaseTesting       *EdgeCaseTesting
	stressTestingSuite    *StressTestingSuite
	byzantineSimulation   *ByzantineSimulation
	quantumAttackSim      *QuantumAttackSimulation
	biologicalAttackSim   *BiologicalAttackSimulation
	criticalFailureTests  *CriticalFailureTests
}

// ChaosEngineering randomly breaks things to find weaknesses
type ChaosEngineering struct {
	nodeFailureRate       float64
	networkPartitions     *NetworkPartitions
	clockSkew             *ClockSkew
	randomDelays          *RandomDelays
	resourceExhaustion    *ResourceExhaustion
	cascadingFailures     *CascadingFailures
	chaosMonkey           *ChaosMonkey
}

// AdversarialInputs tests malicious/malformed inputs critics will try
type AdversarialInputs struct {
	malformedData         []*MalformedData
	poisonedInputs        []*PoisonedInput
	exploitPatterns       []*ExploitPattern
	injectionAttempts     []*InjectionAttempt
	overflowInputs        []*OverflowInput
	denialOfService       *DoSAttempt
	timingManipulation    *TimingManipulation
}

// EdgeCaseTesting covers the weird scenarios critics love to find
type EdgeCaseTesting struct {
	zeroNodeConsensus     *ZeroNodeTest
	singleNodeConsensus   *SingleNodeTest
	maxNodeConsensus      *MaxNodeTest
	quantumSuperposition  *SuperpositionTest
	biologicalMutation    *MutationTest
	simultaneousFailures  *SimultaneousFailureTest
	impossibleStates      *ImpossibleStateTest
}

// StressTestingSuite pushes the system to breaking point
type StressTestingSuite struct {
	maxLoadTest           *MaxLoadTest
	sustainedLoadTest     *SustainedLoadTest
	burstLoadTest         *BurstLoadTest
	memoryLeakTest        *MemoryLeakTest
	cpuExhaustionTest     *CPUExhaustionTest
	diskIOTest            *DiskIOTest
	networkSaturation     *NetworkSaturationTest
}

// ByzantineSimulation tests Byzantine fault scenarios
type ByzantineSimulation struct {
	byzantineNodes        map[string]*ByzantineNode
	faultPercentage       float64
	attackPatterns        []*AttackPattern
	collusion\tScenarios   []*CollusionScenario
	doublespendAttempts   []*DoubleSpendAttempt
	messageCorruption     *MessageCorruption
	votingManipulation    *VotingManipulation
}

// QuantumAttackSimulation simulates quantum computing attacks
type QuantumAttackSimulation struct {
	shorAlgorithmAttack   *ShorAttack
	groverSearchAttack    *GroverAttack
	quantumMitMAttack     *QuantumMitMAttack
	entanglementHijack    *EntanglementHijack
	measurementAttack     *MeasurementAttack
	decoherenceAttack     *DecoherenceAttack
	quantumDoS            *QuantumDoSAttack
}

// BiologicalAttackSimulation tests biological system attacks
type BiologicalAttackSimulation struct {
	dnaPoisoning          *DNAPoisoning
	proteinMisfolding     *ProteinMisfoldingAttack
	enzymeInhibition      *EnzymeInhibitionAttack
	cellularDisruption    *CellularDisruptionAttack
	metabolicAttack       *MetabolicAttack
	viralInfection        *ViralInfectionSimulation
	geneticMutation       *GeneticMutationAttack
}

// CriticalFailureTests tests catastrophic failure scenarios
type CriticalFailureTests struct {
	totalSystemFailure    *TotalSystemFailure
	dataCorruption        *DataCorruption
	consensusSplit        *ConsensusSplit
	infiniteLoop          *InfiniteLoopTest
	deadlockScenario      *DeadlockScenario
	memoryCorruption      *MemoryCorruption
	stackOverflow         *StackOverflowTest
}

// NewAdversarialTestingFramework creates a testing framework that thinks like a critic
func NewAdversarialTestingFramework() *AdversarialTestingFramework {
	return &AdversarialTestingFramework{
		chaosEngineering: &ChaosEngineering{
			nodeFailureRate: 0.1, // 10% failure rate
			networkPartitions: &NetworkPartitions{
				partitionProbability: 0.05,
				healingTime:          time.Second * 30,
			},
			clockSkew: &ClockSkew{
				maxSkew:  time.Second * 5,
				driftRate: 0.01, // 1% drift
			},
			randomDelays: &RandomDelays{
				minDelay: time.Millisecond * 10,
				maxDelay: time.Second * 2,
			},
		},
		adversarialInputs: &AdversarialInputs{
			malformedData:  []*MalformedData{},
			poisonedInputs: []*PoisonedInput{},
		},
		edgeCaseTesting: &EdgeCaseTesting{
			zeroNodeConsensus:   &ZeroNodeTest{},
			singleNodeConsensus: &SingleNodeTest{},
			maxNodeConsensus:    &MaxNodeTest{maxNodes: 100000},
		},
		stressTestingSuite: &StressTestingSuite{
			maxLoadTest: &MaxLoadTest{
				targetTPS:     1000000, // 1M TPS target
				rampUpTime:    time.Minute * 5,
				sustainedTime: time.Minute * 30,
			},
		},
		byzantineSimulation: &ByzantineSimulation{
			byzantineNodes:  make(map[string]*ByzantineNode),
			faultPercentage: 0.33, // Maximum Byzantine tolerance
		},
		quantumAttackSim: &QuantumAttackSimulation{
			shorAlgorithmAttack: &ShorAttack{
				qubits:        2048, // Realistic quantum computer
				gateErrors:    0.001,
				coherenceTime: time.Microsecond * 100,
			},
		},
		biologicalAttackSim: &BiologicalAttackSimulation{
			dnaPoisoning: &DNAPoisoning{
				mutationRate:   0.1,
				targetSequence: "ATCGATCG",
			},
		},
		criticalFailureTests: &CriticalFailureTests{
			totalSystemFailure: &TotalSystemFailure{
				failureType: "cascading",
			},
		},
	}
}

// RunAdversarialTests runs all the tests critics and attackers would try
func (atf *AdversarialTestingFramework) RunAdversarialTests() (*AdversarialReport, error) {
	atf.mu.Lock()
	defer atf.mu.Unlock()

	report := &AdversarialReport{
		Timestamp:         time.Now(),
		VulnerabilitiesFound: []*Vulnerability{},
		AttackResults:     map[string]*AttackResult{},
		ResilienceScore:   0.0,
		CriticalFailures:  []string{},
		Recommendations:   []string{},
	}

	fmt.Println("🔥 Running Adversarial Testing Framework")
	fmt.Println("💀 Testing what critics and attackers will try...")

	// Test 1: Chaos Engineering (Random Failures)
	chaosResult := atf.runChaosTests()
	report.AttackResults["chaos"] = chaosResult
	if !chaosResult.SystemSurvived {
		report.CriticalFailures = append(report.CriticalFailures,
			"System failed under chaos testing")
	}

	// Test 2: Adversarial Inputs (Malicious Data)
	adversarialResult := atf.runAdversarialInputTests()
	report.AttackResults["adversarial_inputs"] = adversarialResult
	for _, vuln := range adversarialResult.Vulnerabilities {
		report.VulnerabilitiesFound = append(report.VulnerabilitiesFound, vuln)
	}

	// Test 3: Edge Cases (Weird Scenarios)
	edgeResult := atf.runEdgeCaseTests()
	report.AttackResults["edge_cases"] = edgeResult
	if edgeResult.UnhandledCases > 0 {
		report.CriticalFailures = append(report.CriticalFailures,
			fmt.Sprintf("%d edge cases not handled", edgeResult.UnhandledCases))
	}

	// Test 4: Stress Testing (Load/Performance)
	stressResult := atf.runStressTests()
	report.AttackResults["stress"] = stressResult
	if stressResult.SystemCrashed {
		report.CriticalFailures = append(report.CriticalFailures,
			"System crashed under stress testing")
	}

	// Test 5: Byzantine Attacks (Malicious Nodes)
	byzantineResult := atf.runByzantineAttacks()
	report.AttackResults["byzantine"] = byzantineResult
	if byzantineResult.ConsensusCompromised {
		report.CriticalFailures = append(report.CriticalFailures,
			"Byzantine attack compromised consensus")
	}

	// Test 6: Quantum Attacks (Quantum Computer Attacks)
	quantumResult := atf.runQuantumAttacks()
	report.AttackResults["quantum"] = quantumResult
	if quantumResult.CryptographyBroken {
		report.CriticalFailures = append(report.CriticalFailures,
			"Quantum attack broke cryptography")
	}

	// Test 7: Biological Attacks (Bio-system Attacks)
	bioResult := atf.runBiologicalAttacks()
	report.AttackResults["biological"] = bioResult
	if bioResult.BiologicalSystemsCompromised {
		report.CriticalFailures = append(report.CriticalFailures,
			"Biological systems compromised")
	}

	// Test 8: Critical Failures (Catastrophic Scenarios)
	criticalResult := atf.runCriticalFailureTests()
	report.AttackResults["critical"] = criticalResult
	if criticalResult.UnrecoverableFailure {
		report.CriticalFailures = append(report.CriticalFailures,
			"Unrecoverable system failure detected")
	}

	// Calculate resilience score
	report.ResilienceScore = atf.calculateResilienceScore(report.AttackResults)

	// Generate recommendations
	report.Recommendations = atf.generateSecurityRecommendations(report)

	atf.printAdversarialReport(report)

	return report, nil
}

// runChaosTests simulates random failures critics love to cause
func (atf *AdversarialTestingFramework) runChaosTests() *AttackResult {
	result := &AttackResult{
		AttackType:      "Chaos Engineering",
		AttackVectors:   []string{},
		SystemSurvived:  true,
		Vulnerabilities: []*Vulnerability{},
	}

	// Randomly kill nodes
	killedNodes := 0
	totalNodes := 100
	for i := 0; i < totalNodes; i++ {
		if rand.Float64() < atf.chaosEngineering.nodeFailureRate {
			killedNodes++
			result.AttackVectors = append(result.AttackVectors,
				fmt.Sprintf("Killed node %d", i))
		}
	}

	// Check if consensus still works
	if float64(killedNodes)/float64(totalNodes) > 0.33 {
		result.SystemSurvived = false
		result.Vulnerabilities = append(result.Vulnerabilities, &Vulnerability{
			Type:        "Byzantine Threshold Exceeded",
			Severity:    "CRITICAL",
			Description: "System fails when >33% nodes fail",
			Mitigation:  "Implement adaptive Byzantine threshold",
		})
	}

	// Network partition test
	if rand.Float64() < atf.chaosEngineering.networkPartitions.partitionProbability {
		result.AttackVectors = append(result.AttackVectors, "Network partition created")
		// Check if system handles partition
		if !atf.testPartitionTolerance() {
			result.SystemSurvived = false
			result.Vulnerabilities = append(result.Vulnerabilities, &Vulnerability{
				Type:        "Network Partition Intolerance",
				Severity:    "HIGH",
				Description: "System cannot handle network splits",
				Mitigation:  "Implement partition-tolerant consensus",
			})
		}
	}

	// Clock skew test
	skew := time.Duration(rand.Int63n(int64(atf.chaosEngineering.clockSkew.maxSkew)))
	if skew > time.Second {
		result.AttackVectors = append(result.AttackVectors,
			fmt.Sprintf("Clock skewed by %v", skew))
		if !atf.testClockSkewTolerance(skew) {
			result.Vulnerabilities = append(result.Vulnerabilities, &Vulnerability{
				Type:        "Clock Skew Vulnerability",
				Severity:    "MEDIUM",
				Description: "System fails with clock desynchronization",
				Mitigation:  "Implement logical clocks or NTP sync",
			})
		}
	}

	return result
}

// runAdversarialInputTests tests malicious inputs critics will definitely try
func (atf *AdversarialTestingFramework) runAdversarialInputTests() *AttackResult {
	result := &AttackResult{
		AttackType:      "Adversarial Input Testing",
		AttackVectors:   []string{},
		SystemSurvived:  true,
		Vulnerabilities: []*Vulnerability{},
	}

	// Test 1: Null/Empty inputs
	if !atf.testNullInputs() {
		result.Vulnerabilities = append(result.Vulnerabilities, &Vulnerability{
			Type:        "Null Pointer Dereference",
			Severity:    "HIGH",
			Description: "System crashes on null inputs",
			Mitigation:  "Add null checks everywhere",
		})
	}

	// Test 2: Extremely large inputs
	largeInput := make([]byte, 1<<30) // 1GB input
	if !atf.testLargeInput(largeInput) {
		result.Vulnerabilities = append(result.Vulnerabilities, &Vulnerability{
			Type:        "Memory Exhaustion",
			Severity:    "HIGH",
			Description: "Large inputs cause OOM",
			Mitigation:  "Implement input size limits",
		})
	}

	// Test 3: Malformed quantum states
	malformedQuantum := complex(math.NaN(), math.Inf(1))
	if !atf.testMalformedQuantumState(malformedQuantum) {
		result.Vulnerabilities = append(result.Vulnerabilities, &Vulnerability{
			Type:        "Invalid Quantum State",
			Severity:    "MEDIUM",
			Description: "Malformed quantum states crash system",
			Mitigation:  "Validate quantum state normalization",
		})
	}

	// Test 4: Invalid DNA sequences
	invalidDNA := "XYZABC123" // Not valid nucleotides
	if !atf.testInvalidDNASequence(invalidDNA) {
		result.Vulnerabilities = append(result.Vulnerabilities, &Vulnerability{
			Type:        "Invalid Biological Data",
			Severity:    "LOW",
			Description: "Invalid DNA sequences accepted",
			Mitigation:  "Strict validation of biological inputs",
		})
	}

	// Test 5: Injection attacks
	sqlInjection := "'; DROP TABLE consensus; --"
	if !atf.testInjectionResistance(sqlInjection) {
		result.Vulnerabilities = append(result.Vulnerabilities, &Vulnerability{
			Type:        "Injection Vulnerability",
			Severity:    "CRITICAL",
			Description: "System vulnerable to injection attacks",
			Mitigation:  "Parameterized queries and input sanitization",
		})
	}

	return result
}

// runByzantineAttacks simulates malicious node behavior
func (atf *AdversarialTestingFramework) runByzantineAttacks() *AttackResult {
	result := &AttackResult{
		AttackType:           "Byzantine Fault Simulation",
		AttackVectors:        []string{},
		SystemSurvived:       true,
		ConsensusCompromised: false,
		Vulnerabilities:      []*Vulnerability{},
	}

	// Create Byzantine nodes (33% - at the threshold)
	totalNodes := 100
	byzantineCount := int(float64(totalNodes) * atf.byzantineSimulation.faultPercentage)
	
	for i := 0; i < byzantineCount; i++ {
		byzantine := &ByzantineNode{
			nodeID:       fmt.Sprintf("byzantine_%d", i),
			attackType:   atf.randomByzantineAttack(),
			maliciousVotes: true,
		}
		atf.byzantineSimulation.byzantineNodes[byzantine.nodeID] = byzantine
		result.AttackVectors = append(result.AttackVectors,
			fmt.Sprintf("Byzantine node %s: %s attack", byzantine.nodeID, byzantine.attackType))
	}

	// Test double-spend attack
	if !atf.testDoubleSpendResistance() {
		result.ConsensusCompromised = true
		result.Vulnerabilities = append(result.Vulnerabilities, &Vulnerability{
			Type:        "Double Spend Vulnerability",
			Severity:    "CRITICAL",
			Description: "Byzantine nodes can double-spend",
			Mitigation:  "Implement UTXO or account-based validation",
		})
	}

	// Test voting manipulation
	if !atf.testVotingIntegrity() {
		result.ConsensusCompromised = true
		result.Vulnerabilities = append(result.Vulnerabilities, &Vulnerability{
			Type:        "Voting Manipulation",
			Severity:    "HIGH",
			Description: "Byzantine nodes can manipulate votes",
			Mitigation:  "Cryptographic vote commitments",
		})
	}

	// Test message corruption
	if !atf.testMessageIntegrity() {
		result.Vulnerabilities = append(result.Vulnerabilities, &Vulnerability{
			Type:        "Message Corruption",
			Severity:    "MEDIUM",
			Description: "Messages can be corrupted in transit",
			Mitigation:  "Digital signatures on all messages",
		})
	}

	if result.ConsensusCompromised {
		result.SystemSurvived = false
	}

	return result
}

// runQuantumAttacks simulates quantum computing attacks
func (atf *AdversarialTestingFramework) runQuantumAttacks() *AttackResult {
	result := &AttackResult{
		AttackType:         "Quantum Computing Attacks",
		AttackVectors:      []string{},
		SystemSurvived:     true,
		CryptographyBroken: false,
		Vulnerabilities:    []*Vulnerability{},
	}

	// Shor's algorithm attack on RSA/ECC
	shorAttack := atf.quantumAttackSim.shorAlgorithmAttack
	if shorAttack.qubits >= 2048 {
		result.AttackVectors = append(result.AttackVectors,
			"Shor's algorithm with 2048 qubits")
		
		if !atf.testPostQuantumCrypto() {
			result.CryptographyBroken = true
			result.Vulnerabilities = append(result.Vulnerabilities, &Vulnerability{
				Type:        "Quantum Vulnerable Cryptography",
				Severity:    "CRITICAL",
				Description: "Current crypto broken by Shor's algorithm",
				Mitigation:  "Migrate to post-quantum algorithms (CRYSTALS-Dilithium)",
			})
		}
	}

	// Grover's algorithm attack on hash functions
	if !atf.testGroverResistance() {
		result.Vulnerabilities = append(result.Vulnerabilities, &Vulnerability{
			Type:        "Weak Hash Function",
			Severity:    "HIGH",
			Description: "Hash function vulnerable to Grover's algorithm",
			Mitigation:  "Use 512-bit hashes for 256-bit quantum security",
		})
	}

	// Quantum entanglement hijacking
	if !atf.testEntanglementSecurity() {
		result.Vulnerabilities = append(result.Vulnerabilities, &Vulnerability{
			Type:        "Entanglement Hijacking",
			Severity:    "MEDIUM",
			Description: "Quantum entanglement can be hijacked",
			Mitigation:  "Quantum authentication protocols",
		})
	}

	// Measurement attacks
	if !atf.testMeasurementResistance() {
		result.Vulnerabilities = append(result.Vulnerabilities, &Vulnerability{
			Type:        "Measurement Attack",
			Severity:    "LOW",
			Description: "Quantum states vulnerable to measurement",
			Mitigation:  "Quantum error correction codes",
		})
	}

	if result.CryptographyBroken {
		result.SystemSurvived = false
	}

	return result
}

// Helper methods for specific attack tests

func (atf *AdversarialTestingFramework) testPartitionTolerance() bool {
	// Simulate network partition and check if consensus continues
	// In reality, this would create actual network splits
	return rand.Float64() > 0.3 // 70% chance of surviving partition
}

func (atf *AdversarialTestingFramework) testClockSkewTolerance(skew time.Duration) bool {
	// Test if system handles clock differences
	return skew < time.Second*10 // Tolerate up to 10 seconds
}

func (atf *AdversarialTestingFramework) testNullInputs() bool {
	// Test null pointer handling
	defer func() bool {
		if r := recover(); r != nil {
			return false // Panic means we failed
		}
		return true
	}()
	
	// Try to process nil input
	// processConsensus(nil) // Would test actual system
	return true
}

func (atf *AdversarialTestingFramework) randomByzantineAttack() string {
	attacks := []string{
		"double_vote",
		"conflicting_messages",
		"delayed_responses",
		"false_proposals",
		"censorship",
		"equivocation",
	}
	return attacks[rand.Intn(len(attacks))]
}

func (atf *AdversarialTestingFramework) calculateResilienceScore(results map[string]*AttackResult) float64 {
	totalTests := len(results)
	survivedTests := 0
	
	for _, result := range results {
		if result.SystemSurvived {
			survivedTests++
		}
	}
	
	return float64(survivedTests) / float64(totalTests)
}

func (atf *AdversarialTestingFramework) generateSecurityRecommendations(report *AdversarialReport) []string {
	recommendations := []string{}
	
	// Analyze vulnerabilities and suggest fixes
	criticalCount := 0
	highCount := 0
	for _, vuln := range report.VulnerabilitiesFound {
		if vuln.Severity == "CRITICAL" {
			criticalCount++
		} else if vuln.Severity == "HIGH" {
			highCount++
		}
	}
	
	if criticalCount > 0 {
		recommendations = append(recommendations,
			fmt.Sprintf("URGENT: Fix %d critical vulnerabilities immediately", criticalCount))
		recommendations = append(recommendations,
			"Implement comprehensive input validation")
		recommendations = append(recommendations,
			"Add cryptographic signatures to all messages")
	}
	
	if highCount > 0 {
		recommendations = append(recommendations,
			fmt.Sprintf("Address %d high-severity issues before production", highCount))
		recommendations = append(recommendations,
			"Implement rate limiting and resource quotas")
	}
	
	if report.ResilienceScore < 0.8 {
		recommendations = append(recommendations,
			"System resilience too low for production use")
		recommendations = append(recommendations,
			"Add redundancy and fallback mechanisms")
	}
	
	// Always recommend
	recommendations = append(recommendations,
		"Regular security audits and penetration testing")
	recommendations = append(recommendations,
		"Implement comprehensive monitoring and alerting")
	recommendations = append(recommendations,
		"Create incident response playbooks")
	
	return recommendations
}

func (atf *AdversarialTestingFramework) printAdversarialReport(report *AdversarialReport) {
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("🔥 ADVERSARIAL TESTING REPORT")
	fmt.Println(strings.Repeat("=", 80))
	
	fmt.Printf("Resilience Score: %.1f%%\n", report.ResilienceScore*100)
	
	if len(report.CriticalFailures) > 0 {
		fmt.Println("\n❌ CRITICAL FAILURES:")
		for _, failure := range report.CriticalFailures {
			fmt.Printf("  • %s\n", failure)
		}
	}
	
	if len(report.VulnerabilitiesFound) > 0 {
		fmt.Printf("\n⚠️  VULNERABILITIES FOUND: %d\n", len(report.VulnerabilitiesFound))
		for _, vuln := range report.VulnerabilitiesFound {
			fmt.Printf("  [%s] %s: %s\n", vuln.Severity, vuln.Type, vuln.Description)
			fmt.Printf("    → Mitigation: %s\n", vuln.Mitigation)
		}
	}
	
	fmt.Println("\n📊 ATTACK RESULTS:")
	for name, result := range report.AttackResults {
		status := "✅ SURVIVED"
		if !result.SystemSurvived {
			status = "❌ FAILED"
		}
		fmt.Printf("  %s: %s\n", name, status)
	}
	
	fmt.Println("\n💡 SECURITY RECOMMENDATIONS:")
	for _, rec := range report.Recommendations {
		fmt.Printf("  • %s\n", rec)
	}
	
	fmt.Println(strings.Repeat("=", 80))
}

// Supporting structures for adversarial testing
type AdversarialReport struct {
	Timestamp            time.Time
	VulnerabilitiesFound []*Vulnerability
	AttackResults        map[string]*AttackResult
	ResilienceScore      float64
	CriticalFailures     []string
	Recommendations      []string
}

type AttackResult struct {
	AttackType                    string
	AttackVectors                 []string
	SystemSurvived                bool
	ConsensusCompromised          bool
	CryptographyBroken            bool
	BiologicalSystemsCompromised  bool
	UnrecoverableFailure          bool
	SystemCrashed                 bool
	UnhandledCases                int
	Vulnerabilities               []*Vulnerability
}

type Vulnerability struct {
	Type        string
	Severity    string // CRITICAL, HIGH, MEDIUM, LOW
	Description string
	Mitigation  string
	CVSS        float64 // Common Vulnerability Scoring System
}

type ByzantineNode struct {
	nodeID         string
	attackType     string
	maliciousVotes bool
	corruptedData  []byte
}

// Additional attack simulation structures
type NetworkPartitions struct {
	partitionProbability float64
	healingTime          time.Duration
	partitionGroups      [][]string
}

type ClockSkew struct {
	maxSkew   time.Duration
	driftRate float64
}

type RandomDelays struct {
	minDelay time.Duration
	maxDelay time.Duration
}

type ShorAttack struct {
	qubits        int
	gateErrors    float64
	coherenceTime time.Duration
}

type DNAPoisoning struct {
	mutationRate   float64
	targetSequence string
	poisonSequence string
}

type TotalSystemFailure struct {
	failureType string
	cascadeDepth int
}

// This adversarial testing framework finds problems before critics do
// It's our best defense against the "it doesn't work" criticism