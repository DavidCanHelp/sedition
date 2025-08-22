package main

import (
	"fmt"
	"log"
	"math/big"
	"strings"
	"time"

	"github.com/davidcanhelp/sedition/crypto"
	"github.com/davidcanhelp/sedition/ml"
	"sedition-poc/poc"
)

// AdvancedConsensusDemo demonstrates quantum-resistant cryptography and ML quality analysis
func main() {
	fmt.Println("=== Advanced Proof of Contribution Consensus Demo ===")
	fmt.Println("Featuring Quantum-Resistant Cryptography and Machine Learning Quality Analysis")
	fmt.Println()

	// Initialize enhanced consensus engine
	minStake := big.NewInt(1000)
	blockTime := 5 * time.Second
	engine := poc.NewEnhancedConsensusEngine(minStake, blockTime)

	fmt.Println("1. Initializing Validators with Quantum-Resistant Keys...")
	
	// Register validators
	validators := []struct {
		address string
		stake   int64
	}{
		{"alice@quantum.dev", 5000},
		{"bob@quantum.dev", 3000},
		{"charlie@quantum.dev", 4000},
	}

	seeds := make(map[string][]byte)
	for _, v := range validators {
		seed := make([]byte, 32)
		// Generate deterministic seed for demo
		copy(seed, []byte(v.address)[:min(32, len(v.address))])
		seeds[v.address] = seed
		
		stake := big.NewInt(v.stake)
		if err := engine.RegisterValidator(v.address, stake, seed); err != nil {
			log.Fatalf("Failed to register validator %s: %v", v.address, err)
		}
		fmt.Printf("  ✓ Registered %s with %d tokens\n", v.address, v.stake)
	}
	fmt.Println()

	// Demonstrate quantum-resistant cryptography
	fmt.Println("2. Quantum-Resistant Cryptography Demo...")
	quantumDemo()
	fmt.Println()

	// Demonstrate ML quality analysis
	fmt.Println("3. Machine Learning Quality Analysis Demo...")
	mlDemo()
	fmt.Println()

	// Create and analyze commits with advanced features
	fmt.Println("4. Advanced Commit Analysis with Combined ML and Quantum Features...")
	advancedCommitDemo(engine)
	fmt.Println()

	// Demonstrate consensus with quantum signatures
	fmt.Println("5. Quantum-Resistant Consensus Simulation...")
	quantumConsensusDemo(engine)
	fmt.Println()

	fmt.Println("=== Advanced Demo Complete ===")
}

func quantumDemo() {
	fmt.Println("  Demonstrating SPHINCS+ Post-Quantum Digital Signatures...")
	
	// Create SPHINCS+ key pair
	keyPair, err := crypto.NewSPHINCSKeyPair(crypto.SPHINCS256s)
	if err != nil {
		log.Printf("Failed to create SPHINCS+ key pair: %v", err)
		return
	}
	
	// Sign a message
	message := []byte("Quantum-resistant blockchain consensus")
	signature, err := keyPair.Sign(message)
	if err != nil {
		log.Printf("Failed to sign message: %v", err)
		return
	}
	
	fmt.Printf("  ✓ Generated SPHINCS+ key pair (256-bit security)\n")
	fmt.Printf("  ✓ Signed message: '%s'\n", string(message))
	fmt.Printf("  ✓ Signature size: %d bytes\n", len(signature.GetSignature()))
	
	// Verify signature
	isValid := signature.Verify(keyPair.GetPublicKey())
	fmt.Printf("  ✓ Signature verification: %v\n", isValid)
	
	fmt.Println()
	fmt.Println("  Demonstrating Kyber Post-Quantum Key Encapsulation...")
	
	// Create Kyber key pair
	kyberKeyPair, err := crypto.NewKyberKeyPair(crypto.Kyber1024)
	if err != nil {
		log.Printf("Failed to create Kyber key pair: %v", err)
		return
	}
	
	// Perform key encapsulation
	sharedSecret, ciphertext, err := kyberKeyPair.Encapsulate()
	if err != nil {
		log.Printf("Failed to encapsulate key: %v", err)
		return
	}
	
	fmt.Printf("  ✓ Generated Kyber-1024 key pair\n")
	fmt.Printf("  ✓ Shared secret size: %d bytes\n", len(sharedSecret))
	fmt.Printf("  ✓ Ciphertext size: %d bytes\n", len(ciphertext.GetCiphertext()))
	
	// Decapsulate shared secret
	recoveredSecret, err := kyberKeyPair.Decapsulate(ciphertext)
	if err != nil {
		log.Printf("Failed to decapsulate key: %v", err)
		return
	}
	
	secretsMatch := string(sharedSecret) == string(recoveredSecret)
	fmt.Printf("  ✓ Secret recovery: %v\n", secretsMatch)
}

func mlDemo() {
	fmt.Println("  Analyzing Code Quality with Machine Learning...")
	
	analyzer := ml.NewMLQualityAnalyzer()
	
	// Sample code snippets for analysis
	codeSnippets := []struct {
		name string
		code string
		expected string
	}{
		{
			name: "High-quality Go function",
			code: `func CalculateCompoundInterest(principal, rate float64, years int) float64 {
	if principal <= 0 || rate <= 0 || years <= 0 {
		return 0
	}
	return principal * math.Pow(1+rate/100, float64(years))
}`,
			expected: "high",
		},
		{
			name: "Poor-quality function",
			code: `func calc(p, r, y) {
	return p * pow(1+r/100, y)
}`,
			expected: "low",
		},
		{
			name: "Security vulnerable code",
			code: `func GetUserData(userID string) string {
	query := "SELECT * FROM users WHERE id = '" + userID + "'"
	result := db.Query(query)
	return result
}`,
			expected: "vulnerable",
		},
		{
			name: "Well-structured Go code",
			code: `// UserRepository handles user data operations
type UserRepository struct {
	db *sql.DB
}

// GetUser retrieves a user by ID with proper error handling
func (ur *UserRepository) GetUser(ctx context.Context, userID string) (*User, error) {
	if userID == "" {
		return nil, errors.New("user ID cannot be empty")
	}
	
	query := "SELECT id, name, email FROM users WHERE id = $1"
	row := ur.db.QueryRowContext(ctx, query, userID)
	
	var user User
	if err := row.Scan(&user.ID, &user.Name, &user.Email); err != nil {
		if err == sql.ErrNoRows {
			return nil, ErrUserNotFound
		}
		return nil, fmt.Errorf("failed to scan user: %w", err)
	}
	
	return &user, nil
}`,
			expected: "high",
		},
	}
	
	for _, snippet := range codeSnippets {
		fmt.Printf("  Analyzing: %s\n", snippet.name)
		
		prediction, err := analyzer.AnalyzeCode(snippet.code, "go", map[string]interface{}{
			"author": "demo-user",
			"timestamp": time.Now(),
		})
		
		if err != nil {
			fmt.Printf("    ✗ Analysis failed: %v\n", err)
			continue
		}
		
		fmt.Printf("    ✓ Overall Quality: %.2f\n", prediction.OverallQuality)
		fmt.Printf("    ✓ Confidence: %.2f\n", prediction.Confidence)
		
		// Display component scores
		for component, score := range prediction.ComponentScores {
			fmt.Printf("      %s: %.2f\n", component, score)
		}
		
		// Display recommendations
		if len(prediction.Recommendations) > 0 {
			fmt.Printf("    Recommendations:\n")
			for _, rec := range prediction.Recommendations {
				fmt.Printf("      - %s\n", rec)
			}
		}
		
		// Display risk factors
		if len(prediction.RiskFactors) > 0 {
			fmt.Printf("    Risk Factors:\n")
			for _, risk := range prediction.RiskFactors {
				fmt.Printf("      - %s\n", risk)
			}
		}
		
		fmt.Println()
	}
}

func advancedCommitDemo(engine *poc.EnhancedConsensusEngine) {
	fmt.Println("  Creating commits with ML analysis and quantum signatures...")
	
	// Sample commit with code diff
	commits := []poc.Commit{
		{
			ID:        "commit-1",
			Author:    "alice@quantum.dev",
			Timestamp: time.Now(),
			Message:   "Add secure authentication system",
			FilesChanged: []string{"auth.go", "middleware.go"},
			LinesAdded:   150,
			LinesModified: 25,
			LinesDeleted: 10,
			Diff: `func AuthenticateUser(ctx context.Context, username, password string) (*User, error) {
	if username == "" || password == "" {
		return nil, errors.New("username and password required")
	}
	
	// Hash password with bcrypt
	hashedPassword, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	if err != nil {
		return nil, fmt.Errorf("failed to hash password: %w", err)
	}
	
	// Use prepared statement to prevent SQL injection
	query := "SELECT id, username, email FROM users WHERE username = $1 AND password = $2"
	row := db.QueryRowContext(ctx, query, username, hashedPassword)
	
	var user User
	if err := row.Scan(&user.ID, &user.Username, &user.Email); err != nil {
		if err == sql.ErrNoRows {
			return nil, ErrInvalidCredentials
		}
		return nil, fmt.Errorf("database error: %w", err)
	}
	
	return &user, nil
}`,
		},
		{
			ID:        "commit-2",
			Author:    "bob@quantum.dev",
			Timestamp: time.Now().Add(-1 * time.Hour),
			Message:   "Quick fix - remove debug code",
			FilesChanged: []string{"utils.go"},
			LinesAdded:   0,
			LinesModified: 5,
			LinesDeleted: 20,
			Diff: `func processData(data []byte) error {
	// TODO: add validation
	return nil
}`,
		},
	}
	
	// Create block with advanced analysis
	block, err := engine.CreateBlock("alice@quantum.dev", commits)
	if err != nil {
		log.Printf("Failed to create block: %v", err)
		return
	}
	
	fmt.Printf("  ✓ Created block with %d commits\n", len(commits))
	fmt.Printf("  ✓ Block hash: %x\n", block.Hash[:8])
	
	// Display commit analysis results
	for i, commit := range commits {
		fmt.Printf("\n  Commit %d Analysis:\n", i+1)
		fmt.Printf("    Quality Score: %.3f\n", commit.QualityScore)
		fmt.Printf("    Signature Type: %s\n", commit.SignatureType)
		
		if commit.MLMetrics != nil {
			if mlQuality, ok := commit.MLMetrics["ml_quality"].(float64); ok {
				fmt.Printf("    ML Quality: %.3f\n", mlQuality)
			}
			if confidence, ok := commit.MLMetrics["confidence"].(float64); ok {
				fmt.Printf("    ML Confidence: %.3f\n", confidence)
			}
			if recommendations, ok := commit.MLMetrics["recommendations"].([]string); ok && len(recommendations) > 0 {
				fmt.Printf("    ML Recommendations:\n")
				for _, rec := range recommendations {
					fmt.Printf("      - %s\n", rec)
				}
			}
			if riskFactors, ok := commit.MLMetrics["risk_factors"].([]string); ok && len(riskFactors) > 0 {
				fmt.Printf("    Risk Factors:\n")
				for _, risk := range riskFactors {
					fmt.Printf("      - %s\n", risk)
				}
			}
		}
	}
}

func quantumConsensusDemo(engine *poc.EnhancedConsensusEngine) {
	fmt.Println("  Simulating consensus rounds with quantum-resistant security...")
	
	// Simulate multiple consensus rounds
	for round := 0; round < 3; round++ {
		fmt.Printf("  Round %d:\n", round+1)
		
		// Select leader using VRF
		leader, err := engine.SelectProposer()
		if err != nil {
			log.Printf("Failed to select leader: %v", err)
			continue
		}
		fmt.Printf("    ✓ Selected leader: %s\n", leader)
		
		// Create sample commits for this round
		commits := []poc.Commit{
			{
				ID:        fmt.Sprintf("commit-%d-1", round),
				Author:    leader,
				Timestamp: time.Now(),
				Message:   "Implement new feature",
				FilesChanged: []string{"feature.go"},
				LinesAdded:   50,
				Diff: `func NewFeature() *Feature {
	return &Feature{
		enabled: true,
		config:  loadConfig(),
	}
}`,
			},
		}
		
		// Create block with quantum-resistant signatures
		block, err := engine.CreateBlock(leader, commits)
		if err != nil {
			log.Printf("Failed to create block: %v", err)
			continue
		}
		
		fmt.Printf("    ✓ Created block with quantum signatures\n")
		fmt.Printf("    ✓ Block size: %d bytes\n", len(block.Hash))
		
		// Simulate block validation by other validators
		validators := []string{"alice@quantum.dev", "bob@quantum.dev", "charlie@quantum.dev"}
		validations := 0
		
		for _, validator := range validators {
			if validator == leader {
				continue // Leader doesn't validate their own block
			}
			
			// Simulate validation (in real system this would be more complex)
			isValid := engine.ValidateBlock(block)
			if isValid {
				validations++
				fmt.Printf("    ✓ Validator %s approved the block\n", validator)
			}
		}
		
		if validations >= len(validators)/2 {
			fmt.Printf("    ✓ Block achieved consensus (%d/%d validations)\n", validations, len(validators)-1)
		} else {
			fmt.Printf("    ✗ Block rejected (%d/%d validations)\n", validations, len(validators)-1)
		}
		
		fmt.Println()
		time.Sleep(1 * time.Second) // Simulate block time
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Extension methods for quantum crypto (would normally be in crypto package)
func (sig *crypto.SPHINCSSignature) GetSignature() []byte {
	return sig.Signature
}

func (kp *crypto.SPHINCSKeyPair) GetPublicKey() []byte {
	return kp.PublicKey
}

func (ct *crypto.KyberCiphertext) GetCiphertext() []byte {
	return ct.Ciphertext
}