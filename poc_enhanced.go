// Package poc implements the enhanced Proof of Contribution consensus with real cryptography
package poc

import (
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"math/big"
	"sort"
	"sync"
	"time"

	"github.com/davidcanhelp/sedition/crypto"
)

// EnhancedConsensusEngine implements PoC with real cryptographic primitives
type EnhancedConsensusEngine struct {
	mu sync.RWMutex

	// Validators and their cryptographic identities
	validators map[string]*EnhancedValidator
	signers    map[string]*crypto.Signer

	// Cryptographic components
	vrfs        map[string]*crypto.VRF
	merkleRoots map[int64][]byte // epoch -> state root

	// Quality and reputation systems
	qualityAnalyzer   *QualityAnalyzer
	reputationTracker *ReputationTracker
	metricsCalculator *MetricsCalculator

	// Consensus parameters
	minStakeRequired *big.Int
	blockTime        time.Duration
	epochLength      int64
	committeeSize    int

	// Current state
	currentEpoch     int64
	currentRound     uint64
	lastBlockTime    time.Time
	epochSeed        []byte
	proposerHistory  []string
	blockChain       []*Block
}

// EnhancedValidator represents a validator with cryptographic identity
type EnhancedValidator struct {
	Address         string
	PublicKey       ed25519.PublicKey
	TokenStake      *big.Int
	ReputationScore float64
	RecentContribs  []Contribution
	TotalStake      *big.Int
	VRFPublicKey    ed25519.PublicKey
	LastActivity    time.Time
	IsActive        bool
	SlashingEvents  []SlashingEvent
}

// Block represents a block in the blockchain
type Block struct {
	Height       int64
	PrevHash     []byte
	Timestamp    time.Time
	Proposer     string
	VRFProof     []byte
	Commits      []Commit
	StateRoot    []byte
	Signatures   map[string][]byte // validator -> signature
	Hash         []byte
}

// Commit represents a code commit in a block
type Commit struct {
	ID            string
	Author        string
	Hash          []byte
	Timestamp     time.Time
	Message       string
	FilesChanged  []string
	LinesAdded    int
	LinesModified int
	LinesDeleted  int
	QualityScore  float64
	Signature     []byte // Author's signature
}

// NewEnhancedConsensusEngine creates a new consensus engine with cryptography
func NewEnhancedConsensusEngine(minStake *big.Int, blockTime time.Duration) *EnhancedConsensusEngine {
	// Generate epoch seed
	epochSeed := make([]byte, 32)
	rand.Read(epochSeed)

	return &EnhancedConsensusEngine{
		validators:        make(map[string]*EnhancedValidator),
		signers:          make(map[string]*crypto.Signer),
		vrfs:             make(map[string]*crypto.VRF),
		merkleRoots:      make(map[int64][]byte),
		qualityAnalyzer:  NewQualityAnalyzer(),
		reputationTracker: NewReputationTracker(),
		metricsCalculator: NewMetricsCalculator(),
		minStakeRequired: minStake,
		blockTime:        blockTime,
		epochLength:      100,
		committeeSize:    21,
		currentEpoch:     0,
		currentRound:     0,
		epochSeed:        epochSeed,
		blockChain:       make([]*Block, 0),
	}
}

// RegisterValidator registers a new validator with cryptographic keys
func (e *EnhancedConsensusEngine) RegisterValidator(address string, stake *big.Int, seed []byte) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if stake.Cmp(e.minStakeRequired) < 0 {
		return fmt.Errorf("insufficient stake: %s < %s", stake.String(), e.minStakeRequired.String())
	}

	// Create cryptographic identity
	signer, err := crypto.NewSignerFromSeed(seed)
	if err != nil {
		return fmt.Errorf("failed to create signer: %w", err)
	}

	vrf, err := crypto.NewVRFFromSeed(seed)
	if err != nil {
		return fmt.Errorf("failed to create VRF: %w", err)
	}

	validator := &EnhancedValidator{
		Address:         address,
		PublicKey:       signer.GetPublicKey(),
		TokenStake:      new(big.Int).Set(stake),
		ReputationScore: 5.0, // Start with neutral reputation
		RecentContribs:  make([]Contribution, 0),
		TotalStake:      new(big.Int).Set(stake),
		VRFPublicKey:    vrf.GetPublicKey(),
		LastActivity:    time.Now(),
		IsActive:        true,
	}

	e.validators[address] = validator
	e.signers[address] = signer
	e.vrfs[address] = vrf

	// Initialize reputation
	e.reputationTracker.InitializeReputation(address)

	return nil
}

// SelectBlockProposer uses VRF-based sortition to select the next proposer
func (e *EnhancedConsensusEngine) SelectBlockProposer() (string, *crypto.VRFSortitionProof, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if len(e.validators) == 0 {
		return "", nil, errors.New("no validators registered")
	}

	// Calculate total stake
	totalStake := big.NewInt(0)
	activeValidators := make([]*EnhancedValidator, 0)
	for _, v := range e.validators {
		if v.IsActive {
			e.calculateEnhancedStake(v)
			totalStake.Add(totalStake, v.TotalStake)
			activeValidators = append(activeValidators, v)
		}
	}

	if len(activeValidators) == 0 {
		return "", nil, errors.New("no active validators")
	}

	// Perform VRF-based sortition
	var bestProof *crypto.VRFSortitionProof
	var bestValidator string
	bestValue := new(big.Int)

	for addr, v := range e.validators {
		if !v.IsActive {
			continue
		}

		vrf := e.vrfs[addr]
		if vrf == nil {
			continue
		}

		// Perform sortition
		proof, err := crypto.Sortition(
			vrf,
			e.epochSeed,
			e.currentRound,
			"proposer",
			v.TotalStake,
			totalStake,
			1, // Expected 1 proposer
		)
		if err != nil {
			continue
		}

		// Check if selected (J > 0) and has best value
		if proof.J > 0 {
			proofValue := proof.VRFOutput.GetRandomness()
			if bestProof == nil || proofValue.Cmp(bestValue) < 0 {
				bestProof = proof
				bestValidator = addr
				bestValue = proofValue
			}
		}
	}

	if bestProof == nil {
		return "", nil, errors.New("no validator selected in sortition")
	}

	// Add to proposer history for fairness tracking
	e.proposerHistory = append(e.proposerHistory, bestValidator)
	if len(e.proposerHistory) > 100 {
		e.proposerHistory = e.proposerHistory[1:]
	}

	return bestValidator, bestProof, nil
}

// ProposeBlock creates a new block proposal
func (e *EnhancedConsensusEngine) ProposeBlock(proposer string, commits []Commit, sortitionProof *crypto.VRFSortitionProof) (*Block, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	validator, exists := e.validators[proposer]
	if !exists {
		return nil, errors.New("proposer not found")
	}

	signer, exists := e.signers[proposer]
	if !exists {
		return nil, errors.New("proposer signer not found")
	}

	// Get previous block hash
	var prevHash []byte
	if len(e.blockChain) > 0 {
		prevHash = e.blockChain[len(e.blockChain)-1].Hash
	} else {
		prevHash = make([]byte, 32) // Genesis block has zero prev hash
	}

	// Analyze and sign each commit
	for i := range commits {
		quality := e.qualityAnalyzer.AnalyzeCommit(&commits[i])
		commits[i].QualityScore = quality.OverallScore

		// Sign commit
		commitData := e.serializeCommit(&commits[i])
		sig, err := signer.Sign(commitData)
		if err != nil {
			return nil, fmt.Errorf("failed to sign commit: %w", err)
		}
		commits[i].Signature = sig
	}

	// Create Merkle tree of commits
	commitHashes := make([][]byte, len(commits))
	for i, commit := range commits {
		commitHashes[i] = commit.Hash
	}

	var stateRoot []byte
	if len(commitHashes) > 0 {
		merkleTree, err := crypto.NewMerkleTree(commitHashes)
		if err != nil {
			return nil, fmt.Errorf("failed to create merkle tree: %w", err)
		}
		stateRoot = merkleTree.GetRoot()
	} else {
		stateRoot = make([]byte, 32)
	}

	// Create block
	block := &Block{
		Height:     int64(len(e.blockChain)),
		PrevHash:   prevHash,
		Timestamp:  time.Now(),
		Proposer:   proposer,
		VRFProof:   sortitionProof.VRFOutput.Proof,
		Commits:    commits,
		StateRoot:  stateRoot,
		Signatures: make(map[string][]byte),
	}

	// Calculate block hash
	block.Hash = e.calculateBlockHash(block)

	// Proposer signs the block
	blockSig, err := signer.Sign(block.Hash)
	if err != nil {
		return nil, fmt.Errorf("failed to sign block: %w", err)
	}
	block.Signatures[proposer] = blockSig

	return block, nil
}

// ValidateBlock validates a proposed block
func (e *EnhancedConsensusEngine) ValidateBlock(block *Block) error {
	e.mu.RLock()
	defer e.mu.RUnlock()

	// Verify proposer exists and is active
	proposer, exists := e.validators[block.Proposer]
	if !exists {
		return errors.New("proposer not found")
	}
	if !proposer.IsActive {
		return errors.New("proposer is not active")
	}

	// Verify VRF proof
	vrfOutput := &crypto.VRFOutput{
		Proof: block.VRFProof,
	}

	// Calculate total stake for verification
	totalStake := big.NewInt(0)
	for _, v := range e.validators {
		if v.IsActive {
			totalStake.Add(totalStake, v.TotalStake)
		}
	}

	// Verify sortition proof
	valid, err := crypto.VerifySortition(
		proposer.VRFPublicKey,
		e.epochSeed,
		e.currentRound,
		"proposer",
		&crypto.VRFSortitionProof{
			VRFOutput: vrfOutput,
			Stake:     proposer.TotalStake,
		},
		totalStake,
		1,
	)
	if err != nil || !valid {
		return errors.New("invalid VRF proof")
	}

	// Verify block hash
	calculatedHash := e.calculateBlockHash(block)
	if !bytes.Equal(calculatedHash, block.Hash) {
		return errors.New("invalid block hash")
	}

	// Verify proposer signature
	proposerSig, exists := block.Signatures[block.Proposer]
	if !exists {
		return errors.New("proposer signature missing")
	}

	if !ed25519.Verify(proposer.PublicKey, block.Hash, proposerSig) {
		return errors.New("invalid proposer signature")
	}

	// Verify Merkle root
	if len(block.Commits) > 0 {
		commitHashes := make([][]byte, len(block.Commits))
		for i, commit := range block.Commits {
			commitHashes[i] = commit.Hash
		}

		merkleTree, err := crypto.NewMerkleTree(commitHashes)
		if err != nil {
			return fmt.Errorf("failed to verify merkle tree: %w", err)
		}

		if !bytes.Equal(merkleTree.GetRoot(), block.StateRoot) {
			return errors.New("invalid state root")
		}
	}

	// Validate each commit
	for _, commit := range block.Commits {
		if err := e.validateCommit(&commit); err != nil {
			return fmt.Errorf("invalid commit %s: %w", commit.ID, err)
		}
	}

	return nil
}

// VoteOnBlock allows validators to vote on a block
func (e *EnhancedConsensusEngine) VoteOnBlock(voter string, block *Block, approve bool) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	validator, exists := e.validators[voter]
	if !exists {
		return errors.New("voter not found")
	}

	signer, exists := e.signers[voter]
	if !exists {
		return errors.New("voter signer not found")
	}

	if !approve {
		// For now, just don't sign if not approving
		return nil
	}

	// Sign the block hash
	signature, err := signer.Sign(block.Hash)
	if err != nil {
		return fmt.Errorf("failed to sign block: %w", err)
	}

	block.Signatures[voter] = signature

	// Check if we have enough signatures (2/3 of total stake)
	signedStake := big.NewInt(0)
	totalStake := big.NewInt(0)

	for addr, v := range e.validators {
		if v.IsActive {
			totalStake.Add(totalStake, v.TotalStake)
			if _, signed := block.Signatures[addr]; signed {
				signedStake.Add(signedStake, v.TotalStake)
			}
		}
	}

	// Check if we have 2/3 majority
	threshold := new(big.Int).Mul(totalStake, big.NewInt(2))
	threshold.Div(threshold, big.NewInt(3))

	if signedStake.Cmp(threshold) >= 0 {
		// Block is finalized
		e.finalizeBlock(block)
	}

	return nil
}

// finalizeBlock adds a block to the chain
func (e *EnhancedConsensusEngine) finalizeBlock(block *Block) {
	e.blockChain = append(e.blockChain, block)
	e.currentRound++
	e.lastBlockTime = block.Timestamp

	// Update validator contributions
	for _, commit := range block.Commits {
		if validator, exists := e.validators[commit.Author]; exists {
			validator.RecentContribs = append(validator.RecentContribs, Contribution{
				ID:           commit.ID,
				Timestamp:    commit.Timestamp,
				QualityScore: commit.QualityScore,
			})
			validator.LastActivity = time.Now()

			// Update reputation based on quality
			e.reputationTracker.UpdateReputation(commit.Author, commit.QualityScore)
		}
	}

	// Check epoch transition
	if int64(len(e.blockChain))%e.epochLength == 0 {
		e.transitionEpoch()
	}

	// Store Merkle root for this block
	e.merkleRoots[block.Height] = block.StateRoot
}

// transitionEpoch handles epoch transitions
func (e *EnhancedConsensusEngine) transitionEpoch() {
	e.currentEpoch++

	// Generate new epoch seed using VRF
	seedData := make([]byte, 40)
	copy(seedData, e.epochSeed)
	binary.BigEndian.PutUint64(seedData[32:], uint64(e.currentEpoch))

	hash := sha256.Sum256(seedData)
	e.epochSeed = hash[:]

	// Clear old contributions
	for _, v := range e.validators {
		if len(v.RecentContribs) > 10 {
			v.RecentContribs = v.RecentContribs[len(v.RecentContribs)-10:]
		}
	}

	// Apply reputation decay
	for addr := range e.validators {
		e.reputationTracker.ApplyDecay(addr)
	}
}

// calculateEnhancedStake calculates total stake with all factors
func (e *EnhancedConsensusEngine) calculateEnhancedStake(v *EnhancedValidator) {
	// Get current reputation
	reputation := e.reputationTracker.GetReputation(v.Address)
	
	// Calculate reputation multiplier (0.5 to 2.0)
	repMultiplier := math.Max(0.5, math.Min(2.0, reputation/5.0))

	// Calculate contribution bonus
	contribBonus := 1.0
	if len(v.RecentContribs) > 0 {
		totalQuality := 0.0
		for _, c := range v.RecentContribs {
			totalQuality += c.QualityScore
		}
		avgQuality := totalQuality / float64(len(v.RecentContribs))
		contribBonus = 1.0 + (avgQuality / 100.0)
	}

	// Calculate total stake
	baseStake := new(big.Float).SetInt(v.TokenStake)
	totalStake := new(big.Float).Mul(baseStake, big.NewFloat(repMultiplier))
	totalStake = new(big.Float).Mul(totalStake, big.NewFloat(contribBonus))

	v.TotalStake, _ = totalStake.Int(nil)
}

// calculateBlockHash computes the hash of a block
func (e *EnhancedConsensusEngine) calculateBlockHash(block *Block) []byte {
	data := make([]byte, 0)
	
	heightBytes := make([]byte, 8)
	binary.BigEndian.PutUint64(heightBytes, uint64(block.Height))
	data = append(data, heightBytes...)
	
	data = append(data, block.PrevHash...)
	
	timeBytes := make([]byte, 8)
	binary.BigEndian.PutUint64(timeBytes, uint64(block.Timestamp.Unix()))
	data = append(data, timeBytes...)
	
	data = append(data, []byte(block.Proposer)...)
	data = append(data, block.VRFProof...)
	data = append(data, block.StateRoot...)

	hash := sha256.Sum256(data)
	return hash[:]
}

// serializeCommit serializes a commit for signing
func (e *EnhancedConsensusEngine) serializeCommit(commit *Commit) []byte {
	data := make([]byte, 0)
	data = append(data, []byte(commit.ID)...)
	data = append(data, []byte(commit.Author)...)
	data = append(data, commit.Hash...)
	
	timeBytes := make([]byte, 8)
	binary.BigEndian.PutUint64(timeBytes, uint64(commit.Timestamp.Unix()))
	data = append(data, timeBytes...)
	
	data = append(data, []byte(commit.Message)...)
	
	return data
}

// validateCommit validates a single commit
func (e *EnhancedConsensusEngine) validateCommit(commit *Commit) error {
	if commit.ID == "" {
		return errors.New("commit ID is empty")
	}

	if commit.Author == "" {
		return errors.New("commit author is empty")
	}

	if len(commit.Hash) != 32 {
		return errors.New("invalid commit hash length")
	}

	// Verify commit signature if author is a validator
	if validator, exists := e.validators[commit.Author]; exists {
		commitData := e.serializeCommit(commit)
		if !ed25519.Verify(validator.PublicKey, commitData, commit.Signature) {
			return errors.New("invalid commit signature")
		}
	}

	// Verify quality score is reasonable
	if commit.QualityScore < 0 || commit.QualityScore > 100 {
		return errors.New("invalid quality score")
	}

	return nil
}

// GetBlockByHeight returns a block at the specified height
func (e *EnhancedConsensusEngine) GetBlockByHeight(height int64) (*Block, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if height < 0 || height >= int64(len(e.blockChain)) {
		return nil, errors.New("block not found")
	}

	return e.blockChain[height], nil
}

// GetLatestBlock returns the most recent block
func (e *EnhancedConsensusEngine) GetLatestBlock() *Block {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if len(e.blockChain) == 0 {
		return nil
	}

	return e.blockChain[len(e.blockChain)-1]
}

// GetChainHeight returns the current blockchain height
func (e *EnhancedConsensusEngine) GetChainHeight() int64 {
	e.mu.RLock()
	defer e.mu.RUnlock()

	return int64(len(e.blockChain))
}

// GetValidatorInfo returns information about a validator
func (e *EnhancedConsensusEngine) GetValidatorInfo(address string) (*EnhancedValidator, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	validator, exists := e.validators[address]
	if !exists {
		return nil, errors.New("validator not found")
	}

	return validator, nil
}

// GetCommittee selects a committee of validators for consensus
func (e *EnhancedConsensusEngine) GetCommittee() ([]*EnhancedValidator, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	// Get all active validators
	activeValidators := make([]*EnhancedValidator, 0)
	for _, v := range e.validators {
		if v.IsActive {
			activeValidators = append(activeValidators, v)
		}
	}

	if len(activeValidators) < e.committeeSize {
		return activeValidators, nil
	}

	// Sort by VRF output for deterministic selection
	type validatorVRF struct {
		validator *EnhancedValidator
		vrfValue  *big.Int
	}

	vrfOutputs := make([]validatorVRF, 0)
	for addr, v := range e.validators {
		if !v.IsActive {
			continue
		}

		vrf := e.vrfs[addr]
		if vrf == nil {
			continue
		}

		// Generate VRF for committee selection
		message := append(e.epochSeed, []byte("committee")...)
		output, err := vrf.Prove(message)
		if err != nil {
			continue
		}

		vrfOutputs = append(vrfOutputs, validatorVRF{
			validator: v,
			vrfValue:  output.GetRandomness(),
		})
	}

	// Sort by VRF value
	sort.Slice(vrfOutputs, func(i, j int) bool {
		return vrfOutputs[i].vrfValue.Cmp(vrfOutputs[j].vrfValue) < 0
	})

	// Select top committee members
	committee := make([]*EnhancedValidator, 0, e.committeeSize)
	for i := 0; i < e.committeeSize && i < len(vrfOutputs); i++ {
		committee = append(committee, vrfOutputs[i].validator)
	}

	return committee, nil
}

// SlashValidator applies slashing to a misbehaving validator
func (e *EnhancedConsensusEngine) SlashValidator(address string, reason SlashingReason, evidence string) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	validator, exists := e.validators[address]
	if !exists {
		return errors.New("validator not found")
	}

	// Calculate slashing amount based on reason
	slashAmount := e.calculateSlashAmount(validator.TokenStake, reason)

	// Apply slashing
	validator.TokenStake.Sub(validator.TokenStake, slashAmount)

	// Record slashing event
	event := SlashingEvent{
		Timestamp: time.Now(),
		Reason:    reason,
		Amount:    slashAmount,
		Evidence:  evidence,
	}
	validator.SlashingEvents = append(validator.SlashingEvents, event)

	// Apply reputation penalty
	penalty := e.getReputationPenalty(reason)
	e.reputationTracker.ApplySlashing(address, penalty)

	// Deactivate if stake below minimum
	if validator.TokenStake.Cmp(e.minStakeRequired) < 0 {
		validator.IsActive = false
	}

	return nil
}

// calculateSlashAmount determines the amount to slash based on violation
func (e *EnhancedConsensusEngine) calculateSlashAmount(stake *big.Int, reason SlashingReason) *big.Int {
	percentage := 0.1 // Default 10%

	switch reason {
	case MaliciousCode:
		percentage = 0.5
	case FalseContribution:
		percentage = 0.3
	case DoubleProposal:
		percentage = 0.4
	case NetworkAttack:
		percentage = 0.7
	case QualityViolation:
		percentage = 0.2
	}

	slashAmount := new(big.Float).SetInt(stake)
	slashAmount.Mul(slashAmount, big.NewFloat(percentage))
	
	result, _ := slashAmount.Int(nil)
	return result
}

// getReputationPenalty returns the reputation penalty for a slashing reason
func (e *EnhancedConsensusEngine) getReputationPenalty(reason SlashingReason) float64 {
	switch reason {
	case MaliciousCode:
		return 0.5
	case FalseContribution:
		return 0.3
	case DoubleProposal:
		return 0.4
	case NetworkAttack:
		return 0.7
	case QualityViolation:
		return 0.2
	default:
		return 0.1
	}
}

// ExportGenesisState exports the current state for genesis block
func (e *EnhancedConsensusEngine) ExportGenesisState() *GenesisState {
	e.mu.RLock()
	defer e.mu.RUnlock()

	validators := make([]GenesisValidator, 0)
	for addr, v := range e.validators {
		validators = append(validators, GenesisValidator{
			Address:         addr,
			PublicKey:       hex.EncodeToString(v.PublicKey),
			TokenStake:      v.TokenStake.String(),
			ReputationScore: v.ReputationScore,
		})
	}

	return &GenesisState{
		Validators:   validators,
		EpochLength:  e.epochLength,
		BlockTime:    e.blockTime,
		MinStake:     e.minStakeRequired.String(),
		GenesisTime:  time.Now(),
	}
}

// GenesisState represents the initial state
type GenesisState struct {
	Validators   []GenesisValidator
	EpochLength  int64
	BlockTime    time.Duration
	MinStake     string
	GenesisTime  time.Time
}

// GenesisValidator represents a validator in genesis
type GenesisValidator struct {
	Address         string
	PublicKey       string
	TokenStake      string
	ReputationScore float64
}