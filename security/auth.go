// Package security implements authentication and security features
package security

import (
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"sync"
	"time"

	"golang.org/x/crypto/bcrypt"
	"golang.org/x/crypto/ed25519"
)

// TokenManager manages JWT-like tokens for API authentication
type TokenManager struct {
	mu           sync.RWMutex
	secretKey    []byte
	tokens       map[string]*Token
	refreshTokens map[string]*RefreshToken
	maxTokenAge  time.Duration
	maxRefreshAge time.Duration
}

// Token represents an API access token
type Token struct {
	ID        string    `json:"id"`
	Subject   string    `json:"sub"`
	IssuedAt  time.Time `json:"iat"`
	ExpiresAt time.Time `json:"exp"`
	Scope     []string  `json:"scope"`
	ClientID  string    `json:"client_id"`
}

// RefreshToken for token renewal
type RefreshToken struct {
	ID        string
	TokenID   string
	Subject   string
	ExpiresAt time.Time
}

// RateLimiter implements token bucket rate limiting
type RateLimiter struct {
	mu      sync.RWMutex
	buckets map[string]*TokenBucket
	rate    int           // tokens per interval
	burst   int           // max burst size
	window  time.Duration // time window
}

// TokenBucket for rate limiting
type TokenBucket struct {
	tokens    int
	lastRefill time.Time
}

// APIKey represents an API key for service authentication
type APIKey struct {
	Key         string    `json:"key"`
	Secret      string    `json:"-"`
	Name        string    `json:"name"`
	Permissions []string  `json:"permissions"`
	CreatedAt   time.Time `json:"created_at"`
	LastUsed    time.Time `json:"last_used"`
	Active      bool      `json:"active"`
}

// SecurityConfig holds security configuration
type SecurityConfig struct {
	EnableAuth      bool
	EnableRateLimit bool
	EnableTLS       bool
	SecretKey       string
	TokenExpiry     time.Duration
	RefreshExpiry   time.Duration
	RateLimit       int
	RateBurst       int
	RateWindow      time.Duration
	AllowedOrigins  []string
	TrustedProxies  []string
}

// NewTokenManager creates a new token manager
func NewTokenManager(secretKey string) *TokenManager {
	key := []byte(secretKey)
	if len(key) < 32 {
		// Pad or generate a proper key
		paddedKey := make([]byte, 32)
		copy(paddedKey, key)
		key = paddedKey
	}

	return &TokenManager{
		secretKey:     key,
		tokens:        make(map[string]*Token),
		refreshTokens: make(map[string]*RefreshToken),
		maxTokenAge:   1 * time.Hour,
		maxRefreshAge: 24 * time.Hour,
	}
}

// GenerateToken creates a new access token
func (tm *TokenManager) GenerateToken(subject, clientID string, scope []string) (*Token, string, error) {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	tokenID := generateRandomID()
	now := time.Now()

	token := &Token{
		ID:        tokenID,
		Subject:   subject,
		IssuedAt:  now,
		ExpiresAt: now.Add(tm.maxTokenAge),
		Scope:     scope,
		ClientID:  clientID,
	}

	// Create token string
	tokenData, err := json.Marshal(token)
	if err != nil {
		return nil, "", err
	}

	signature := tm.sign(tokenData)
	tokenString := base64.URLEncoding.EncodeToString(tokenData) + "." + signature

	tm.tokens[tokenID] = token
	return token, tokenString, nil
}

// GenerateRefreshToken creates a refresh token
func (tm *TokenManager) GenerateRefreshToken(tokenID, subject string) (*RefreshToken, string, error) {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	refreshID := generateRandomID()

	refreshToken := &RefreshToken{
		ID:        refreshID,
		TokenID:   tokenID,
		Subject:   subject,
		ExpiresAt: time.Now().Add(tm.maxRefreshAge),
	}

	tm.refreshTokens[refreshID] = refreshToken
	return refreshToken, refreshID, nil
}

// ValidateToken validates an access token
func (tm *TokenManager) ValidateToken(tokenString string) (*Token, error) {
	parts := strings.Split(tokenString, ".")
	if len(parts) != 2 {
		return nil, errors.New("invalid token format")
	}

	tokenData, err := base64.URLEncoding.DecodeString(parts[0])
	if err != nil {
		return nil, err
	}

	// Verify signature
	expectedSig := tm.sign(tokenData)
	if parts[1] != expectedSig {
		return nil, errors.New("invalid token signature")
	}

	var token Token
	if err := json.Unmarshal(tokenData, &token); err != nil {
		return nil, err
	}

	// Check expiration
	if time.Now().After(token.ExpiresAt) {
		return nil, errors.New("token expired")
	}

	tm.mu.RLock()
	defer tm.mu.RUnlock()

	// Verify token exists and hasn't been revoked
	if storedToken, exists := tm.tokens[token.ID]; exists {
		return storedToken, nil
	}

	return nil, errors.New("token not found or revoked")
}

// RevokeToken revokes an access token
func (tm *TokenManager) RevokeToken(tokenID string) error {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	delete(tm.tokens, tokenID)
	return nil
}

// RefreshAccessToken creates a new access token from refresh token
func (tm *TokenManager) RefreshAccessToken(refreshTokenID string) (*Token, string, error) {
	tm.mu.Lock()
	refreshToken, exists := tm.refreshTokens[refreshTokenID]
	tm.mu.Unlock()

	if !exists {
		return nil, "", errors.New("refresh token not found")
	}

	if time.Now().After(refreshToken.ExpiresAt) {
		return nil, "", errors.New("refresh token expired")
	}

	// Generate new access token
	return tm.GenerateToken(refreshToken.Subject, "refresh", []string{"api"})
}

// sign creates HMAC signature
func (tm *TokenManager) sign(data []byte) string {
	h := hmac.New(sha256.New, tm.secretKey)
	h.Write(data)
	return base64.URLEncoding.EncodeToString(h.Sum(nil))
}

// NewRateLimiter creates a new rate limiter
func NewRateLimiter(rate, burst int, window time.Duration) *RateLimiter {
	return &RateLimiter{
		buckets: make(map[string]*TokenBucket),
		rate:    rate,
		burst:   burst,
		window:  window,
	}
}

// Allow checks if request is allowed
func (rl *RateLimiter) Allow(key string) bool {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	bucket, exists := rl.buckets[key]
	if !exists {
		bucket = &TokenBucket{
			tokens:    rl.burst,
			lastRefill: time.Now(),
		}
		rl.buckets[key] = bucket
	}

	// Refill tokens based on time passed
	now := time.Now()
	elapsed := now.Sub(bucket.lastRefill)
	tokensToAdd := int(elapsed.Seconds() * float64(rl.rate) / rl.window.Seconds())

	bucket.tokens = min(bucket.tokens+tokensToAdd, rl.burst)
	bucket.lastRefill = now

	// Check if request is allowed
	if bucket.tokens > 0 {
		bucket.tokens--
		return true
	}

	return false
}

// Middleware provides HTTP middleware for authentication
type Middleware struct {
	tokenManager *TokenManager
	rateLimiter  *RateLimiter
	config       *SecurityConfig
}

// NewMiddleware creates security middleware
func NewMiddleware(config *SecurityConfig) *Middleware {
	return &Middleware{
		tokenManager: NewTokenManager(config.SecretKey),
		rateLimiter:  NewRateLimiter(config.RateLimit, config.RateBurst, config.RateWindow),
		config:       config,
	}
}

// Authenticate validates request authentication
func (m *Middleware) Authenticate(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if !m.config.EnableAuth {
			next(w, r)
			return
		}

		// Extract token from header
		authHeader := r.Header.Get("Authorization")
		if authHeader == "" {
			http.Error(w, "Missing authorization header", http.StatusUnauthorized)
			return
		}

		parts := strings.Split(authHeader, " ")
		if len(parts) != 2 || parts[0] != "Bearer" {
			http.Error(w, "Invalid authorization header", http.StatusUnauthorized)
			return
		}

		token, err := m.tokenManager.ValidateToken(parts[1])
		if err != nil {
			http.Error(w, err.Error(), http.StatusUnauthorized)
			return
		}

		// Add token info to request context
		r.Header.Set("X-User-ID", token.Subject)
		r.Header.Set("X-Client-ID", token.ClientID)

		next(w, r)
	}
}

// RateLimit applies rate limiting
func (m *Middleware) RateLimit(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if !m.config.EnableRateLimit {
			next(w, r)
			return
		}

		// Use IP address or user ID as key
		key := r.RemoteAddr
		if userID := r.Header.Get("X-User-ID"); userID != "" {
			key = userID
		}

		if !m.rateLimiter.Allow(key) {
			http.Error(w, "Rate limit exceeded", http.StatusTooManyRequests)
			return
		}

		next(w, r)
	}
}

// CORS handles cross-origin requests
func (m *Middleware) CORS(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")
		allowed := false

		// Check if origin is allowed
		for _, allowedOrigin := range m.config.AllowedOrigins {
			if allowedOrigin == "*" || allowedOrigin == origin {
				allowed = true
				break
			}
		}

		if allowed {
			w.Header().Set("Access-Control-Allow-Origin", origin)
			w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
			w.Header().Set("Access-Control-Allow-Credentials", "true")
		}

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		next(w, r)
	}
}

// SecurityHeaders adds security headers
func (m *Middleware) SecurityHeaders(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("X-Frame-Options", "DENY")
		w.Header().Set("X-XSS-Protection", "1; mode=block")
		w.Header().Set("Referrer-Policy", "strict-origin-when-cross-origin")

		if m.config.EnableTLS {
			w.Header().Set("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
		}

		next(w, r)
	}
}

// Helper functions

func generateRandomID() string {
	b := make([]byte, 16)
	rand.Read(b)
	return hex.EncodeToString(b)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// HashPassword creates bcrypt hash of password
func HashPassword(password string) (string, error) {
	bytes, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	return string(bytes), err
}

// CheckPassword compares password with hash
func CheckPassword(password, hash string) bool {
	err := bcrypt.CompareHashAndPassword([]byte(hash), []byte(password))
	return err == nil
}

// GenerateAPIKey creates a new API key
func GenerateAPIKey(name string, permissions []string) (*APIKey, error) {
	keyBytes := make([]byte, 32)
	if _, err := rand.Read(keyBytes); err != nil {
		return nil, err
	}

	secretBytes := make([]byte, 32)
	if _, err := rand.Read(secretBytes); err != nil {
		return nil, err
	}

	return &APIKey{
		Key:         hex.EncodeToString(keyBytes),
		Secret:      hex.EncodeToString(secretBytes),
		Name:        name,
		Permissions: permissions,
		CreatedAt:   time.Now(),
		Active:      true,
	}, nil
}

// ValidateAPIKey checks if API key is valid
func ValidateAPIKey(key, secret string, apiKeys map[string]*APIKey) bool {
	if apiKey, exists := apiKeys[key]; exists {
		return apiKey.Active && apiKey.Secret == secret
	}
	return false
}

// SignMessage creates Ed25519 signature
func SignMessage(privateKey ed25519.PrivateKey, message []byte) []byte {
	return ed25519.Sign(privateKey, message)
}

// VerifySignature verifies Ed25519 signature
func VerifySignature(publicKey ed25519.PublicKey, message, signature []byte) bool {
	return ed25519.Verify(publicKey, message, signature)
}