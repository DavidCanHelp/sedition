package logging

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"time"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"gopkg.in/natefinch/lumberjack.v2"
)

// LogLevel represents the severity of a log message
type LogLevel int

const (
	DebugLevel LogLevel = iota
	InfoLevel
	WarnLevel
	ErrorLevel
	FatalLevel
)

// Logger is the main logging interface
type Logger interface {
	Debug(msg string, fields ...Field)
	Info(msg string, fields ...Field)
	Warn(msg string, fields ...Field)
	Error(msg string, fields ...Field)
	Fatal(msg string, fields ...Field)
	With(fields ...Field) Logger
	WithContext(ctx context.Context) Logger
	Sync() error
}

// Field represents a structured logging field
type Field struct {
	Key   string
	Value interface{}
}

// StructuredLogger implements the Logger interface with structured logging
type StructuredLogger struct {
	zap    *zap.Logger
	sugar  *zap.SugaredLogger
	config *LogConfig
	mu     sync.RWMutex
}

// LogConfig contains logger configuration
type LogConfig struct {
	Level          string        `json:"level"`
	Format         string        `json:"format"`        // "json" or "console"
	OutputPath     string        `json:"output_path"`   // stdout, stderr, or file path
	ErrorPath      string        `json:"error_path"`    // stderr or file path
	EnableRotation bool          `json:"enable_rotation"`
	MaxSize        int           `json:"max_size_mb"`   // Maximum size in megabytes
	MaxBackups     int           `json:"max_backups"`   // Maximum number of old log files
	MaxAge         int           `json:"max_age_days"`  // Maximum days to retain old logs
	Compress       bool          `json:"compress"`      // Compress rotated files
	EnableCaller   bool          `json:"enable_caller"` // Include caller information
	EnableStack    bool          `json:"enable_stack"`  // Include stack trace for errors
	SampleRate     int           `json:"sample_rate"`   // Sample rate for debug logs
	Module         string        `json:"module"`        // Module name for filtering
}

// DefaultLogConfig returns default logging configuration
func DefaultLogConfig() *LogConfig {
	return &LogConfig{
		Level:          "info",
		Format:         "json",
		OutputPath:     "stdout",
		ErrorPath:      "stderr",
		EnableRotation: true,
		MaxSize:        100,
		MaxBackups:     5,
		MaxAge:         30,
		Compress:       true,
		EnableCaller:   true,
		EnableStack:    true,
		SampleRate:     100,
	}
}

// Global logger instance
var (
	globalLogger *StructuredLogger
	once         sync.Once
)

// Initialize initializes the global logger
func Initialize(config *LogConfig) error {
	var err error
	once.Do(func() {
		globalLogger, err = NewStructuredLogger(config)
	})
	return err
}

// GetLogger returns the global logger instance
func GetLogger() Logger {
	if globalLogger == nil {
		// Initialize with defaults if not already done
		Initialize(DefaultLogConfig())
	}
	return globalLogger
}

// NewStructuredLogger creates a new structured logger
func NewStructuredLogger(config *LogConfig) (*StructuredLogger, error) {
	// Parse log level
	level, err := parseLevel(config.Level)
	if err != nil {
		return nil, fmt.Errorf("invalid log level: %w", err)
	}

	// Create encoder config
	encoderConfig := zapcore.EncoderConfig{
		TimeKey:        "timestamp",
		LevelKey:       "level",
		NameKey:        "logger",
		CallerKey:      "caller",
		FunctionKey:    zapcore.OmitKey,
		MessageKey:     "message",
		StacktraceKey:  "stacktrace",
		LineEnding:     zapcore.DefaultLineEnding,
		EncodeLevel:    zapcore.CapitalLevelEncoder,
		EncodeTime:     zapcore.ISO8601TimeEncoder,
		EncodeDuration: zapcore.StringDurationEncoder,
		EncodeCaller:   zapcore.ShortCallerEncoder,
	}

	// Choose encoder based on format
	var encoder zapcore.Encoder
	if config.Format == "console" {
		encoderConfig.EncodeLevel = zapcore.CapitalColorLevelEncoder
		encoder = zapcore.NewConsoleEncoder(encoderConfig)
	} else {
		encoder = zapcore.NewJSONEncoder(encoderConfig)
	}

	// Setup output writers
	writers := []zapcore.WriteSyncer{}

	// Main output
	mainWriter, err := getWriter(config.OutputPath, config)
	if err != nil {
		return nil, fmt.Errorf("failed to create main writer: %w", err)
	}
	writers = append(writers, mainWriter)

	// Error output (for error level and above)
	if config.ErrorPath != "" && config.ErrorPath != config.OutputPath {
		errorWriter, err := getWriter(config.ErrorPath, config)
		if err != nil {
			return nil, fmt.Errorf("failed to create error writer: %w", err)
		}
		writers = append(writers, errorWriter)
	}

	// Create core
	core := zapcore.NewTee(
		zapcore.NewCore(encoder, writers[0], level),
	)

	// Add sampling for debug logs if configured
	if config.SampleRate > 0 && level == zapcore.DebugLevel {
		core = zapcore.NewSamplerWithOptions(
			core,
			time.Second,
			config.SampleRate,
			config.SampleRate/10,
		)
	}

	// Build logger options
	options := []zap.Option{
		zap.AddStacktrace(zapcore.ErrorLevel),
	}

	if config.EnableCaller {
		options = append(options, zap.AddCaller())
	}

	if config.Module != "" {
		options = append(options, zap.Fields(zap.String("module", config.Module)))
	}

	// Create logger
	zapLogger := zap.New(core, options...)

	return &StructuredLogger{
		zap:    zapLogger,
		sugar:  zapLogger.Sugar(),
		config: config,
	}, nil
}

// getWriter creates a WriteSyncer for the given path
func getWriter(path string, config *LogConfig) (zapcore.WriteSyncer, error) {
	switch path {
	case "stdout":
		return zapcore.AddSync(os.Stdout), nil
	case "stderr":
		return zapcore.AddSync(os.Stderr), nil
	default:
		// File output with rotation if enabled
		if config.EnableRotation {
			return zapcore.AddSync(&lumberjack.Logger{
				Filename:   path,
				MaxSize:    config.MaxSize,
				MaxBackups: config.MaxBackups,
				MaxAge:     config.MaxAge,
				Compress:   config.Compress,
				LocalTime:  true,
			}), nil
		}

		// Simple file output
		file, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
		if err != nil {
			return nil, err
		}
		return zapcore.AddSync(file), nil
	}
}

// parseLevel parses string log level to zapcore.Level
func parseLevel(level string) (zapcore.Level, error) {
	switch level {
	case "debug":
		return zapcore.DebugLevel, nil
	case "info":
		return zapcore.InfoLevel, nil
	case "warn", "warning":
		return zapcore.WarnLevel, nil
	case "error":
		return zapcore.ErrorLevel, nil
	case "fatal":
		return zapcore.FatalLevel, nil
	default:
		return zapcore.InfoLevel, fmt.Errorf("unknown level: %s", level)
	}
}

// Logger interface implementation

func (l *StructuredLogger) Debug(msg string, fields ...Field) {
	l.zap.Debug(msg, convertFields(fields)...)
}

func (l *StructuredLogger) Info(msg string, fields ...Field) {
	l.zap.Info(msg, convertFields(fields)...)
}

func (l *StructuredLogger) Warn(msg string, fields ...Field) {
	l.zap.Warn(msg, convertFields(fields)...)
}

func (l *StructuredLogger) Error(msg string, fields ...Field) {
	l.zap.Error(msg, convertFields(fields)...)
}

func (l *StructuredLogger) Fatal(msg string, fields ...Field) {
	l.zap.Fatal(msg, convertFields(fields)...)
}

func (l *StructuredLogger) With(fields ...Field) Logger {
	return &StructuredLogger{
		zap:    l.zap.With(convertFields(fields)...),
		sugar:  l.sugar,
		config: l.config,
	}
}

func (l *StructuredLogger) WithContext(ctx context.Context) Logger {
	// Extract request ID or trace ID from context if available
	fields := []zap.Field{}

	if requestID, ok := ctx.Value("request_id").(string); ok {
		fields = append(fields, zap.String("request_id", requestID))
	}

	if traceID, ok := ctx.Value("trace_id").(string); ok {
		fields = append(fields, zap.String("trace_id", traceID))
	}

	if userID, ok := ctx.Value("user_id").(string); ok {
		fields = append(fields, zap.String("user_id", userID))
	}

	return &StructuredLogger{
		zap:    l.zap.With(fields...),
		sugar:  l.sugar,
		config: l.config,
	}
}

func (l *StructuredLogger) Sync() error {
	return l.zap.Sync()
}

// Helper functions

func convertFields(fields []Field) []zap.Field {
	zapFields := make([]zap.Field, len(fields))
	for i, f := range fields {
		zapFields[i] = zap.Any(f.Key, f.Value)
	}
	return zapFields
}

// Convenience functions for structured fields

func String(key, value string) Field {
	return Field{Key: key, Value: value}
}

func Int(key string, value int) Field {
	return Field{Key: key, Value: value}
}

func Int64(key string, value int64) Field {
	return Field{Key: key, Value: value}
}

func Float64(key string, value float64) Field {
	return Field{Key: key, Value: value}
}

func Bool(key string, value bool) Field {
	return Field{Key: key, Value: value}
}

func Time(key string, value time.Time) Field {
	return Field{Key: key, Value: value}
}

func Duration(key string, value time.Duration) Field {
	return Field{Key: key, Value: value}
}

func Error(err error) Field {
	return Field{Key: "error", Value: err.Error()}
}

func Any(key string, value interface{}) Field {
	return Field{Key: key, Value: value}
}

// Module-specific loggers

type ModuleLogger struct {
	logger Logger
	module string
}

func NewModuleLogger(module string) *ModuleLogger {
	return &ModuleLogger{
		logger: GetLogger().With(String("module", module)),
		module: module,
	}
}

func (m *ModuleLogger) Debug(msg string, fields ...Field) {
	m.logger.Debug(msg, fields...)
}

func (m *ModuleLogger) Info(msg string, fields ...Field) {
	m.logger.Info(msg, fields...)
}

func (m *ModuleLogger) Warn(msg string, fields ...Field) {
	m.logger.Warn(msg, fields...)
}

func (m *ModuleLogger) Error(msg string, fields ...Field) {
	m.logger.Error(msg, fields...)
}

func (m *ModuleLogger) Fatal(msg string, fields ...Field) {
	m.logger.Fatal(msg, fields...)
}

// Performance logger for tracking operation timings

type PerformanceLogger struct {
	logger    Logger
	operation string
	startTime time.Time
	fields    []Field
}

func StartOperation(operation string) *PerformanceLogger {
	return &PerformanceLogger{
		logger:    GetLogger(),
		operation: operation,
		startTime: time.Now(),
		fields:    []Field{String("operation", operation)},
	}
}

func (p *PerformanceLogger) AddField(field Field) *PerformanceLogger {
	p.fields = append(p.fields, field)
	return p
}

func (p *PerformanceLogger) Complete() {
	duration := time.Since(p.startTime)
	p.fields = append(p.fields, Duration("duration", duration))
	p.logger.Info("operation completed", p.fields...)
}

func (p *PerformanceLogger) Failed(err error) {
	duration := time.Since(p.startTime)
	p.fields = append(p.fields, Duration("duration", duration), Error(err))
	p.logger.Error("operation failed", p.fields...)
}

// Audit logger for security events

type AuditLogger struct {
	logger Logger
	writer io.Writer
}

func NewAuditLogger(auditFile string) (*AuditLogger, error) {
	file, err := os.OpenFile(auditFile, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return nil, err
	}

	return &AuditLogger{
		logger: GetLogger().With(String("type", "audit")),
		writer: file,
	}, nil
}

func (a *AuditLogger) LogEvent(event AuditEvent) {
	// Write to audit file
	data, _ := json.Marshal(event)
	a.writer.Write(append(data, '\n'))

	// Also log to main logger
	a.logger.Info("audit event",
		String("event_type", event.Type),
		String("user", event.User),
		String("action", event.Action),
		Any("details", event.Details),
	)
}

type AuditEvent struct {
	Timestamp time.Time              `json:"timestamp"`
	Type      string                 `json:"type"`
	User      string                 `json:"user"`
	Action    string                 `json:"action"`
	Resource  string                 `json:"resource"`
	Result    string                 `json:"result"`
	Details   map[string]interface{} `json:"details,omitempty"`
}

// Metrics logger for performance metrics

type MetricsLogger struct {
	logger   Logger
	interval time.Duration
	metrics  map[string]interface{}
	mu       sync.RWMutex
	stop     chan bool
}

func NewMetricsLogger(interval time.Duration) *MetricsLogger {
	ml := &MetricsLogger{
		logger:   GetLogger().With(String("type", "metrics")),
		interval: interval,
		metrics:  make(map[string]interface{}),
		stop:     make(chan bool),
	}

	go ml.logPeriodically()
	return ml
}

func (m *MetricsLogger) Record(key string, value interface{}) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.metrics[key] = value
}

func (m *MetricsLogger) logPeriodically() {
	ticker := time.NewTicker(m.interval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			m.logMetrics()
		case <-m.stop:
			return
		}
	}
}

func (m *MetricsLogger) logMetrics() {
	m.mu.RLock()
	metrics := make(map[string]interface{})
	for k, v := range m.metrics {
		metrics[k] = v
	}
	m.mu.RUnlock()

	// Add system metrics
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)

	metrics["memory_alloc_mb"] = memStats.Alloc / 1024 / 1024
	metrics["memory_sys_mb"] = memStats.Sys / 1024 / 1024
	metrics["num_goroutines"] = runtime.NumGoroutine()

	m.logger.Info("system metrics", Any("metrics", metrics))
}

func (m *MetricsLogger) Stop() {
	close(m.stop)
}

// Context logger for request tracing

type ContextLogger struct {
	requestID string
	logger    Logger
}

func WithRequestID(requestID string) *ContextLogger {
	return &ContextLogger{
		requestID: requestID,
		logger:    GetLogger().With(String("request_id", requestID)),
	}
}

func (c *ContextLogger) Log() Logger {
	return c.logger
}

// LogRotator handles log rotation manually

type LogRotator struct {
	currentFile *os.File
	basePath    string
	maxSize     int64
	maxFiles    int
	mu          sync.Mutex
}

func NewLogRotator(basePath string, maxSizeMB int, maxFiles int) *LogRotator {
	return &LogRotator{
		basePath: basePath,
		maxSize:  int64(maxSizeMB * 1024 * 1024),
		maxFiles: maxFiles,
	}
}

func (r *LogRotator) Write(p []byte) (n int, err error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.currentFile == nil {
		if err := r.openNewFile(); err != nil {
			return 0, err
		}
	}

	// Check if rotation is needed
	info, err := r.currentFile.Stat()
	if err == nil && info.Size()+int64(len(p)) > r.maxSize {
		r.rotate()
	}

	return r.currentFile.Write(p)
}

func (r *LogRotator) openNewFile() error {
	filename := fmt.Sprintf("%s.%s.log", r.basePath, time.Now().Format("2006-01-02-15-04-05"))
	file, err := os.OpenFile(filename, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return err
	}
	r.currentFile = file
	return nil
}

func (r *LogRotator) rotate() error {
	if r.currentFile != nil {
		r.currentFile.Close()
	}

	// Clean up old files if needed
	r.cleanOldFiles()

	return r.openNewFile()
}

func (r *LogRotator) cleanOldFiles() {
	pattern := fmt.Sprintf("%s.*.log", r.basePath)
	files, err := filepath.Glob(pattern)
	if err != nil {
		return
	}

	if len(files) > r.maxFiles {
		// Sort files by modification time and remove oldest
		for i := 0; i < len(files)-r.maxFiles; i++ {
			os.Remove(files[i])
		}
	}
}