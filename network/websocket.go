package network

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

// WebSocketServer handles real-time WebSocket connections
type WebSocketServer struct {
	mu         sync.RWMutex
	clients    map[*Client]bool
	broadcast  chan WSMessage
	register   chan *Client
	unregister chan *Client
	upgrader   websocket.Upgrader
}

// Client represents a WebSocket client
type Client struct {
	ID         string
	conn       *websocket.Conn
	send       chan []byte
	server     *WebSocketServer
	subscribed map[string]bool
}

// WSMessage represents a WebSocket message
type WSMessage struct {
	Type    string          `json:"type"`
	Channel string          `json:"channel,omitempty"`
	Data    json.RawMessage `json:"data"`
}

// SubscribeMessage for channel subscriptions
type SubscribeMessage struct {
	Type     string   `json:"type"`
	Channels []string `json:"channels"`
}

// BlockMessage for broadcasting blocks
type BlockMessage struct {
	Type  string      `json:"type"`
	Block interface{} `json:"block"`
}

// TransactionMessage for broadcasting transactions
type TransactionMessage struct {
	Type        string      `json:"type"`
	Transaction interface{} `json:"transaction"`
}

// NewWebSocketServer creates a new WebSocket server
func NewWebSocketServer() *WebSocketServer {
	return &WebSocketServer{
		clients:    make(map[*Client]bool),
		broadcast:  make(chan WSMessage, 256),
		register:   make(chan *Client),
		unregister: make(chan *Client),
		upgrader: websocket.Upgrader{
			ReadBufferSize:  1024,
			WriteBufferSize: 1024,
			CheckOrigin: func(r *http.Request) bool {
				// Configure CORS properly in production
				return true
			},
		},
	}
}

// Run starts the WebSocket server
func (ws *WebSocketServer) Run(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			ws.shutdown()
			return

		case client := <-ws.register:
			ws.mu.Lock()
			ws.clients[client] = true
			ws.mu.Unlock()
			log.Printf("WebSocket client connected: %s", client.ID)

		case client := <-ws.unregister:
			ws.mu.Lock()
			if _, ok := ws.clients[client]; ok {
				delete(ws.clients, client)
				close(client.send)
				ws.mu.Unlock()
				log.Printf("WebSocket client disconnected: %s", client.ID)
			} else {
				ws.mu.Unlock()
			}

		case message := <-ws.broadcast:
			ws.broadcastMessage(message)
		}
	}
}

// broadcastMessage sends message to subscribed clients
func (ws *WebSocketServer) broadcastMessage(message WSMessage) {
	ws.mu.RLock()
	defer ws.mu.RUnlock()

	data, err := json.Marshal(message)
	if err != nil {
		log.Printf("Error marshaling message: %v", err)
		return
	}

	for client := range ws.clients {
		// Only send if client is subscribed to this channel
		if message.Channel == "" || client.isSubscribed(message.Channel) {
			select {
			case client.send <- data:
			default:
				// Client's send channel is full, close it
				close(client.send)
				delete(ws.clients, client)
			}
		}
	}
}

// HandleWebSocket handles WebSocket upgrade requests
func (ws *WebSocketServer) HandleWebSocket(w http.ResponseWriter, r *http.Request) {
	conn, err := ws.upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("WebSocket upgrade error: %v", err)
		return
	}

	client := &Client{
		ID:         fmt.Sprintf("client_%d", time.Now().UnixNano()),
		conn:       conn,
		send:       make(chan []byte, 256),
		server:     ws,
		subscribed: make(map[string]bool),
	}

	ws.register <- client

	// Start goroutines for reading and writing
	go client.writePump()
	go client.readPump()
}

// BroadcastBlock broadcasts a new block to all subscribed clients
func (ws *WebSocketServer) BroadcastBlock(block interface{}) {
	data, _ := json.Marshal(BlockMessage{
		Type:  "block",
		Block: block,
	})

	ws.broadcast <- WSMessage{
		Type:    "block",
		Channel: "blocks",
		Data:    data,
	}
}

// BroadcastTransaction broadcasts a new transaction to all subscribed clients
func (ws *WebSocketServer) BroadcastTransaction(tx interface{}) {
	data, _ := json.Marshal(TransactionMessage{
		Type:        "transaction",
		Transaction: tx,
	})

	ws.broadcast <- WSMessage{
		Type:    "transaction",
		Channel: "transactions",
		Data:    data,
	}
}

// BroadcastValidatorUpdate broadcasts validator status changes
func (ws *WebSocketServer) BroadcastValidatorUpdate(validator interface{}) {
	data, _ := json.Marshal(map[string]interface{}{
		"type":      "validator",
		"validator": validator,
	})

	ws.broadcast <- WSMessage{
		Type:    "validator",
		Channel: "validators",
		Data:    data,
	}
}

// BroadcastPeerUpdate broadcasts peer connection changes
func (ws *WebSocketServer) BroadcastPeerUpdate(peer interface{}, action string) {
	data, _ := json.Marshal(map[string]interface{}{
		"type":   "peer",
		"action": action,
		"peer":   peer,
	})

	ws.broadcast <- WSMessage{
		Type:    "peer",
		Channel: "peers",
		Data:    data,
	}
}

// shutdown closes all client connections
func (ws *WebSocketServer) shutdown() {
	ws.mu.Lock()
	defer ws.mu.Unlock()

	for client := range ws.clients {
		close(client.send)
		client.conn.Close()
	}
	ws.clients = make(map[*Client]bool)
}

// Client methods

// readPump pumps messages from the websocket connection to the server
func (c *Client) readPump() {
	defer func() {
		c.server.unregister <- c
		c.conn.Close()
	}()

	c.conn.SetReadDeadline(time.Now().Add(60 * time.Second))
	c.conn.SetPongHandler(func(string) error {
		c.conn.SetReadDeadline(time.Now().Add(60 * time.Second))
		return nil
	})

	for {
		_, message, err := c.conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				log.Printf("WebSocket error: %v", err)
			}
			break
		}

		// Handle incoming messages
		c.handleMessage(message)
	}
}

// writePump pumps messages from the server to the websocket connection
func (c *Client) writePump() {
	ticker := time.NewTicker(54 * time.Second)
	defer func() {
		ticker.Stop()
		c.conn.Close()
	}()

	for {
		select {
		case message, ok := <-c.send:
			c.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if !ok {
				// Server closed the channel
				c.conn.WriteMessage(websocket.CloseMessage, []byte{})
				return
			}

			w, err := c.conn.NextWriter(websocket.TextMessage)
			if err != nil {
				return
			}
			w.Write(message)

			// Add queued messages to the current websocket message
			n := len(c.send)
			for i := 0; i < n; i++ {
				w.Write([]byte{'\n'})
				w.Write(<-c.send)
			}

			if err := w.Close(); err != nil {
				return
			}

		case <-ticker.C:
			c.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if err := c.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}
}

// handleMessage processes incoming client messages
func (c *Client) handleMessage(data []byte) {
	var msg WSMessage
	if err := json.Unmarshal(data, &msg); err != nil {
		log.Printf("Error unmarshaling message: %v", err)
		return
	}

	switch msg.Type {
	case "subscribe":
		var sub SubscribeMessage
		if err := json.Unmarshal(data, &sub); err != nil {
			log.Printf("Error unmarshaling subscribe message: %v", err)
			return
		}
		c.subscribe(sub.Channels)

	case "unsubscribe":
		var sub SubscribeMessage
		if err := json.Unmarshal(data, &sub); err != nil {
			log.Printf("Error unmarshaling unsubscribe message: %v", err)
			return
		}
		c.unsubscribe(sub.Channels)

	case "ping":
		// Send pong response
		response, _ := json.Marshal(WSMessage{Type: "pong"})
		select {
		case c.send <- response:
		default:
		}

	default:
		log.Printf("Unknown message type: %s", msg.Type)
	}
}

// subscribe adds channels to client's subscription list
func (c *Client) subscribe(channels []string) {
	for _, channel := range channels {
		c.subscribed[channel] = true
		log.Printf("Client %s subscribed to %s", c.ID, channel)
	}

	// Send confirmation
	response, _ := json.Marshal(map[string]interface{}{
		"type":     "subscribed",
		"channels": channels,
	})

	select {
	case c.send <- response:
	default:
	}
}

// unsubscribe removes channels from client's subscription list
func (c *Client) unsubscribe(channels []string) {
	for _, channel := range channels {
		delete(c.subscribed, channel)
		log.Printf("Client %s unsubscribed from %s", c.ID, channel)
	}

	// Send confirmation
	response, _ := json.Marshal(map[string]interface{}{
		"type":     "unsubscribed",
		"channels": channels,
	})

	select {
	case c.send <- response:
	default:
	}
}

// isSubscribed checks if client is subscribed to a channel
func (c *Client) isSubscribed(channel string) bool {
	return c.subscribed[channel]
}