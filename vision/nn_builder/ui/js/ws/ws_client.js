// WebSocket client, ported from race_nav/server/agv_dashboard/js/ws/ws_client.js
// {topic, data} envelope, auto-reconnect, singleton export.

class WebSocketClient {
  constructor() {
    this.ws = null;
    this.topicHandlers = {};
    this.reconnectDelay = 2000;
    this.isConnecting = false;
  }

  connect() {
    if (this.isConnecting) return;

    const serverIp = window.location.hostname;
    const port = window.location.port || 5050;
    const WS_URL = `ws://${serverIp}:${port}/ws`;

    console.log("Connecting WebSocket:", WS_URL);
    this.isConnecting = true;

    this.ws = new WebSocket(WS_URL);

    this.ws.onopen = () => {
      this.isConnecting = false;
      this.updateStatus("Connected", "status-good");
      this.send("/page/load", { page: window.location.pathname });
    };

    this.ws.onclose = () => {
      this.isConnecting = false;
      this.updateStatus("Disconnected", "status-bad");
      setTimeout(() => this.connect(), this.reconnectDelay);
    };

    this.ws.onerror = () => {
      this.isConnecting = false;
    };

    this.ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.topic && this.topicHandlers[msg.topic]) {
          this.topicHandlers[msg.topic](msg.data);
        }
      } catch (error) {
        console.error("Failed to parse WebSocket message:", error);
      }
    };
  }

  updateStatus(text, className) {
    const el = document.getElementById("ws-status");
    if (el) {
      el.textContent = text;
      el.className = className;
    }
  }

  subscribe(topic, callback) {
    this.topicHandlers[topic] = callback;
  }

  unsubscribe(topic) {
    delete this.topicHandlers[topic];
  }

  send(topic, data) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return false;
    this.ws.send(JSON.stringify({ topic, data }));
    return true;
  }

  isConnected() {
    return this.ws && this.ws.readyState === WebSocket.OPEN;
  }
}

const wsClient = new WebSocketClient();
wsClient.connect();

export default wsClient;
