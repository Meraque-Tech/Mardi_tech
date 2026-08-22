#pragma once

#include <opencv2/opencv.hpp>
#include <arpa/inet.h>
#include <unistd.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <atomic>
#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

// Minimal MJPEG-over-HTTP streamer: connect a browser to
// http://<host>:<port>/ to view frames pushed via push_frame().
//
// Camera capture and JPEG encoding never perform client network I/O. Each
// client has an independent sender thread which always takes the newest JPEG;
// a slow or half-open remote connection can therefore neither stall inference
// nor prevent healthy clients from receiving frames.
class MjpegServer {
public:
    bool start(int port) {
        listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
        if (listen_fd_ < 0) return false;

        int opt = 1;
        setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(port);

        if (bind(listen_fd_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0 ||
            listen(listen_fd_, 16) < 0) {
            close(listen_fd_);
            listen_fd_ = -1;
            return false;
        }

        running_ = true;
        accept_thread_ = std::thread(&MjpegServer::accept_loop, this);
        worker_thread_ = std::thread(&MjpegServer::worker_loop, this);
        return true;
    }

    // Cheap: only keep the most recent camera frame. If encoding falls behind,
    // intermediate frames are intentionally dropped rather than queued.
    void push_frame(const cv::Mat &frame) {
        if (active_clients_.load() == 0) return;
        {
            std::lock_guard<std::mutex> lock(frame_mutex_);
            frame.copyTo(latest_frame_);
            has_new_frame_ = true;
        }
        frame_cv_.notify_one();
    }

    void stop() {
        if (!running_.exchange(false)) return;

        frame_cv_.notify_all();
        encoded_cv_.notify_all();
        if (listen_fd_ >= 0) {
            shutdown(listen_fd_, SHUT_RDWR);
            close(listen_fd_);
            listen_fd_ = -1;
        }
        if (accept_thread_.joinable()) accept_thread_.join();

        // Wake a client currently blocked in send() before joining it.
        {
            std::lock_guard<std::mutex> lock(clients_mutex_);
            for (const auto &client : clients_) shutdown(client->fd, SHUT_RDWR);
        }

        if (worker_thread_.joinable()) worker_thread_.join();

        std::vector<std::shared_ptr<Client>> clients;
        {
            std::lock_guard<std::mutex> lock(clients_mutex_);
            clients.swap(clients_);
        }
        for (const auto &client : clients) {
            if (client->thread.joinable()) client->thread.join();
        }
    }

    ~MjpegServer() { stop(); }

private:
    struct Client {
        int fd = -1;
        std::string peer;
        std::atomic<bool> active{true};
        std::thread thread;
        uint64_t frames_sent = 0;
    };

    static bool send_all(int fd, const char *data, size_t len) {
        size_t sent = 0;
        while (sent < len) {
            const ssize_t n = send(fd, data + sent, len - sent, MSG_NOSIGNAL);
            if (n <= 0) {
                if (n == 0) errno = EPIPE;
                return false;
            }
            sent += static_cast<size_t>(n);
        }
        return true;
    }

    static std::string trim(const std::string &value) {
        const auto first = value.find_first_not_of(" \t");
        if (first == std::string::npos) return "";
        const auto last = value.find_last_not_of(" \t");
        return value.substr(first, last - first + 1);
    }

    static std::string lower(std::string value) {
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        return value;
    }

    static std::string log_safe(std::string value) {
        for (char &ch : value) {
            const auto byte = static_cast<unsigned char>(ch);
            if (byte < 32 || byte == 127) ch = ' ';
        }
        constexpr size_t max_log_length = 180;
        if (value.size() > max_log_length) {
            value.resize(max_log_length);
            value += "...";
        }
        return value;
    }

    static bool read_http_request(
        int fd, std::string &method, std::string &target,
        std::string &user_agent, std::string &failure) {
        constexpr size_t max_header_bytes = 16 * 1024;
        std::string request;
        request.reserve(2048);
        char chunk[2048];

        size_t header_end = std::string::npos;
        while ((header_end = request.find("\r\n\r\n")) == std::string::npos) {
            const ssize_t received = recv(fd, chunk, sizeof(chunk), 0);
            if (received > 0) {
                request.append(chunk, static_cast<size_t>(received));
                if (request.size() > max_header_bytes) {
                    failure = "request headers too large";
                    return false;
                }
                continue;
            }
            if (received == 0) {
                failure = "client closed before request";
                return false;
            }
            if (errno == EINTR) continue;
            failure = (errno == EAGAIN || errno == EWOULDBLOCK)
                ? "request timeout"
                : std::string("request read failed: ") + std::strerror(errno);
            return false;
        }

        const size_t request_line_end = request.find("\r\n");
        if (request_line_end == std::string::npos || request_line_end > header_end) {
            failure = "malformed request line";
            return false;
        }

        std::string version;
        std::istringstream request_line(request.substr(0, request_line_end));
        if (!(request_line >> method >> target >> version) ||
            version.rfind("HTTP/", 0) != 0 || target.empty() || target[0] != '/') {
            failure = "malformed request line";
            return false;
        }

        size_t cursor = request_line_end + 2;
        while (cursor < header_end) {
            size_t line_end = request.find("\r\n", cursor);
            if (line_end == std::string::npos || line_end > header_end) line_end = header_end;
            const std::string line = request.substr(cursor, line_end - cursor);
            const size_t colon = line.find(':');
            if (colon != std::string::npos &&
                lower(trim(line.substr(0, colon))) == "user-agent") {
                user_agent = trim(line.substr(colon + 1));
            }
            cursor = line_end + 2;
        }
        return true;
    }

    static bool send_empty_response(
        int fd, const char *status, const char *extra_headers = "") {
        const std::string response =
            std::string("HTTP/1.1 ") + status + "\r\n" +
            "Access-Control-Allow-Origin: *\r\n" +
            "Access-Control-Allow-Methods: GET, OPTIONS\r\n" +
            extra_headers +
            "Content-Length: 0\r\n"
            "Connection: close\r\n\r\n";
        return send_all(fd, response.data(), response.size());
    }

    void worker_loop() {
        cv::Mat frame;
        std::vector<uchar> jpg;
        while (running_) {
            {
                std::unique_lock<std::mutex> lock(frame_mutex_);
                frame_cv_.wait(lock, [this] { return has_new_frame_ || !running_; });
                if (!running_) break;
                latest_frame_.copyTo(frame);
                has_new_frame_ = false;
            }

            if (active_clients_.load() == 0) continue;
            if (!cv::imencode(".jpg", frame, jpg, {cv::IMWRITE_JPEG_QUALITY, 80})) {
                std::cerr << "MJPEG frame encoding failed" << std::endl;
                continue;
            }

            {
                std::lock_guard<std::mutex> lock(encoded_mutex_);
                latest_jpg_ = jpg;
                ++encoded_sequence_;
            }
            encoded_cv_.notify_all();
        }
    }

    void client_loop(const std::shared_ptr<Client> &client) {
        std::string method;
        std::string target;
        std::string user_agent;
        std::string request_failure;
        if (!read_http_request(
                client->fd, method, target, user_agent, request_failure)) {
            errno = 0;
            finish_client(client, request_failure.c_str());
            return;
        }

        std::cout << "MJPEG request: " << client->peer
                  << " method=" << log_safe(method)
                  << " target=" << log_safe(target)
                  << " user_agent=\"" << log_safe(user_agent) << "\""
                  << std::endl;

        if (method == "OPTIONS") {
            send_empty_response(
                client->fd, "204 No Content",
                "Access-Control-Allow-Headers: *\r\n");
            errno = 0;
            finish_client(client, "preflight complete");
            return;
        }
        if (method != "GET") {
            send_empty_response(client->fd, "405 Method Not Allowed", "Allow: GET, OPTIONS\r\n");
            errno = 0;
            finish_client(client, "method not allowed");
            return;
        }

        static const char *response =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
            "Cache-Control: no-store, no-cache, must-revalidate\r\n"
            "Pragma: no-cache\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Methods: GET, OPTIONS\r\n"
            "Cross-Origin-Resource-Policy: cross-origin\r\n"
            "X-Content-Type-Options: nosniff\r\n"
            "Connection: close\r\n\r\n";

        if (!send_all(client->fd, response, std::strlen(response))) {
            finish_client(client, "response send failed");
            return;
        }

        uint64_t seen_sequence = 0;
        while (running_ && client->active.load()) {
            std::vector<uchar> jpg;
            uint64_t sequence = 0;
            {
                std::unique_lock<std::mutex> lock(encoded_mutex_);
                encoded_cv_.wait(lock, [this, seen_sequence] {
                    return encoded_sequence_ != seen_sequence || !running_;
                });
                if (!running_) break;
                sequence = encoded_sequence_;
                jpg = latest_jpg_;
            }
            if (jpg.empty() || sequence == seen_sequence) continue;
            seen_sequence = sequence;

            const std::string header =
                "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                std::to_string(jpg.size()) + "\r\nX-Frame-Sequence: " +
                std::to_string(sequence) + "\r\n\r\n";

            if (!send_all(client->fd, header.data(), header.size()) ||
                !send_all(client->fd, reinterpret_cast<const char *>(jpg.data()), jpg.size()) ||
                !send_all(client->fd, "\r\n", 2)) {
                finish_client(client, "frame send failed");
                return;
            }
            ++client->frames_sent;
        }

        errno = 0;
        finish_client(client, running_ ? "client closed" : "server stopping");
    }

    void finish_client(const std::shared_ptr<Client> &client, const char *reason) {
        if (!client->active.exchange(false)) return;
        const int saved_errno = errno;
        shutdown(client->fd, SHUT_RDWR);
        close(client->fd);
        active_clients_.fetch_sub(1);
        std::cerr << "MJPEG client disconnected: " << client->peer
                  << " frames=" << client->frames_sent
                  << " reason=" << reason;
        if (saved_errno != 0) std::cerr << " error=" << std::strerror(saved_errno);
        std::cerr << std::endl;
    }

    void reap_inactive_clients() {
        std::vector<std::shared_ptr<Client>> inactive;
        {
            std::lock_guard<std::mutex> lock(clients_mutex_);
            auto it = clients_.begin();
            while (it != clients_.end()) {
                if (!(*it)->active.load()) {
                    inactive.push_back(*it);
                    it = clients_.erase(it);
                } else {
                    ++it;
                }
            }
        }
        for (const auto &client : inactive) {
            if (client->thread.joinable()) client->thread.join();
        }
    }

    void accept_loop() {
        while (running_) {
            sockaddr_in peer_addr{};
            socklen_t peer_len = sizeof(peer_addr);
            const int client_fd = accept(
                listen_fd_, reinterpret_cast<sockaddr *>(&peer_addr), &peer_len);
            if (client_fd < 0) {
                if (!running_) break;
                continue;
            }

            timeval tv{};
            tv.tv_sec = 1;
            setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

            timeval receive_timeout{};
            receive_timeout.tv_sec = 3;
            setsockopt(
                client_fd, SOL_SOCKET, SO_RCVTIMEO,
                &receive_timeout, sizeof(receive_timeout));

            int keepalive = 1;
            setsockopt(client_fd, SOL_SOCKET, SO_KEEPALIVE, &keepalive, sizeof(keepalive));
#ifdef TCP_KEEPIDLE
            int keepidle = 5;
            setsockopt(client_fd, IPPROTO_TCP, TCP_KEEPIDLE, &keepidle, sizeof(keepidle));
#endif
#ifdef TCP_KEEPINTVL
            int keepintvl = 2;
            setsockopt(client_fd, IPPROTO_TCP, TCP_KEEPINTVL, &keepintvl, sizeof(keepintvl));
#endif
#ifdef TCP_KEEPCNT
            int keepcnt = 3;
            setsockopt(client_fd, IPPROTO_TCP, TCP_KEEPCNT, &keepcnt, sizeof(keepcnt));
#endif
#ifdef TCP_USER_TIMEOUT
            unsigned int user_timeout_ms = 5000;
            setsockopt(
                client_fd, IPPROTO_TCP, TCP_USER_TIMEOUT,
                &user_timeout_ms, sizeof(user_timeout_ms));
#endif

            char address[INET_ADDRSTRLEN] = "unknown";
            inet_ntop(AF_INET, &peer_addr.sin_addr, address, sizeof(address));
            auto client = std::make_shared<Client>();
            client->fd = client_fd;
            client->peer = std::string(address) + ":" +
                           std::to_string(ntohs(peer_addr.sin_port));
            active_clients_.fetch_add(1);
            client->thread = std::thread(&MjpegServer::client_loop, this, client);
            {
                std::lock_guard<std::mutex> lock(clients_mutex_);
                clients_.push_back(client);
            }
            std::cout << "MJPEG client connected: " << client->peer << std::endl;
            reap_inactive_clients();
        }
        reap_inactive_clients();
    }

    int listen_fd_ = -1;
    std::atomic<bool> running_{false};
    std::atomic<size_t> active_clients_{0};
    std::thread accept_thread_;
    std::thread worker_thread_;

    std::vector<std::shared_ptr<Client>> clients_;
    std::mutex clients_mutex_;

    cv::Mat latest_frame_;
    bool has_new_frame_ = false;
    std::mutex frame_mutex_;
    std::condition_variable frame_cv_;

    std::vector<uchar> latest_jpg_;
    uint64_t encoded_sequence_ = 0;
    std::mutex encoded_mutex_;
    std::condition_variable encoded_cv_;
};
