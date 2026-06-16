#pragma once

#include <opencv2/opencv.hpp>
#include <arpa/inet.h>
#include <unistd.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <thread>
#include <mutex>
#include <vector>
#include <atomic>
#include <cstring>

// Minimal MJPEG-over-HTTP streamer: connect a browser to
// http://<host>:<port>/ to view frames pushed via push_frame().
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

        if (bind(listen_fd_, (sockaddr *)&addr, sizeof(addr)) < 0) return false;
        if (listen(listen_fd_, 5) < 0) return false;

        running_ = true;
        accept_thread_ = std::thread(&MjpegServer::accept_loop, this);
        return true;
    }

    void push_frame(const cv::Mat &frame) {
        std::vector<uchar> jpg;
        cv::imencode(".jpg", frame, jpg, {cv::IMWRITE_JPEG_QUALITY, 80});

        std::string header =
            "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
            std::to_string(jpg.size()) + "\r\n\r\n";

        std::lock_guard<std::mutex> lock(clients_mutex_);
        for (auto it = clients_.begin(); it != clients_.end();) {
            int fd = *it;
            bool ok = send_all(fd, header.data(), header.size()) &&
                      send_all(fd, reinterpret_cast<const char *>(jpg.data()), jpg.size()) &&
                      send_all(fd, "\r\n", 2);
            if (!ok) {
                close(fd);
                it = clients_.erase(it);
            } else {
                ++it;
            }
        }
    }

    void stop() {
        running_ = false;
        if (listen_fd_ >= 0) {
            shutdown(listen_fd_, SHUT_RDWR);
            close(listen_fd_);
        }
        if (accept_thread_.joinable()) accept_thread_.join();
        std::lock_guard<std::mutex> lock(clients_mutex_);
        for (int fd : clients_) close(fd);
        clients_.clear();
    }

    ~MjpegServer() { stop(); }

private:
    static bool send_all(int fd, const char *data, size_t len) {
        size_t sent = 0;
        while (sent < len) {
            ssize_t n = send(fd, data + sent, len - sent, MSG_NOSIGNAL);
            if (n <= 0) return false;
            sent += n;
        }
        return true;
    }

    void accept_loop() {
        while (running_) {
            int client_fd = accept(listen_fd_, nullptr, nullptr);
            if (client_fd < 0) {
                if (!running_) break;
                continue;
            }

            static const char *response =
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
                "Cache-Control: no-cache\r\n"
                "Connection: close\r\n\r\n";
            send_all(client_fd, response, strlen(response));

            std::lock_guard<std::mutex> lock(clients_mutex_);
            clients_.push_back(client_fd);
        }
    }

    int listen_fd_ = -1;
    std::atomic<bool> running_{false};
    std::thread accept_thread_;
    std::vector<int> clients_;
    std::mutex clients_mutex_;
};
