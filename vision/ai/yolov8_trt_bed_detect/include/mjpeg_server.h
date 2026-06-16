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
#include <cstring>

// Minimal MJPEG-over-HTTP streamer: connect a browser to
// http://<host>:<port>/ to view frames pushed via push_frame().
//
// push_frame() never blocks the caller on network I/O: it only swaps in the
// latest frame and wakes a dedicated worker thread that does the JPEG encode
// and broadcast. If no clients are connected, encoding is skipped entirely.
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
        worker_thread_ = std::thread(&MjpegServer::worker_loop, this);
        return true;
    }

    // Cheap: just stashes a reference frame and notifies the worker.
    // Never touches the network, never blocks on encode.
    void push_frame(const cv::Mat &frame) {
        {
            std::lock_guard<std::mutex> lock(clients_mutex_);
            if (clients_.empty()) return;  // nobody watching, skip the copy entirely
        }
        std::lock_guard<std::mutex> lock(frame_mutex_);
        frame.copyTo(latest_frame_);
        has_new_frame_ = true;
        frame_cv_.notify_one();
    }

    void stop() {
        running_ = false;
        frame_cv_.notify_all();
        if (listen_fd_ >= 0) {
            shutdown(listen_fd_, SHUT_RDWR);
            close(listen_fd_);
        }
        if (accept_thread_.joinable()) accept_thread_.join();
        if (worker_thread_.joinable()) worker_thread_.join();
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

    void worker_loop() {
        cv::Mat frame;
        std::vector<uchar> jpg;
        while (running_) {
            {
                std::unique_lock<std::mutex> lock(frame_mutex_);
                frame_cv_.wait(lock, [this] { return has_new_frame_ || !running_; });
                if (!running_) break;
                frame = latest_frame_;  // shares buffer; we only read it below
                has_new_frame_ = false;
            }

            {
                std::lock_guard<std::mutex> lock(clients_mutex_);
                if (clients_.empty()) continue;
            }

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
    }

    void accept_loop() {
        while (running_) {
            int client_fd = accept(listen_fd_, nullptr, nullptr);
            if (client_fd < 0) {
                if (!running_) break;
                continue;
            }

            // A slow/stalled client should time out instead of blocking the
            // worker thread's send() indefinitely.
            timeval tv{};
            tv.tv_sec = 1;
            tv.tv_usec = 0;
            setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

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
    std::thread worker_thread_;
    std::vector<int> clients_;
    std::mutex clients_mutex_;

    cv::Mat latest_frame_;
    bool has_new_frame_ = false;
    std::mutex frame_mutex_;
    std::condition_variable frame_cv_;
};
