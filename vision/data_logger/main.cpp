#include <iostream>
#include <string>
#include <csignal>
#include <unistd.h>
#include <sys/wait.h>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

static pid_t child_pid = -1;

static void stop_child() {
    if (child_pid > 0) {
        kill(child_pid, SIGINT);
        waitpid(child_pid, nullptr, 0);
        child_pid = -1;
        std::cout << "Logger stopped.\n";
    } else {
        std::cout << "No logger is running.\n";
    }
}

static void on_signal(int) {
    stop_child();
    std::exit(0);
}

static pid_t launch(const std::string& binary, const std::vector<std::string>& args) {
    if (!fs::exists(binary)) {
        std::cerr << "Binary not found: " << binary
                  << "\nRun: cmake --build build\n";
        return -1;
    }

    pid_t pid = fork();
    if (pid == 0) {
        // child
        std::vector<const char*> argv;
        argv.push_back(binary.c_str());
        for (auto& a : args) argv.push_back(a.c_str());
        argv.push_back(nullptr);
        execv(binary.c_str(), const_cast<char* const*>(argv.data()));
        std::perror("execv");
        std::exit(1);
    }
    return pid;
}

static void print_menu() {
    std::cout << "\n=== Data Logger Control ===\n"
              << "  1 - Start frame logger\n"
              << "  2 - Start video logger\n"
              << "  3 - Stop logger\n"
              << "  q - Quit\n"
              << "Choice: ";
}

int main(int argc, char** argv) {
    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    // Resolve binary paths relative to this executable
    std::string base = fs::canonical("/proc/self/exe").parent_path().string();
    std::string frame_bin = base + "/frame_logger";
    std::string video_bin = base + "/video_logger";

    // Default logger args — edit or extend as needed
    std::vector<std::string> frame_args = {"--device", "0", "--fps", "5", "--show"};
    std::vector<std::string> video_args = {"--device", "0", "--fps", "30", "--show"};

    std::string choice;
    while (true) {
        // Reap finished child so status is current
        if (child_pid > 0 && waitpid(child_pid, nullptr, WNOHANG) == child_pid) {
            child_pid = -1;
            std::cout << "\n[Logger exited]\n";
        }

        print_menu();
        if (!std::getline(std::cin, choice)) break;

        if (choice == "1") {
            if (child_pid > 0) { std::cout << "Stop the current logger first (3).\n"; continue; }
            child_pid = launch(frame_bin, frame_args);
            if (child_pid > 0) std::cout << "Frame logger started (pid " << child_pid << ").\n";

        } else if (choice == "2") {
            if (child_pid > 0) { std::cout << "Stop the current logger first (3).\n"; continue; }
            child_pid = launch(video_bin, video_args);
            if (child_pid > 0) std::cout << "Video logger started (pid " << child_pid << ").\n";

        } else if (choice == "3") {
            stop_child();

        } else if (choice == "q" || choice == "Q") {
            stop_child();
            break;

        } else {
            std::cout << "Unknown option.\n";
        }
    }

    return 0;
}
