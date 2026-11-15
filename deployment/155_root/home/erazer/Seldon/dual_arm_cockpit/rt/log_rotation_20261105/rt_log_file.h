#pragma once
// Used only by the existing background data writer, never the control thread.
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

namespace seldon {
inline const char* rt_log_directory() {
    return "/mnt/data/erazer/book_grasp/rt_logs/telemetry";
}

inline void require_log_volume() {
    struct stat system{}, data{};
    if (::stat("/", &system) || ::stat("/mnt/data", &data) ||
        system.st_dev == data.st_dev)
        throw std::runtime_error("RT logging requires the mounted /mnt/data volume; no root-disk fallback");
}

class RtLogFile {
    std::filesystem::path directory_;
    FILE* file_ = nullptr;
    int lock_ = -1;
    const long limit_;
    const int backups_;

    std::filesystem::path path(int index = 0) const {
        return directory_ / (index ? "rt_data.log." + std::to_string(index) : "rt_data.log");
    }
    static bool regular_or_absent(const std::filesystem::path& p) {
        auto status = std::filesystem::symlink_status(p);
        if (status.type() == std::filesystem::file_type::not_found) return false;
        if (status.type() != std::filesystem::file_type::regular)
            throw std::runtime_error("Refusing non-regular RT log path: " + p.string());
        return true;
    }
    void rotate() {
        // Check the complete, fixed namespace before deleting anything.
        for (int i = 0; i <= backups_; ++i) regular_or_absent(path(i));
        if (regular_or_absent(path(backups_))) std::filesystem::remove(path(backups_));
        for (int i = backups_ - 1; i >= 0; --i)
            if (regular_or_absent(path(i))) std::filesystem::rename(path(i), path(i + 1));
    }
    void create() {
        int fd = ::open(path().c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
        if (fd < 0) throw std::runtime_error("Cannot create RT log: " + std::string(std::strerror(errno)));
        file_ = ::fdopen(fd, "wb");
        if (!file_) { ::close(fd); throw std::runtime_error("Cannot open RT log stream"); }
    }
    void finish_file() {
        FILE* old = file_; file_ = nullptr;
        if (old && std::fclose(old)) throw std::runtime_error("RT log flush/close failed");
    }
public:
    // 256 MiB per file, seven archives plus current: approximately 2 GiB total.
    explicit RtLogFile(long bytes = 256L * 1024 * 1024, int backups = 7)
        : limit_(bytes), backups_(backups) {
        if (bytes <= 0 || backups < 1) throw std::invalid_argument("Invalid RT log retention");
    }
    RtLogFile(const RtLogFile&) = delete;
    RtLogFile& operator=(const RtLogFile&) = delete;
    ~RtLogFile() { close(); }
    FILE* open(const std::filesystem::path& directory) {
        if (lock_ != -1) throw std::logic_error("RT log already open");
        directory_ = directory;
        std::filesystem::create_directories(directory_);
        lock_ = ::open((directory_ / ".rt_data.lock").c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
        if (lock_ < 0) throw std::runtime_error("Cannot open RT log lock");
        if (::flock(lock_, LOCK_EX | LOCK_NB)) {
            ::close(lock_); lock_ = -1;
            throw std::runtime_error("Another RT process owns this log directory");
        }
        try {
            rotate(); // Apply the same retention across restarts, not per session.
            create();
            return file_;
        } catch (...) { close(); throw; }
    }
    FILE* after_record() noexcept {
        // Rotate between complete rows; the bound can exceed by at most one row.
        try {
            if (!file_ || std::ferror(file_)) throw std::runtime_error("RT log write failed");
            long position = std::ftell(file_);
            if (position < 0) throw std::runtime_error("RT log position failed");
            if (position >= limit_) {
                finish_file();
                rotate();
                create();
            }
            return file_;
        } catch (const std::exception& error) {
            // Keep draining the existing queue, with no root-disk fallback or
            // exception escaping the writer thread into hardware control.
            std::fprintf(stderr, "[RT_LOG_DISABLED] %s; fix storage before restarting RT\n", error.what());
            if (file_) std::fclose(file_);
            file_ = nullptr;
            return nullptr;
        }
    }
    void close() noexcept {
        if (file_) std::fclose(file_);
        file_ = nullptr;
        if (lock_ >= 0) ::close(lock_);
        lock_ = -1;
    }
};
} // namespace seldon
