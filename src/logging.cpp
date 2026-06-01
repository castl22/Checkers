#include "checkers/logging.hpp"

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fcntl.h>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace checkers {

namespace {

std::string make_timestamp() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm tm_value{};
#if defined(_WIN32)
    localtime_s(&tm_value, &now_time);
#else
    localtime_r(&now_time, &tm_value);
#endif

    std::ostringstream stream;
    stream << std::put_time(&tm_value, "%Y%m%d_%H%M%S");
    return stream.str();
}

std::string sanitize_token(std::string value) {
    for (char& ch : value) {
        const bool keep = (ch >= 'a' && ch <= 'z')
            || (ch >= 'A' && ch <= 'Z')
            || (ch >= '0' && ch <= '9')
            || ch == '_'
            || ch == '-'
            || ch == '.';
        if (!keep) {
            ch = '_';
        }
    }
    return value;
}

std::string env_or_default(const char* name, const std::string& fallback) {
    if (const char* value = std::getenv(name)) {
        if (*value != '\0') {
            return value;
        }
    }
    return fallback;
}

std::string read_run_id_file(const std::filesystem::path& path) {
    std::ifstream stream(path);
    std::string value;
    if (stream.is_open()) {
        std::getline(stream, value);
    }
    return value;
}

std::string resolve_shared_run_id(const std::filesystem::path& logs_root) {
    if (const char* run_id = std::getenv("CHECKERS_LOG_RUN_ID")) {
        if (*run_id != '\0') {
            return run_id;
        }
    }

    const std::string coordinator_key = sanitize_token(
        env_or_default("FLUX_JOB_ID", "") + "_"
        + env_or_default("MASTER_ADDR", "local") + "_"
        + env_or_default("MASTER_PORT", "0") + "_"
        + env_or_default("WORLD_SIZE", "1"));
    const std::filesystem::path run_id_file = logs_root / (".run_" + coordinator_key + ".id");

    if (std::filesystem::exists(run_id_file)) {
        const std::string existing = read_run_id_file(run_id_file);
        if (!existing.empty()) {
            return existing;
        }
    }

    const std::string candidate = make_timestamp();
    const int fd = ::open(run_id_file.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (fd >= 0) {
        const ssize_t ignored = ::write(fd, candidate.data(), candidate.size());
        (void)ignored;
        ::close(fd);
        return candidate;
    }

    const std::string existing = read_run_id_file(run_id_file);
    if (!existing.empty()) {
        return existing;
    }

    return candidate;
}

} // namespace

std::shared_ptr<RankLogger> RankLogger::create(int global_rank, int local_rank) {
    const std::filesystem::path logs_root = std::filesystem::current_path() / "logs";
    std::filesystem::create_directories(logs_root);
    const std::filesystem::path log_root = logs_root / resolve_shared_run_id(logs_root);
    const std::filesystem::path log_dir = log_root / ("rank_" + std::to_string(global_rank));
    const std::filesystem::path log_path = log_dir / "checkers.log";
    std::filesystem::create_directories(log_dir);
    return std::shared_ptr<RankLogger>(new RankLogger(global_rank, local_rank, log_dir, log_path));
}

RankLogger::RankLogger(int global_rank,
                       int local_rank,
                       std::filesystem::path log_dir,
                       std::filesystem::path log_path)
    : global_rank_(global_rank)
    , local_rank_(local_rank)
    , log_dir_(std::move(log_dir))
    , log_path_(std::move(log_path))
    , stream_(log_path_, std::ios::out | std::ios::trunc)
{
    if (!stream_.is_open()) {
        throw std::runtime_error("Failed to open rank log file: " + log_path_.string());
    }

    write_line("[checkers] log initialized");
    write_line("  global_rank=" + std::to_string(global_rank_) + " local_rank=" + std::to_string(local_rank_));
    write_line("  log_path=" + log_path_.string());
}

void RankLogger::log_message(const std::string& message) {
    write_line(message);
}

void RankLogger::log_tensor_discovered(const std::string& source,
                                       const TensorMetadata& meta,
                                       void* ptr,
                                       size_t byte_size) {
    std::ostringstream stream;
    stream << "[discovery] source=" << source
           << " name=" << meta.name
           << " dtype=" << format_dtype(meta.data_type)
           << " shape=" << format_shape(meta)
           << " bytes=" << byte_size
        << " logical_shape=" << format_logical_shape(meta)
        << " logical_bytes=" << (meta.logical_byte_size == 0 ? byte_size : meta.logical_byte_size)
           << " ptr=" << ptr;
    write_line(stream.str());
}

void RankLogger::log_tensor_skipped(const std::string& name, const std::string& reason) {
    write_line("[discovery] skipped name=" + name + " reason=" + reason);
}

void RankLogger::log_batch_metrics(size_t batch_index,
                                   size_t tensor_count,
                                   size_t buffer_bytes,
                                   double pass1_ms,
                                   double allocation_ms,
                                   double kernel_ms) {
    std::ostringstream stream;
    stream << "[batch] index=" << batch_index
           << " tensors=" << tensor_count
           << " buffer_bytes=" << buffer_bytes
           << " pass1_ms=" << pass1_ms
           << " allocation_ms=" << allocation_ms
           << " kernel_ms=" << kernel_ms;
    write_line(stream.str());
}

void RankLogger::log_summary(size_t discovered_tensors,
                             size_t skipped_tensors,
                             size_t tracked_tensors,
                             size_t slab_count,
                             size_t tensor_bytes,
                             size_t buffer_bytes,
                             double discovery_ms,
                             double pass1_ms,
                             double allocation_ms,
                             double kernel_ms) {
    std::ostringstream stream;
    stream << "[summary] discovered=" << discovered_tensors
           << " skipped=" << skipped_tensors
           << " tracked=" << tracked_tensors
           << " slabs=" << slab_count
           << " tensor_bytes=" << tensor_bytes
           << " buffer_bytes=" << buffer_bytes
           << " discovery_ms=" << discovery_ms
           << " pass1_ms=" << pass1_ms
           << " allocation_ms=" << allocation_ms
           << " kernel_ms=" << kernel_ms;
    write_line(stream.str());
}

std::string RankLogger::format_shape(const TensorMetadata& meta) {
    std::ostringstream stream;
    stream << "[";
    for (size_t i = 0; i < meta.shape.size(); ++i) {
        if (i > 0) {
            stream << ",";
        }
        stream << meta.shape[i];
    }
    stream << "]";
    return stream.str();
}

std::string RankLogger::format_logical_shape(const TensorMetadata& meta) {
    const auto& shape = meta.logical_shape.empty() ? meta.shape : meta.logical_shape;

    std::ostringstream stream;
    stream << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            stream << ",";
        }
        stream << shape[i];
    }
    stream << "]";
    return stream.str();
}

std::string RankLogger::format_dtype(DataType data_type) {
    switch (data_type) {
        case DataType::Float32:
            return "float32";
        case DataType::BFloat16:
            return "bfloat16";
        case DataType::Float16:
            return "float16";
        case DataType::Int8:
            return "int8";
        case DataType::Int32:
            return "int32";
        default:
            return "unknown";
    }
}

std::string RankLogger::format_contiguity(Contiguity contiguity) {
    switch (contiguity) {
        case Contiguity::Contiguous:
            return "contiguous";
        case Contiguity::NonContiguous:
            return "non_contiguous";
        case Contiguity::Scalar:
            return "scalar";
        default:
            return "unknown";
    }
}

void RankLogger::write_line(const std::string& line) {
    std::lock_guard lock(mutex_);
    stream_ << line << '\n';
    stream_.flush();
}

} // namespace checkers