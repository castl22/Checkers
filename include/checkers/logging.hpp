#pragma once

#include "TensorResource.hpp"

#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <array>

namespace checkers {

class RankLogger {
public:
    static std::shared_ptr<RankLogger> create(int global_rank, int local_rank);

    RankLogger(const RankLogger&) = delete;
    RankLogger& operator=(const RankLogger&) = delete;

    void log_message(const std::string& message);
    void log_tensor_discovered(const std::string& source,
                               const TensorMetadata& meta,
                               void* ptr,
                               size_t byte_size);
    void log_tensor_skipped(const std::string& name, const std::string& reason);
    void log_batch_metrics(size_t batch_index,
                           size_t tensor_count,
                           size_t buffer_bytes,
                           double pass1_ms,
                           double allocation_ms,
                           double kernel_ms);
    void log_summary(const std::array<TensorCategoryStats, tracked_tensor_category_count>& category_stats,
                     size_t skipped_tensors,
                     size_t slab_count,
                     size_t batch_count);

    int global_rank() const { return global_rank_; }
    int local_rank() const { return local_rank_; }
    const std::filesystem::path& log_path() const { return log_path_; }

private:
    RankLogger(int global_rank,
               int local_rank,
               std::filesystem::path log_dir,
               std::filesystem::path log_path);

    static std::string format_shape(const TensorMetadata& meta);
    static std::string format_logical_shape(const TensorMetadata& meta);
    static std::string format_dtype(DataType data_type);
    static std::string format_contiguity(Contiguity contiguity);

    void write_line(const std::string& line);

    int global_rank_;
    int local_rank_;
    std::filesystem::path log_dir_;
    std::filesystem::path log_path_;
    std::mutex mutex_;
    std::ofstream stream_;
};

} // namespace checkers