#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <zstd.h>

namespace {

namespace fs = std::filesystem;

constexpr double k_megabyte_divisor = 1024.0 * 1024.0;
constexpr std::size_t k_default_max_codebook_values = 4096;

struct TensorEntry {
    std::string tensor_key;
    std::string state_type;
    std::string category;
    int layer {0};
    std::size_t numel {0};
    fs::path file_path;
};

struct GroupDefinition {
    std::string state_type;
    std::string category;
    int group_id {0};
    std::vector<int> layers;
    int representative_layer {0};
};

struct CompressionConfig {
    fs::path export_dir;
    std::size_t max_codebook_values {k_default_max_codebook_values};
    int zstd_level {3};
    std::optional<std::string> state_type_filter;
    std::optional<std::string> category_filter;
};

struct GroupCodebook {
    std::vector<std::uint32_t> values;
    std::unordered_map<std::uint32_t, std::uint32_t> ids;
    std::size_t code_width_bytes {1};
};

struct TensorCodebookResult {
    std::string state_type;
    std::string category;
    int group_id {-1};
    int layer {0};
    bool is_representative {false};
    std::size_t original_bytes {0};
    std::size_t payload_bytes {0};
    std::size_t codebook_bytes_charged {0};
    std::size_t exact_reconstruction_bytes {0};
    std::size_t hits {0};
    std::size_t misses {0};
    std::size_t code_width_bytes {0};
    double hit_fraction {0.0};
    double encode_time_ms {0.0};
    double exact_reconstruction_ratio {1.0};
};

struct GroupCodebookSummary {
    std::string state_type;
    std::string category;
    int group_id {0};
    std::vector<int> layers;
    int representative_layer {0};
    std::size_t codebook_value_count {0};
    std::size_t code_width_bytes {0};
    std::size_t codebook_bytes {0};
    std::size_t original_bytes {0};
    std::size_t payload_bytes {0};
    std::size_t exact_reconstruction_bytes {0};
    std::size_t hits {0};
    std::size_t misses {0};
    double average_hit_fraction {0.0};
    double exact_reconstruction_ratio {1.0};
    double total_time_ms {0.0};
};

struct CheckpointCodebookSummary {
    std::size_t original_bytes {0};
    std::size_t exact_reconstruction_bytes {0};
    std::size_t grouped_tensor_count {0};
    std::size_t uncompressed_tensor_count {0};
    std::size_t total_codebook_bytes {0};
    double exact_reconstruction_ratio {1.0};
};

double safe_divide(double numerator, double denominator)
{
    if (std::abs(denominator) <= std::numeric_limits<double>::epsilon()) {
        return 0.0;
    }
    return numerator / denominator;
}

double bytes_to_mb(std::size_t bytes)
{
    return static_cast<double>(bytes) / k_megabyte_divisor;
}

std::vector<std::string> parse_csv_row(const std::string& line)
{
    std::vector<std::string> columns;
    std::string current;
    bool inside_quotes = false;

    for (char character : line) {
        if (character == '"') {
            inside_quotes = !inside_quotes;
            continue;
        }
        if (character == ',' && !inside_quotes) {
            columns.push_back(current);
            current.clear();
            continue;
        }
        current.push_back(character);
    }

    columns.push_back(current);

    for (auto& column : columns) {
        const auto first = column.find_first_not_of(" \t\r");
        const auto last = column.find_last_not_of(" \t\r");
        if (first == std::string::npos) {
            column.clear();
            continue;
        }
        column = column.substr(first, last - first + 1);
    }

    return columns;
}

std::vector<int> parse_group_layers(const std::string& text)
{
    std::vector<int> layers;
    std::stringstream stream(text);
    int value = 0;
    while (stream >> value) {
        layers.push_back(value);
    }
    return layers;
}

std::vector<float> load_tensor_file(const fs::path& file_path, std::size_t numel)
{
    std::ifstream stream(file_path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Failed to open tensor payload: " + file_path.string());
    }

    std::vector<float> tensor(numel, 0.0f);
    stream.read(reinterpret_cast<char*>(tensor.data()), static_cast<std::streamsize>(numel * sizeof(float)));
    if (stream.gcount() != static_cast<std::streamsize>(numel * sizeof(float))) {
        throw std::runtime_error("Tensor payload size mismatch: " + file_path.string());
    }
    return tensor;
}

std::vector<std::uint32_t> tensor_to_raw_bits(const std::vector<float>& tensor)
{
    std::vector<std::uint32_t> bits(tensor.size(), 0u);
    for (std::size_t index = 0; index < tensor.size(); ++index) {
        std::memcpy(&bits[index], &tensor[index], sizeof(float));
    }
    return bits;
}

std::vector<std::uint8_t> compress_zstd(const std::vector<std::uint8_t>& data, int level)
{
    if (data.empty()) {
        return {};
    }

    std::vector<std::uint8_t> compressed(ZSTD_compressBound(data.size()));
    const auto compressed_size = ZSTD_compress(compressed.data(), compressed.size(), data.data(), data.size(), level);
    if (ZSTD_isError(compressed_size)) {
        throw std::runtime_error(std::string("ZSTD_compress failed: ") + ZSTD_getErrorName(compressed_size));
    }
    compressed.resize(compressed_size);
    return compressed;
}

void append_value(std::vector<std::uint8_t>& buffer, std::uint32_t value, std::size_t width_bytes)
{
    for (std::size_t byte = 0; byte < width_bytes; ++byte) {
        buffer.push_back(static_cast<std::uint8_t>((value >> (8 * byte)) & 0xffu));
    }
}

std::size_t choose_code_width(std::size_t codebook_size)
{
    if (codebook_size <= (1ull << 8)) {
        return 1;
    }
    if (codebook_size <= (1ull << 16)) {
        return 2;
    }
    return 4;
}

std::map<std::tuple<std::string, std::string, int>, TensorEntry> load_manifest(const fs::path& manifest_path)
{
    std::ifstream stream(manifest_path);
    if (!stream) {
        throw std::runtime_error("Failed to open manifest: " + manifest_path.string());
    }

    std::string line;
    std::getline(stream, line);

    std::map<std::tuple<std::string, std::string, int>, TensorEntry> entries;
    while (std::getline(stream, line)) {
        if (line.empty()) {
            continue;
        }

        const auto columns = parse_csv_row(line);
        if (columns.size() != 7) {
            throw std::runtime_error("Malformed manifest row: " + line);
        }

        TensorEntry entry;
        entry.tensor_key = columns[0];
        entry.layer = std::stoi(columns[1]);
        entry.category = columns[2];
        entry.state_type = columns[3];
        entry.numel = static_cast<std::size_t>(std::stoull(columns[4]));
        entry.file_path = manifest_path.parent_path() / columns[6];
        entries[{entry.state_type, entry.category, entry.layer}] = entry;
    }
    return entries;
}

std::vector<GroupDefinition> load_groups(const fs::path& grouping_summary_path, const CompressionConfig& config)
{
    std::ifstream stream(grouping_summary_path);
    if (!stream) {
        throw std::runtime_error("Failed to open grouping summary: " + grouping_summary_path.string());
    }

    std::string line;
    std::getline(stream, line);

    std::vector<GroupDefinition> groups;
    while (std::getline(stream, line)) {
        if (line.empty()) {
            continue;
        }

        const auto columns = parse_csv_row(line);
        if (columns.size() < 10) {
            throw std::runtime_error("Malformed grouping summary row: " + line);
        }

        GroupDefinition group;
        group.state_type = columns[0];
        group.category = columns[1];
        group.group_id = std::stoi(columns[2]);
        group.layers = parse_group_layers(columns[4]);
        group.representative_layer = std::stoi(columns[5]);

        if (config.state_type_filter && group.state_type != *config.state_type_filter) {
            continue;
        }
        if (config.category_filter && group.category != *config.category_filter) {
            continue;
        }
        groups.push_back(group);
    }

    return groups;
}

GroupCodebook build_group_codebook(const std::vector<std::uint32_t>& representative_bits, std::size_t max_values)
{
    std::unordered_map<std::uint32_t, std::size_t> counts;
    counts.reserve(representative_bits.size());
    for (auto value : representative_bits) {
        ++counts[value];
    }

    std::vector<std::pair<std::uint32_t, std::size_t>> ranked(counts.begin(), counts.end());
    std::sort(
        ranked.begin(),
        ranked.end(),
        [](const auto& lhs, const auto& rhs) {
            if (lhs.second != rhs.second) {
                return lhs.second > rhs.second;
            }
            return lhs.first < rhs.first;
        });

    GroupCodebook codebook;
    for (const auto& [value, count] : ranked) {
        if (count < 2) {
            break;
        }
        if (codebook.values.size() >= max_values) {
            break;
        }
        codebook.ids.emplace(value, static_cast<std::uint32_t>(codebook.values.size()));
        codebook.values.push_back(value);
    }
    codebook.code_width_bytes = choose_code_width(codebook.values.size());
    return codebook;
}

TensorCodebookResult encode_tensor_with_codebook(const TensorEntry& tensor,
                                                 const GroupCodebook& codebook,
                                                 bool is_representative,
                                                 int group_id,
                                                 std::size_t codebook_bytes_charged,
                                                 int zstd_level)
{
    const auto start_time = std::chrono::high_resolution_clock::now();
    const auto bits = tensor_to_raw_bits(load_tensor_file(tensor.file_path, tensor.numel));

    std::vector<std::uint8_t> bitmap((bits.size() + 7) / 8, 0u);
    std::vector<std::uint8_t> hit_codes;
    std::vector<std::uint8_t> misses;
    hit_codes.reserve(bits.size() * codebook.code_width_bytes);
    misses.reserve(bits.size() * sizeof(std::uint32_t));

    std::size_t hits = 0;
    for (std::size_t index = 0; index < bits.size(); ++index) {
        const auto found = codebook.ids.find(bits[index]);
        if (found != codebook.ids.end()) {
            bitmap[index / 8] |= static_cast<std::uint8_t>(1u << (index % 8));
            append_value(hit_codes, found->second, codebook.code_width_bytes);
            ++hits;
        } else {
            append_value(misses, bits[index], sizeof(std::uint32_t));
        }
    }

    std::vector<std::uint8_t> payload;
    payload.reserve(bitmap.size() + hit_codes.size() + misses.size() + 3 * sizeof(std::uint32_t));
    append_value(payload, static_cast<std::uint32_t>(bitmap.size()), sizeof(std::uint32_t));
    append_value(payload, static_cast<std::uint32_t>(hit_codes.size()), sizeof(std::uint32_t));
    append_value(payload, static_cast<std::uint32_t>(misses.size()), sizeof(std::uint32_t));
    payload.insert(payload.end(), bitmap.begin(), bitmap.end());
    payload.insert(payload.end(), hit_codes.begin(), hit_codes.end());
    payload.insert(payload.end(), misses.begin(), misses.end());

    const auto compressed_payload = compress_zstd(payload, zstd_level);

    TensorCodebookResult result {};
    result.state_type = tensor.state_type;
    result.category = tensor.category;
    result.group_id = group_id;
    result.layer = tensor.layer;
    result.is_representative = is_representative;
    result.original_bytes = tensor.numel * sizeof(float);
    result.payload_bytes = compressed_payload.size();
    result.codebook_bytes_charged = codebook_bytes_charged;
    result.exact_reconstruction_bytes = result.payload_bytes + result.codebook_bytes_charged;
    result.hits = hits;
    result.misses = bits.size() - hits;
    result.code_width_bytes = codebook.code_width_bytes;
    result.hit_fraction = safe_divide(static_cast<double>(result.hits), static_cast<double>(bits.size()));
    result.encode_time_ms =
        std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count();
    result.exact_reconstruction_ratio = safe_divide(
        static_cast<double>(result.original_bytes),
        static_cast<double>(result.exact_reconstruction_bytes));
    return result;
}

void write_tensor_report(const fs::path& output_path, const std::vector<TensorCodebookResult>& rows)
{
    std::ofstream stream(output_path);
    stream << "state_type,category,group_id,layer,is_representative,original_size_mb,payload_size_mb,codebook_size_mb_charged,compressed_size_mb,hits,misses,hit_fraction,code_width_bytes,compression_ratio,time_ms\n";
    for (const auto& row : rows) {
        stream << row.state_type << ','
               << row.category << ','
               << row.group_id << ','
               << row.layer << ','
               << (row.is_representative ? "true" : "false") << ','
               << bytes_to_mb(row.original_bytes) << ','
               << bytes_to_mb(row.payload_bytes) << ','
               << bytes_to_mb(row.codebook_bytes_charged) << ','
               << bytes_to_mb(row.exact_reconstruction_bytes) << ','
               << row.hits << ','
               << row.misses << ','
               << row.hit_fraction << ','
               << row.code_width_bytes << ','
               << row.exact_reconstruction_ratio << ','
               << row.encode_time_ms << '\n';
    }
}

void write_group_report(const fs::path& output_path, const std::vector<GroupCodebookSummary>& rows)
{
    std::ofstream stream(output_path);
    stream << "state_type,category,group_id,layers,representative_layer,codebook_value_count,code_width_bytes,codebook_size_mb,original_group_size_mb,payload_group_size_mb,compressed_group_size_mb,total_hits,total_misses,average_hit_fraction,compression_ratio,total_time_ms\n";
    for (const auto& row : rows) {
        std::stringstream layers_stream;
        for (std::size_t index = 0; index < row.layers.size(); ++index) {
            if (index > 0) {
                layers_stream << ' ';
            }
            layers_stream << row.layers[index];
        }

        stream << row.state_type << ','
               << row.category << ','
               << row.group_id << ','
               << '"' << layers_stream.str() << '"' << ','
               << row.representative_layer << ','
               << row.codebook_value_count << ','
               << row.code_width_bytes << ','
               << bytes_to_mb(row.codebook_bytes) << ','
               << bytes_to_mb(row.original_bytes) << ','
               << bytes_to_mb(row.payload_bytes) << ','
               << bytes_to_mb(row.exact_reconstruction_bytes) << ','
               << row.hits << ','
               << row.misses << ','
               << row.average_hit_fraction << ','
               << row.exact_reconstruction_ratio << ','
               << row.total_time_ms << '\n';
    }
}

void write_checkpoint_summary(const fs::path& output_path,
                              const CheckpointCodebookSummary& summary,
                              const std::map<std::string, std::pair<std::size_t, std::size_t>>& state_type_totals,
                              const CompressionConfig& config)
{
    std::ofstream stream(output_path);
    stream << std::fixed << std::setprecision(6);
    stream << "transform_mode: raw_exact_value_codebook\n";
    stream << "payload_codec: zstd\n";
    stream << "max_codebook_values: " << config.max_codebook_values << '\n';
    stream << "checkpoint_original_size_mb: " << bytes_to_mb(summary.original_bytes) << '\n';
    stream << "checkpoint_compressed_size_mb: " << bytes_to_mb(summary.exact_reconstruction_bytes) << '\n';
    stream << "checkpoint_compression_ratio: " << summary.exact_reconstruction_ratio << '\n';
    stream << "grouped_tensor_count: " << summary.grouped_tensor_count << '\n';
    stream << "uncompressed_tensor_count: " << summary.uncompressed_tensor_count << '\n';
    stream << "codebook_size_mb_total: " << bytes_to_mb(summary.total_codebook_bytes) << "\n\n";

    for (const auto& [state_type, totals] : state_type_totals) {
        stream << state_type << ": original_size_mb=" << bytes_to_mb(totals.first)
               << ", compressed_size_mb=" << bytes_to_mb(totals.second)
               << ", compression_ratio=" << safe_divide(static_cast<double>(totals.first), static_cast<double>(totals.second))
               << '\n';
    }
}

CompressionConfig parse_args(int argc, char** argv)
{
    if (argc < 2) {
        throw std::runtime_error(
            "Usage: tensor_group_codebook <export_dir> [--max-codebook-values N] [--zstd-level N] [--state-type NAME] [--category NAME]");
    }

    CompressionConfig config {};
    config.export_dir = fs::path(argv[1]);
    for (int index = 2; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--max-codebook-values" && index + 1 < argc) {
            config.max_codebook_values = static_cast<std::size_t>(std::stoull(argv[++index]));
        } else if (argument == "--zstd-level" && index + 1 < argc) {
            config.zstd_level = std::stoi(argv[++index]);
        } else if (argument == "--state-type" && index + 1 < argc) {
            config.state_type_filter = std::string(argv[++index]);
        } else if (argument == "--category" && index + 1 < argc) {
            config.category_filter = std::string(argv[++index]);
        } else {
            throw std::runtime_error("Unknown or incomplete argument: " + argument);
        }
    }

    return config;
}

void run_group_codebook(const CompressionConfig& config)
{
    const auto manifest_entries = load_manifest(config.export_dir / "manifest.csv");
    const auto groups = load_groups(config.export_dir / "analysis" / "plots" / "grouping_summary.csv", config);
    const fs::path output_dir = config.export_dir / "codebook";
    fs::create_directories(output_dir);

    std::set<std::tuple<std::string, std::string, int>> grouped_keys;
    std::vector<TensorCodebookResult> tensor_rows;
    std::vector<GroupCodebookSummary> group_rows;
    CheckpointCodebookSummary checkpoint_summary {};
    std::map<std::string, std::pair<std::size_t, std::size_t>> state_type_totals;

    for (const auto& group : groups) {
        const auto representative_it = manifest_entries.find({group.state_type, group.category, group.representative_layer});
        if (representative_it == manifest_entries.end()) {
            throw std::runtime_error("Missing representative tensor in manifest for group");
        }

        const auto representative_bits = tensor_to_raw_bits(
            load_tensor_file(representative_it->second.file_path, representative_it->second.numel));
        const auto codebook = build_group_codebook(representative_bits, config.max_codebook_values);

        GroupCodebookSummary group_summary {};
        group_summary.state_type = group.state_type;
        group_summary.category = group.category;
        group_summary.group_id = group.group_id;
        group_summary.layers = group.layers;
        group_summary.representative_layer = group.representative_layer;
        group_summary.codebook_value_count = codebook.values.size();
        group_summary.code_width_bytes = codebook.code_width_bytes;
        group_summary.codebook_bytes = codebook.values.size() * sizeof(std::uint32_t);

        bool charged_codebook = false;
        double hit_fraction_sum = 0.0;
        for (int layer : group.layers) {
            const auto tensor_it = manifest_entries.find({group.state_type, group.category, layer});
            if (tensor_it == manifest_entries.end()) {
                throw std::runtime_error("Missing grouped tensor in manifest");
            }

            grouped_keys.insert({group.state_type, group.category, layer});
            const auto codebook_bytes = charged_codebook ? 0u : group_summary.codebook_bytes;
            auto tensor_result = encode_tensor_with_codebook(
                tensor_it->second,
                codebook,
                layer == group.representative_layer,
                group.group_id,
                codebook_bytes,
                config.zstd_level);
            charged_codebook = true;

            group_summary.original_bytes += tensor_result.original_bytes;
            group_summary.payload_bytes += tensor_result.payload_bytes;
            group_summary.exact_reconstruction_bytes += tensor_result.exact_reconstruction_bytes;
            group_summary.hits += tensor_result.hits;
            group_summary.misses += tensor_result.misses;
            hit_fraction_sum += tensor_result.hit_fraction;
            group_summary.total_time_ms += tensor_result.encode_time_ms;
            tensor_rows.push_back(tensor_result);
        }

        group_summary.average_hit_fraction = safe_divide(hit_fraction_sum, static_cast<double>(group.layers.size()));
        group_summary.exact_reconstruction_ratio = safe_divide(
            static_cast<double>(group_summary.original_bytes),
            static_cast<double>(group_summary.exact_reconstruction_bytes));
        group_rows.push_back(group_summary);

        checkpoint_summary.original_bytes += group_summary.original_bytes;
        checkpoint_summary.exact_reconstruction_bytes += group_summary.exact_reconstruction_bytes;
        checkpoint_summary.grouped_tensor_count += group.layers.size();
        checkpoint_summary.total_codebook_bytes += group_summary.codebook_bytes;
        state_type_totals[group.state_type].first += group_summary.original_bytes;
        state_type_totals[group.state_type].second += group_summary.exact_reconstruction_bytes;
    }

    for (const auto& [key, entry] : manifest_entries) {
        if (config.state_type_filter && entry.state_type != *config.state_type_filter) {
            continue;
        }
        if (config.category_filter && entry.category != *config.category_filter) {
            continue;
        }
        if (grouped_keys.contains(key)) {
            continue;
        }

        const auto original_bytes = entry.numel * sizeof(float);
        checkpoint_summary.original_bytes += original_bytes;
        checkpoint_summary.exact_reconstruction_bytes += original_bytes;
        ++checkpoint_summary.uncompressed_tensor_count;
        state_type_totals[entry.state_type].first += original_bytes;
        state_type_totals[entry.state_type].second += original_bytes;
    }

    checkpoint_summary.exact_reconstruction_ratio = safe_divide(
        static_cast<double>(checkpoint_summary.original_bytes),
        static_cast<double>(checkpoint_summary.exact_reconstruction_bytes));

    write_tensor_report(output_dir / "tensor_codebook.csv", tensor_rows);
    write_group_report(output_dir / "group_codebook.csv", group_rows);
    write_checkpoint_summary(output_dir / "checkpoint_codebook_summary.txt", checkpoint_summary, state_type_totals, config);

    std::cout << "Wrote tensor codebook report to " << (output_dir / "tensor_codebook.csv") << std::endl;
    std::cout << "Wrote group codebook report to " << (output_dir / "group_codebook.csv") << std::endl;
    std::cout << "Wrote checkpoint summary to " << (output_dir / "checkpoint_codebook_summary.txt") << std::endl;
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const auto config = parse_args(argc, argv);
        run_group_codebook(config);
    } catch (const std::exception& error) {
        std::cerr << "tensor_group_codebook failed: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
