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
constexpr double k_default_bloom_bits_per_value = 10.0;
constexpr std::size_t k_default_bloom_hashes = 7;

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
    double bloom_bits_per_value {k_default_bloom_bits_per_value};
    std::size_t bloom_hashes {k_default_bloom_hashes};
    int zstd_level {3};
    std::optional<std::string> state_type_filter;
    std::optional<std::string> category_filter;
};

struct BloomFilter {
    std::vector<std::uint64_t> words;
    std::size_t bit_count {0};
    std::size_t hash_count {0};

    void set_bit(std::size_t bit)
    {
        words[bit / 64] |= (1ull << (bit % 64));
    }

    bool test_bit(std::size_t bit) const
    {
        return (words[bit / 64] & (1ull << (bit % 64))) != 0;
    }
};

struct ReferenceDictionary {
    std::vector<std::uint32_t> values;
    std::unordered_map<std::uint32_t, std::uint32_t> ids;
    std::size_t index_width_bytes {4};
    BloomFilter bloom;
};

struct TensorBloomResult {
    std::string state_type;
    std::string category;
    int group_id {-1};
    int layer {0};
    bool is_representative {false};
    std::size_t original_bytes {0};
    std::size_t payload_bytes {0};
    std::size_t dictionary_bytes_charged {0};
    std::size_t bloom_bytes_charged {0};
    std::size_t exact_reconstruction_bytes {0};
    std::size_t bloom_positive_count {0};
    std::size_t exact_match_count {0};
    std::size_t miss_count {0};
    double bloom_positive_fraction {0.0};
    double exact_match_fraction {0.0};
    double encode_time_ms {0.0};
    double exact_reconstruction_ratio {1.0};
};

struct GroupBloomSummary {
    std::string state_type;
    std::string category;
    int group_id {0};
    std::vector<int> layers;
    int representative_layer {0};
    std::size_t dictionary_value_count {0};
    std::size_t dictionary_bytes {0};
    std::size_t bloom_bytes {0};
    std::size_t index_width_bytes {0};
    std::size_t original_bytes {0};
    std::size_t payload_bytes {0};
    std::size_t exact_reconstruction_bytes {0};
    std::size_t bloom_positive_count {0};
    std::size_t exact_match_count {0};
    std::size_t miss_count {0};
    double average_bloom_positive_fraction {0.0};
    double average_exact_match_fraction {0.0};
    double exact_reconstruction_ratio {1.0};
    double total_time_ms {0.0};
};

struct CheckpointBloomSummary {
    std::size_t original_bytes {0};
    std::size_t exact_reconstruction_bytes {0};
    std::size_t grouped_tensor_count {0};
    std::size_t uncompressed_tensor_count {0};
    std::size_t total_dictionary_bytes {0};
    std::size_t total_bloom_bytes {0};
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

std::uint64_t mix64(std::uint64_t value)
{
    value ^= value >> 33;
    value *= 0xff51afd7ed558ccdULL;
    value ^= value >> 33;
    value *= 0xc4ceb9fe1a85ec53ULL;
    value ^= value >> 33;
    return value;
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

std::size_t choose_index_width(std::size_t dictionary_size)
{
    if (dictionary_size <= (1ull << 8)) {
        return 1;
    }
    if (dictionary_size <= (1ull << 16)) {
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

ReferenceDictionary build_reference_dictionary(const std::vector<std::uint32_t>& representative_bits,
                                               double bloom_bits_per_value,
                                               std::size_t bloom_hashes)
{
    ReferenceDictionary dictionary;
    dictionary.ids.reserve(representative_bits.size());
    for (auto value : representative_bits) {
        if (!dictionary.ids.contains(value)) {
            dictionary.ids.emplace(value, static_cast<std::uint32_t>(dictionary.values.size()));
            dictionary.values.push_back(value);
        }
    }

    dictionary.index_width_bytes = choose_index_width(dictionary.values.size());
    dictionary.bloom.hash_count = bloom_hashes;
    dictionary.bloom.bit_count = static_cast<std::size_t>(std::ceil(dictionary.values.size() * bloom_bits_per_value));
    dictionary.bloom.bit_count = std::max<std::size_t>(dictionary.bloom.bit_count, 64);
    dictionary.bloom.words.assign((dictionary.bloom.bit_count + 63) / 64, 0ull);

    for (auto value : dictionary.values) {
        const auto base_hash = mix64(value);
        const auto step_hash = mix64(value ^ 0x9e3779b97f4a7c15ULL) | 1ull;
        for (std::size_t hash_index = 0; hash_index < dictionary.bloom.hash_count; ++hash_index) {
            dictionary.bloom.set_bit((base_hash + hash_index * step_hash) % dictionary.bloom.bit_count);
        }
    }
    return dictionary;
}

bool bloom_maybe_contains(const BloomFilter& bloom, std::uint32_t value)
{
    const auto base_hash = mix64(value);
    const auto step_hash = mix64(value ^ 0x9e3779b97f4a7c15ULL) | 1ull;
    for (std::size_t hash_index = 0; hash_index < bloom.hash_count; ++hash_index) {
        if (!bloom.test_bit((base_hash + hash_index * step_hash) % bloom.bit_count)) {
            return false;
        }
    }
    return true;
}

TensorBloomResult encode_tensor_with_bloom(const TensorEntry& tensor,
                                           const ReferenceDictionary& dictionary,
                                           bool is_representative,
                                           int group_id,
                                           std::size_t dictionary_bytes_charged,
                                           std::size_t bloom_bytes_charged,
                                           int zstd_level)
{
    const auto start_time = std::chrono::high_resolution_clock::now();
    const auto bits = tensor_to_raw_bits(load_tensor_file(tensor.file_path, tensor.numel));

    std::vector<std::uint8_t> bitmap((bits.size() + 7) / 8, 0u);
    std::vector<std::uint8_t> indices;
    std::vector<std::uint8_t> misses;
    indices.reserve(bits.size() * dictionary.index_width_bytes);
    misses.reserve(bits.size() * sizeof(std::uint32_t));

    std::size_t bloom_positive_count = 0;
    std::size_t exact_match_count = 0;
    for (std::size_t index = 0; index < bits.size(); ++index) {
        if (bloom_maybe_contains(dictionary.bloom, bits[index])) {
            ++bloom_positive_count;
            const auto found = dictionary.ids.find(bits[index]);
            if (found != dictionary.ids.end()) {
                bitmap[index / 8] |= static_cast<std::uint8_t>(1u << (index % 8));
                append_value(indices, found->second, dictionary.index_width_bytes);
                ++exact_match_count;
                continue;
            }
        }
        append_value(misses, bits[index], sizeof(std::uint32_t));
    }

    std::vector<std::uint8_t> payload;
    payload.reserve(bitmap.size() + indices.size() + misses.size() + 3 * sizeof(std::uint32_t));
    append_value(payload, static_cast<std::uint32_t>(bitmap.size()), sizeof(std::uint32_t));
    append_value(payload, static_cast<std::uint32_t>(indices.size()), sizeof(std::uint32_t));
    append_value(payload, static_cast<std::uint32_t>(misses.size()), sizeof(std::uint32_t));
    payload.insert(payload.end(), bitmap.begin(), bitmap.end());
    payload.insert(payload.end(), indices.begin(), indices.end());
    payload.insert(payload.end(), misses.begin(), misses.end());

    const auto compressed_payload = compress_zstd(payload, zstd_level);

    TensorBloomResult result {};
    result.state_type = tensor.state_type;
    result.category = tensor.category;
    result.group_id = group_id;
    result.layer = tensor.layer;
    result.is_representative = is_representative;
    result.original_bytes = tensor.numel * sizeof(float);
    result.payload_bytes = compressed_payload.size();
    result.dictionary_bytes_charged = dictionary_bytes_charged;
    result.bloom_bytes_charged = bloom_bytes_charged;
    result.exact_reconstruction_bytes = result.payload_bytes + result.dictionary_bytes_charged + result.bloom_bytes_charged;
    result.bloom_positive_count = bloom_positive_count;
    result.exact_match_count = exact_match_count;
    result.miss_count = bits.size() - exact_match_count;
    result.bloom_positive_fraction = safe_divide(static_cast<double>(bloom_positive_count), static_cast<double>(bits.size()));
    result.exact_match_fraction = safe_divide(static_cast<double>(exact_match_count), static_cast<double>(bits.size()));
    result.encode_time_ms =
        std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count();
    result.exact_reconstruction_ratio = safe_divide(
        static_cast<double>(result.original_bytes),
        static_cast<double>(result.exact_reconstruction_bytes));
    return result;
}

void write_tensor_report(const fs::path& output_path, const std::vector<TensorBloomResult>& rows)
{
    std::ofstream stream(output_path);
    stream << "state_type,category,group_id,layer,is_representative,original_size_mb,payload_size_mb,dictionary_size_mb_charged,bloom_size_mb_charged,compressed_size_mb,bloom_positive_count,exact_match_count,miss_count,bloom_positive_fraction,exact_match_fraction,compression_ratio,time_ms\n";
    for (const auto& row : rows) {
        stream << row.state_type << ','
               << row.category << ','
               << row.group_id << ','
               << row.layer << ','
               << (row.is_representative ? "true" : "false") << ','
               << bytes_to_mb(row.original_bytes) << ','
               << bytes_to_mb(row.payload_bytes) << ','
               << bytes_to_mb(row.dictionary_bytes_charged) << ','
               << bytes_to_mb(row.bloom_bytes_charged) << ','
               << bytes_to_mb(row.exact_reconstruction_bytes) << ','
               << row.bloom_positive_count << ','
               << row.exact_match_count << ','
               << row.miss_count << ','
               << row.bloom_positive_fraction << ','
               << row.exact_match_fraction << ','
               << row.exact_reconstruction_ratio << ','
               << row.encode_time_ms << '\n';
    }
}

void write_group_report(const fs::path& output_path, const std::vector<GroupBloomSummary>& rows)
{
    std::ofstream stream(output_path);
    stream << "state_type,category,group_id,layers,representative_layer,dictionary_value_count,index_width_bytes,dictionary_size_mb,bloom_size_mb,original_group_size_mb,payload_group_size_mb,compressed_group_size_mb,bloom_positive_count,exact_match_count,miss_count,average_bloom_positive_fraction,average_exact_match_fraction,compression_ratio,total_time_ms\n";
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
               << row.dictionary_value_count << ','
               << row.index_width_bytes << ','
               << bytes_to_mb(row.dictionary_bytes) << ','
               << bytes_to_mb(row.bloom_bytes) << ','
               << bytes_to_mb(row.original_bytes) << ','
               << bytes_to_mb(row.payload_bytes) << ','
               << bytes_to_mb(row.exact_reconstruction_bytes) << ','
               << row.bloom_positive_count << ','
               << row.exact_match_count << ','
               << row.miss_count << ','
               << row.average_bloom_positive_fraction << ','
               << row.average_exact_match_fraction << ','
               << row.exact_reconstruction_ratio << ','
               << row.total_time_ms << '\n';
    }
}

void write_checkpoint_summary(const fs::path& output_path,
                              const CheckpointBloomSummary& summary,
                              const std::map<std::string, std::pair<std::size_t, std::size_t>>& state_type_totals,
                              const CompressionConfig& config)
{
    std::ofstream stream(output_path);
    stream << std::fixed << std::setprecision(6);
    stream << "transform_mode: bloom_guarded_exact_value_reference\n";
    stream << "payload_codec: zstd\n";
    stream << "bloom_bits_per_value: " << config.bloom_bits_per_value << '\n';
    stream << "bloom_hashes: " << config.bloom_hashes << '\n';
    stream << "checkpoint_original_size_mb: " << bytes_to_mb(summary.original_bytes) << '\n';
    stream << "checkpoint_compressed_size_mb: " << bytes_to_mb(summary.exact_reconstruction_bytes) << '\n';
    stream << "checkpoint_compression_ratio: " << summary.exact_reconstruction_ratio << '\n';
    stream << "grouped_tensor_count: " << summary.grouped_tensor_count << '\n';
    stream << "uncompressed_tensor_count: " << summary.uncompressed_tensor_count << '\n';
    stream << "dictionary_size_mb_total: " << bytes_to_mb(summary.total_dictionary_bytes) << '\n';
    stream << "bloom_size_mb_total: " << bytes_to_mb(summary.total_bloom_bytes) << "\n\n";

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
            "Usage: tensor_group_bloom_delta <export_dir> [--bloom-bits-per-value X] [--bloom-hashes N] [--zstd-level N] [--state-type NAME] [--category NAME]");
    }

    CompressionConfig config {};
    config.export_dir = fs::path(argv[1]);
    for (int index = 2; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--bloom-bits-per-value" && index + 1 < argc) {
            config.bloom_bits_per_value = std::stod(argv[++index]);
        } else if (argument == "--bloom-hashes" && index + 1 < argc) {
            config.bloom_hashes = static_cast<std::size_t>(std::stoull(argv[++index]));
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

void run_group_bloom_delta(const CompressionConfig& config)
{
    const auto manifest_entries = load_manifest(config.export_dir / "manifest.csv");
    const auto groups = load_groups(config.export_dir / "analysis" / "plots" / "grouping_summary.csv", config);
    const fs::path output_dir = config.export_dir / "bloom_delta";
    fs::create_directories(output_dir);

    std::set<std::tuple<std::string, std::string, int>> grouped_keys;
    std::vector<TensorBloomResult> tensor_rows;
    std::vector<GroupBloomSummary> group_rows;
    CheckpointBloomSummary checkpoint_summary {};
    std::map<std::string, std::pair<std::size_t, std::size_t>> state_type_totals;

    for (const auto& group : groups) {
        const auto representative_it = manifest_entries.find({group.state_type, group.category, group.representative_layer});
        if (representative_it == manifest_entries.end()) {
            throw std::runtime_error("Missing representative tensor in manifest for group");
        }

        const auto representative_bits = tensor_to_raw_bits(
            load_tensor_file(representative_it->second.file_path, representative_it->second.numel));
        const auto dictionary = build_reference_dictionary(
            representative_bits,
            config.bloom_bits_per_value,
            config.bloom_hashes);

        GroupBloomSummary group_summary {};
        group_summary.state_type = group.state_type;
        group_summary.category = group.category;
        group_summary.group_id = group.group_id;
        group_summary.layers = group.layers;
        group_summary.representative_layer = group.representative_layer;
        group_summary.dictionary_value_count = dictionary.values.size();
        group_summary.dictionary_bytes = dictionary.values.size() * sizeof(std::uint32_t);
        group_summary.bloom_bytes = dictionary.bloom.words.size() * sizeof(std::uint64_t);
        group_summary.index_width_bytes = dictionary.index_width_bytes;

        bool charged_dictionary = false;
        double bloom_fraction_sum = 0.0;
        double exact_fraction_sum = 0.0;
        for (int layer : group.layers) {
            const auto tensor_it = manifest_entries.find({group.state_type, group.category, layer});
            if (tensor_it == manifest_entries.end()) {
                throw std::runtime_error("Missing grouped tensor in manifest");
            }

            grouped_keys.insert({group.state_type, group.category, layer});
            const auto dictionary_bytes = charged_dictionary ? 0u : group_summary.dictionary_bytes;
            const auto bloom_bytes = charged_dictionary ? 0u : group_summary.bloom_bytes;
            auto tensor_result = encode_tensor_with_bloom(
                tensor_it->second,
                dictionary,
                layer == group.representative_layer,
                group.group_id,
                dictionary_bytes,
                bloom_bytes,
                config.zstd_level);
            charged_dictionary = true;

            group_summary.original_bytes += tensor_result.original_bytes;
            group_summary.payload_bytes += tensor_result.payload_bytes;
            group_summary.exact_reconstruction_bytes += tensor_result.exact_reconstruction_bytes;
            group_summary.bloom_positive_count += tensor_result.bloom_positive_count;
            group_summary.exact_match_count += tensor_result.exact_match_count;
            group_summary.miss_count += tensor_result.miss_count;
            bloom_fraction_sum += tensor_result.bloom_positive_fraction;
            exact_fraction_sum += tensor_result.exact_match_fraction;
            group_summary.total_time_ms += tensor_result.encode_time_ms;
            tensor_rows.push_back(tensor_result);
        }

        group_summary.average_bloom_positive_fraction = safe_divide(bloom_fraction_sum, static_cast<double>(group.layers.size()));
        group_summary.average_exact_match_fraction = safe_divide(exact_fraction_sum, static_cast<double>(group.layers.size()));
        group_summary.exact_reconstruction_ratio = safe_divide(
            static_cast<double>(group_summary.original_bytes),
            static_cast<double>(group_summary.exact_reconstruction_bytes));
        group_rows.push_back(group_summary);

        checkpoint_summary.original_bytes += group_summary.original_bytes;
        checkpoint_summary.exact_reconstruction_bytes += group_summary.exact_reconstruction_bytes;
        checkpoint_summary.grouped_tensor_count += group.layers.size();
        checkpoint_summary.total_dictionary_bytes += group_summary.dictionary_bytes;
        checkpoint_summary.total_bloom_bytes += group_summary.bloom_bytes;
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

    write_tensor_report(output_dir / "tensor_bloom_delta.csv", tensor_rows);
    write_group_report(output_dir / "group_bloom_delta.csv", group_rows);
    write_checkpoint_summary(output_dir / "checkpoint_bloom_delta_summary.txt", checkpoint_summary, state_type_totals, config);

    std::cout << "Wrote tensor bloom-delta report to " << (output_dir / "tensor_bloom_delta.csv") << std::endl;
    std::cout << "Wrote group bloom-delta report to " << (output_dir / "group_bloom_delta.csv") << std::endl;
    std::cout << "Wrote checkpoint summary to " << (output_dir / "checkpoint_bloom_delta_summary.txt") << std::endl;
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const auto config = parse_args(argc, argv);
        run_group_bloom_delta(config);
    } catch (const std::exception& error) {
        std::cerr << "tensor_group_bloom_delta failed: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
