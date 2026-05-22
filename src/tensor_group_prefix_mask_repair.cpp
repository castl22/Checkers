#include <algorithm>
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
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;

using Symbol = std::uint64_t;

constexpr double k_megabyte_divisor = 1024.0 * 1024.0;
constexpr std::size_t k_default_max_rules = 50;
constexpr std::size_t k_min_rule_frequency = 2;
constexpr Symbol k_first_rule_symbol = 1ull << 63;

struct TensorEntry {
    std::string tensor_key;
    std::string state_type;
    std::string category;
    int layer {0};
    std::size_t numel {0};
    std::string dtype;
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
    std::size_t max_rules {k_default_max_rules};
    std::optional<std::string> state_type_filter;
    std::optional<std::string> category_filter;
    std::optional<int> group_id_filter;
    std::optional<std::size_t> masked_mantissa_bits;
};

struct MaskedTensorStreams {
    std::vector<Symbol> masked_words;
    std::size_t masked_bits {0};
    std::size_t remaining_value_bits {0};
    std::size_t masked_only_bytes {0};
    std::size_t residual_raw_bytes {0};
};

struct DTypeLayout {
    std::string name;
    std::size_t storage_bytes {0};
    std::size_t mantissa_bits {0};
};

struct SharedGrammarRule {
    Symbol replacement_symbol {0};
    Symbol lhs {0};
    Symbol rhs {0};
    std::size_t frequency {0};
};

struct GrammarBuildResult {
    std::vector<SharedGrammarRule> rules;
    std::vector<Symbol> compressed_sequence;
    std::size_t max_pair_frequency {0};
};

struct TensorSharedGrammarResult {
    std::string state_type;
    std::string category;
    int group_id {-1};
    int layer {0};
    bool is_representative {false};
    std::string dtype;
    std::size_t numel {0};
    std::size_t original_bytes {0};
    std::size_t masked_mantissa_bits {0};
    std::size_t masked_only_bytes {0};
    std::size_t compressed_token_count {0};
    std::size_t cfg_payload_bytes {0};
    std::size_t grammar_bytes_charged {0};
    std::size_t residual_raw_bytes {0};
    std::size_t lossy_compressed_bytes {0};
    std::size_t rule_count {0};
    double masking_only_ratio {1.0};
    double compression_ratio {1.0};
    double runtime_ms {0.0};
};

struct GroupSharedGrammarSummary {
    std::string state_type;
    std::string category;
    int group_id {0};
    std::vector<int> layers;
    int representative_layer {0};
    std::string dtype;
    std::size_t masked_mantissa_bits {0};
    std::size_t rule_count {0};
    std::size_t original_bytes {0};
    std::size_t masked_only_bytes {0};
    std::size_t cfg_payload_bytes {0};
    std::size_t grammar_bytes {0};
    std::size_t residual_raw_bytes {0};
    std::size_t lossy_compressed_bytes {0};
    std::size_t max_pair_frequency {0};
    double masking_only_ratio {1.0};
    double compression_ratio {1.0};
    double total_time_ms {0.0};
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

std::size_t bits_to_bytes(std::size_t bits)
{
    return (bits + 7u) / 8u;
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
    int layer = 0;
    while (stream >> layer) {
        layers.push_back(layer);
    }
    return layers;
}

DTypeLayout parse_dtype_layout(const std::string& dtype)
{
    if (dtype == "float32") {
        return DTypeLayout {dtype, 4, 23};
    }
    if (dtype == "float16" || dtype == "fp16") {
        return DTypeLayout {dtype, 2, 10};
    }
    if (dtype == "bfloat16" || dtype == "bf16") {
        return DTypeLayout {dtype, 2, 7};
    }
    throw std::runtime_error("Unsupported dtype for mantissa-mask compression: " + dtype);
}

std::size_t bit_width_for_symbol_count(std::size_t symbol_count)
{
    if (symbol_count <= 1) {
        return 1;
    }

    std::size_t width = 0;
    std::size_t values = symbol_count - 1;
    while (values > 0) {
        ++width;
        values >>= 1u;
    }
    return std::max<std::size_t>(width, 1);
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
        entry.dtype = columns[5];
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
        if (config.group_id_filter && group.group_id != *config.group_id_filter) {
            continue;
        }
        groups.push_back(group);
    }

    return groups;
}

std::vector<Symbol> load_raw_words(const TensorEntry& entry, const DTypeLayout& layout)
{
    std::ifstream stream(entry.file_path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Failed to open tensor payload: " + entry.file_path.string());
    }

    const std::size_t total_bytes = entry.numel * layout.storage_bytes;
    std::vector<std::uint8_t> raw(total_bytes, 0u);
    stream.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(total_bytes));
    if (stream.gcount() != static_cast<std::streamsize>(total_bytes)) {
        throw std::runtime_error("Tensor payload size mismatch: " + entry.file_path.string());
    }

    std::vector<Symbol> words(entry.numel, 0);
    if (layout.storage_bytes == 4) {
        for (std::size_t index = 0; index < entry.numel; ++index) {
            words[index] = static_cast<Symbol>(raw[4 * index]) |
                (static_cast<Symbol>(raw[4 * index + 1]) << 8u) |
                (static_cast<Symbol>(raw[4 * index + 2]) << 16u) |
                (static_cast<Symbol>(raw[4 * index + 3]) << 24u);
        }
        return words;
    }

    for (std::size_t index = 0; index < entry.numel; ++index) {
        words[index] = static_cast<Symbol>(raw[2 * index]) |
            (static_cast<Symbol>(raw[2 * index + 1]) << 8u);
    }
    return words;
}

MaskedTensorStreams build_masked_sequence(
    const std::vector<Symbol>& raw_words,
    const DTypeLayout& layout,
    const CompressionConfig& config)
{
    const std::size_t masked_bits = std::min<std::size_t>(
        config.masked_mantissa_bits.value_or(layout.mantissa_bits),
        layout.mantissa_bits);
    const std::size_t total_value_bits = layout.storage_bytes * 8u;
    const Symbol masked_bit_mask = masked_bits == 0
        ? 0
        : ((static_cast<Symbol>(1) << masked_bits) - 1u);

    MaskedTensorStreams streams {};
    streams.masked_bits = masked_bits;
    streams.remaining_value_bits = total_value_bits - masked_bits;
    streams.masked_only_bytes = bits_to_bytes(raw_words.size() * streams.remaining_value_bits);
    streams.residual_raw_bytes = bits_to_bytes(raw_words.size() * masked_bits);
    streams.masked_words.reserve(raw_words.size());
    for (Symbol word : raw_words) {
        streams.masked_words.push_back(word & ~masked_bit_mask);
    }

    return streams;
}

struct PairKey {
    Symbol lhs {0};
    Symbol rhs {0};

    bool operator==(const PairKey& other) const
    {
        return lhs == other.lhs && rhs == other.rhs;
    }
};

struct PairKeyHash {
    std::size_t operator()(const PairKey& key) const
    {
        const auto lhs_hash = std::hash<Symbol> {}(key.lhs);
        const auto rhs_hash = std::hash<Symbol> {}(key.rhs);
        return lhs_hash ^ (rhs_hash + 0x9e3779b97f4a7c15ull + (lhs_hash << 6u) + (lhs_hash >> 2u));
    }
};

std::optional<std::pair<PairKey, std::size_t>> find_best_pair(
    const std::vector<Symbol>& tokens,
    std::size_t& max_pair_frequency)
{
    if (tokens.size() < 2) {
        return std::nullopt;
    }

    std::unordered_map<PairKey, std::size_t, PairKeyHash> counts;
    counts.reserve(tokens.size());
    for (std::size_t index = 0; index + 1 < tokens.size(); ++index) {
        ++counts[{tokens[index], tokens[index + 1]}];
    }

    std::size_t best_count = 0;
    PairKey best_pair {};
    for (const auto& [pair_key, count] : counts) {
        max_pair_frequency = std::max(max_pair_frequency, count);
        if (count > best_count) {
            best_count = count;
            best_pair = pair_key;
        }
    }

    if (best_count < k_min_rule_frequency) {
        return std::nullopt;
    }
    return std::make_pair(best_pair, best_count);
}

std::vector<Symbol> apply_rule(const std::vector<Symbol>& tokens, const SharedGrammarRule& rule, std::size_t& replaced_count)
{
    std::vector<Symbol> output;
    output.reserve(tokens.size());
    replaced_count = 0;

    std::size_t index = 0;
    while (index < tokens.size()) {
        if (index + 1 < tokens.size() && tokens[index] == rule.lhs && tokens[index + 1] == rule.rhs) {
            output.push_back(rule.replacement_symbol);
            index += 2;
            ++replaced_count;
        } else {
            output.push_back(tokens[index]);
            ++index;
        }
    }
    return output;
}

GrammarBuildResult build_shared_grammar(const std::vector<Symbol>& sequence, std::size_t max_rules)
{
    GrammarBuildResult result {};
    result.compressed_sequence = sequence;

    Symbol next_symbol = k_first_rule_symbol;
    result.rules.reserve(max_rules);
    for (std::size_t rule_index = 0; rule_index < max_rules; ++rule_index) {
        const auto best_pair = find_best_pair(result.compressed_sequence, result.max_pair_frequency);
        if (!best_pair) {
            break;
        }

        SharedGrammarRule rule {};
        rule.replacement_symbol = next_symbol++;
        rule.lhs = best_pair->first.lhs;
        rule.rhs = best_pair->first.rhs;
        rule.frequency = best_pair->second;

        std::size_t replaced_count = 0;
        auto updated_tokens = apply_rule(result.compressed_sequence, rule, replaced_count);
        if (updated_tokens.size() >= result.compressed_sequence.size() || replaced_count == 0) {
            break;
        }

        result.compressed_sequence = std::move(updated_tokens);
        result.rules.push_back(rule);
    }

    return result;
}

std::vector<Symbol> apply_existing_grammar(const std::vector<Symbol>& sequence, const std::vector<SharedGrammarRule>& rules)
{
    std::vector<Symbol> current = sequence;
    for (const auto& rule : rules) {
        std::size_t replaced_count = 0;
        current = apply_rule(current, rule, replaced_count);
    }
    return current;
}

std::string format_layers(const std::vector<int>& layers)
{
    std::ostringstream stream;
    for (std::size_t index = 0; index < layers.size(); ++index) {
        if (index > 0) {
            stream << ' ';
        }
        stream << layers[index];
    }
    return stream.str();
}

void write_tensor_report(const fs::path& output_path, const std::vector<TensorSharedGrammarResult>& rows)
{
    std::ofstream stream(output_path);
    stream << "state_type,category,group_id,layer,is_representative,dtype,numel,masked_mantissa_bits,original_size_mb,masked_only_size_mb,cfg_payload_size_mb,grammar_size_mb_charged,lossy_compressed_size_mb,residual_raw_size_mb,masking_only_ratio,compression_ratio,compressed_token_count,rule_count,time_ms\n";
    for (const auto& row : rows) {
        stream << row.state_type << ','
               << row.category << ','
               << row.group_id << ','
               << row.layer << ','
               << (row.is_representative ? 1 : 0) << ','
               << row.dtype << ','
               << row.numel << ','
               << row.masked_mantissa_bits << ','
               << bytes_to_mb(row.original_bytes) << ','
               << bytes_to_mb(row.masked_only_bytes) << ','
               << bytes_to_mb(row.cfg_payload_bytes) << ','
               << bytes_to_mb(row.grammar_bytes_charged) << ','
               << bytes_to_mb(row.lossy_compressed_bytes) << ','
               << bytes_to_mb(row.residual_raw_bytes) << ','
               << row.masking_only_ratio << ','
               << row.compression_ratio << ','
               << row.compressed_token_count << ','
               << row.rule_count << ','
               << row.runtime_ms << '\n';
    }
}

void write_group_report(const fs::path& output_path, const std::vector<GroupSharedGrammarSummary>& rows)
{
    std::ofstream stream(output_path);
    stream << "state_type,category,group_id,layers,representative_layer,dtype,masked_mantissa_bits,rule_count,max_pair_frequency,original_group_size_mb,masked_only_group_size_mb,cfg_payload_group_size_mb,grammar_size_mb,lossy_compressed_group_size_mb,residual_raw_group_size_mb,masking_only_ratio,compression_ratio,total_time_ms\n";
    for (const auto& row : rows) {
        stream << row.state_type << ','
               << row.category << ','
               << row.group_id << ','
               << '"' << format_layers(row.layers) << '"' << ','
               << row.representative_layer << ','
               << row.dtype << ','
               << row.masked_mantissa_bits << ','
               << row.rule_count << ','
               << row.max_pair_frequency << ','
               << bytes_to_mb(row.original_bytes) << ','
               << bytes_to_mb(row.masked_only_bytes) << ','
               << bytes_to_mb(row.cfg_payload_bytes) << ','
               << bytes_to_mb(row.grammar_bytes) << ','
               << bytes_to_mb(row.lossy_compressed_bytes) << ','
               << bytes_to_mb(row.residual_raw_bytes) << ','
               << row.masking_only_ratio << ','
               << row.compression_ratio << ','
               << row.total_time_ms << '\n';
    }
}

void write_summary(
    const fs::path& output_path,
    const std::vector<GroupSharedGrammarSummary>& rows,
    const CompressionConfig& config)
{
    std::ofstream stream(output_path);
    stream << std::fixed << std::setprecision(6);
    stream << "transform_mode: shared_group_mantissa_mask_cfg\n";
    stream << "state_type_filter: " << (config.state_type_filter ? *config.state_type_filter : "<all>") << "\n";
    stream << "category_filter: " << (config.category_filter ? *config.category_filter : "<all>") << "\n";
    stream << "group_id_filter: ";
    if (config.group_id_filter) {
        stream << *config.group_id_filter;
    } else {
        stream << "<all>";
    }
    stream << "\n";
    stream << "masked_mantissa_bits_request: ";
    if (config.masked_mantissa_bits) {
        stream << *config.masked_mantissa_bits;
    } else {
        stream << "all_mantissa_bits";
    }
    stream << "\n";
    stream << "max_rules_shared_grammar: " << config.max_rules << "\n\n";

    std::size_t total_original_bytes = 0;
    std::size_t total_masked_only_bytes = 0;
    std::size_t total_cfg_payload_bytes = 0;
    std::size_t total_lossy_bytes = 0;
    std::size_t total_residual_raw_bytes = 0;
    std::size_t total_grammar_bytes = 0;

    for (const auto& row : rows) {
        total_original_bytes += row.original_bytes;
        total_masked_only_bytes += row.masked_only_bytes;
        total_cfg_payload_bytes += row.cfg_payload_bytes;
        total_lossy_bytes += row.lossy_compressed_bytes;
        total_residual_raw_bytes += row.residual_raw_bytes;
        total_grammar_bytes += row.grammar_bytes;

        stream << row.state_type << '/' << row.category << "/group_" << row.group_id
               << ": layers=" << format_layers(row.layers)
               << ", representative_layer=" << row.representative_layer
               << ", dtype=" << row.dtype
               << ", masked_mantissa_bits=" << row.masked_mantissa_bits
               << ", rule_count=" << row.rule_count
               << ", original_group_size_mb=" << bytes_to_mb(row.original_bytes)
             << ", masked_only_group_size_mb=" << bytes_to_mb(row.masked_only_bytes)
             << ", cfg_payload_group_size_mb=" << bytes_to_mb(row.cfg_payload_bytes)
               << ", grammar_size_mb=" << bytes_to_mb(row.grammar_bytes)
             << ", lossy_compressed_group_size_mb=" << bytes_to_mb(row.lossy_compressed_bytes)
             << ", residual_raw_group_size_mb=" << bytes_to_mb(row.residual_raw_bytes)
             << ", masking_only_ratio=" << row.masking_only_ratio
             << ", compression_ratio=" << row.compression_ratio
               << '\n';
    }

    stream << "\nselected_groups_original_size_mb: " << bytes_to_mb(total_original_bytes) << '\n';
        stream << "selected_groups_masked_only_size_mb: " << bytes_to_mb(total_masked_only_bytes) << '\n';
        stream << "selected_groups_cfg_payload_size_mb: " << bytes_to_mb(total_cfg_payload_bytes) << '\n';
    stream << "selected_groups_total_grammar_mb: " << bytes_to_mb(total_grammar_bytes) << '\n';
        stream << "selected_groups_lossy_compressed_size_mb: " << bytes_to_mb(total_lossy_bytes) << '\n';
        stream << "selected_groups_residual_raw_size_mb: " << bytes_to_mb(total_residual_raw_bytes) << '\n';
        stream << "selected_groups_masking_only_ratio: "
            << safe_divide(static_cast<double>(total_original_bytes), static_cast<double>(total_masked_only_bytes)) << '\n';
        stream << "selected_groups_compression_ratio: "
            << safe_divide(static_cast<double>(total_original_bytes), static_cast<double>(total_lossy_bytes)) << '\n';
}

CompressionConfig parse_args(int argc, char** argv)
{
    if (argc < 2) {
        throw std::runtime_error(
            "Usage: tensor_group_prefix_mask_repair <export_dir> [--state-type NAME] [--category NAME] [--group-id N] [--max-rules N] [--masked-mantissa-bits N]");
    }

    CompressionConfig config {};
    config.export_dir = fs::path(argv[1]);
    for (int index = 2; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--state-type" && index + 1 < argc) {
            config.state_type_filter = std::string(argv[++index]);
        } else if (argument == "--category" && index + 1 < argc) {
            config.category_filter = std::string(argv[++index]);
        } else if (argument == "--group-id" && index + 1 < argc) {
            config.group_id_filter = std::stoi(argv[++index]);
        } else if (argument == "--max-rules" && index + 1 < argc) {
            config.max_rules = static_cast<std::size_t>(std::stoull(argv[++index]));
        } else if (argument == "--masked-mantissa-bits" && index + 1 < argc) {
            config.masked_mantissa_bits = static_cast<std::size_t>(std::stoull(argv[++index]));
        } else {
            throw std::runtime_error("Unknown or incomplete argument: " + argument);
        }
    }
    return config;
}

void run_prefix_mask_repair(const CompressionConfig& config)
{
    const auto manifest_entries = load_manifest(config.export_dir / "manifest.csv");
    const auto groups = load_groups(config.export_dir / "analysis" / "plots" / "grouping_summary.csv", config);
    if (groups.empty()) {
        throw std::runtime_error("No matching grouping rows found for the requested filters");
    }

    const fs::path output_dir = config.export_dir / "prefix_mask_repair";
    fs::create_directories(output_dir);

    std::vector<TensorSharedGrammarResult> tensor_rows;
    std::vector<GroupSharedGrammarSummary> group_rows;

    for (const auto& group : groups) {
        const auto representative_it = manifest_entries.find({group.state_type, group.category, group.representative_layer});
        if (representative_it == manifest_entries.end()) {
            throw std::runtime_error("Missing representative tensor in manifest for requested group");
        }

        const auto representative_layout = parse_dtype_layout(representative_it->second.dtype);
        const auto representative_raw_words = load_raw_words(representative_it->second, representative_layout);
        const auto representative_streams =
            build_masked_sequence(representative_raw_words, representative_layout, config);
        const auto grammar = build_shared_grammar(representative_streams.masked_words, config.max_rules);
        const std::size_t base_symbol_count = static_cast<std::size_t>(1) << representative_streams.remaining_value_bits;
        const std::size_t token_bit_width = bit_width_for_symbol_count(base_symbol_count + grammar.rules.size());
        const auto grammar_bytes = bits_to_bytes(token_bit_width * 2u * grammar.rules.size());

        GroupSharedGrammarSummary group_summary {};
        group_summary.state_type = group.state_type;
        group_summary.category = group.category;
        group_summary.group_id = group.group_id;
        group_summary.layers = group.layers;
        group_summary.representative_layer = group.representative_layer;
        group_summary.dtype = representative_it->second.dtype;
        group_summary.masked_mantissa_bits = representative_streams.masked_bits;
        group_summary.rule_count = grammar.rules.size();
        group_summary.grammar_bytes = grammar_bytes;
        group_summary.max_pair_frequency = grammar.max_pair_frequency;

        bool charged_grammar = false;
        for (int layer : group.layers) {
            const auto tensor_it = manifest_entries.find({group.state_type, group.category, layer});
            if (tensor_it == manifest_entries.end()) {
                throw std::runtime_error("Missing grouped tensor in manifest");
            }
            if (tensor_it->second.dtype != representative_it->second.dtype) {
                throw std::runtime_error("Mixed dtypes inside one group are not supported");
            }

            const auto start_time = std::chrono::high_resolution_clock::now();
            const auto layout = parse_dtype_layout(tensor_it->second.dtype);
            const auto raw_words = load_raw_words(tensor_it->second, layout);
            const auto masked_streams = build_masked_sequence(raw_words, layout, config);
            const auto compressed_sequence = layer == group.representative_layer
                ? grammar.compressed_sequence
                : apply_existing_grammar(masked_streams.masked_words, grammar.rules);
            const auto payload_bytes = bits_to_bytes(token_bit_width * compressed_sequence.size());
            const auto grammar_bytes_charged = charged_grammar ? 0u : grammar_bytes;
            charged_grammar = true;

            TensorSharedGrammarResult tensor_result {};
            tensor_result.state_type = group.state_type;
            tensor_result.category = group.category;
            tensor_result.group_id = group.group_id;
            tensor_result.layer = layer;
            tensor_result.is_representative = layer == group.representative_layer;
            tensor_result.dtype = tensor_it->second.dtype;
            tensor_result.numel = tensor_it->second.numel;
            tensor_result.original_bytes = tensor_it->second.numel * layout.storage_bytes;
            tensor_result.masked_mantissa_bits = masked_streams.masked_bits;
            tensor_result.masked_only_bytes = masked_streams.masked_only_bytes;
            tensor_result.compressed_token_count = compressed_sequence.size();
            tensor_result.cfg_payload_bytes = payload_bytes;
            tensor_result.grammar_bytes_charged = grammar_bytes_charged;
            tensor_result.residual_raw_bytes = masked_streams.residual_raw_bytes;
            tensor_result.lossy_compressed_bytes = payload_bytes + grammar_bytes_charged;
            tensor_result.rule_count = grammar.rules.size();
            tensor_result.masking_only_ratio = safe_divide(
                static_cast<double>(tensor_result.original_bytes),
                static_cast<double>(tensor_result.masked_only_bytes));
            tensor_result.compression_ratio = safe_divide(
                static_cast<double>(tensor_result.original_bytes),
                static_cast<double>(tensor_result.lossy_compressed_bytes));
            tensor_result.runtime_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - start_time).count();
            tensor_rows.push_back(tensor_result);

            group_summary.original_bytes += tensor_result.original_bytes;
            group_summary.masked_only_bytes += tensor_result.masked_only_bytes;
            group_summary.cfg_payload_bytes += tensor_result.cfg_payload_bytes;
            group_summary.residual_raw_bytes += tensor_result.residual_raw_bytes;
            group_summary.lossy_compressed_bytes += tensor_result.lossy_compressed_bytes;
            group_summary.total_time_ms += tensor_result.runtime_ms;
        }

        group_summary.masking_only_ratio = safe_divide(
            static_cast<double>(group_summary.original_bytes),
            static_cast<double>(group_summary.masked_only_bytes));
        group_summary.compression_ratio = safe_divide(
            static_cast<double>(group_summary.original_bytes),
            static_cast<double>(group_summary.lossy_compressed_bytes));
        group_rows.push_back(group_summary);
    }

    write_tensor_report(output_dir / "tensor_prefix_mask_repair.csv", tensor_rows);
    write_group_report(output_dir / "group_prefix_mask_repair.csv", group_rows);
    write_summary(output_dir / "checkpoint_prefix_mask_repair_summary.txt", group_rows, config);

    std::cout << "Wrote tensor prefix-mask report to " << (output_dir / "tensor_prefix_mask_repair.csv") << std::endl;
    std::cout << "Wrote group prefix-mask report to " << (output_dir / "group_prefix_mask_repair.csv") << std::endl;
    std::cout << "Wrote prefix-mask summary to " << (output_dir / "checkpoint_prefix_mask_repair_summary.txt") << std::endl;
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const auto config = parse_args(argc, argv);
        run_prefix_mask_repair(config);
    } catch (const std::exception& error) {
        std::cerr << "tensor_group_prefix_mask_repair failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
