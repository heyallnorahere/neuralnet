// quick and dirty resource generator

#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <fstream>
#include <regex>
#include <optional>
#include <memory>

#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

#include <cstdint>
#include <cstring>
#include <stddef.h>

static const std::unordered_map<char, std::string> s_aliases = { { 'o', "output" },
                                                                 { 'd', "data-template" },
                                                                 { 't', "template" },
                                                                 { 'i', "inputs" },
                                                                 { 'r', "relative" } };

static const std::unordered_set<std::string> s_boolean_flags = { "stdout" };
static bool s_logging_enabled = true;

struct token_t {
    size_t offset, length;
    std::string name;
};

static void read_file(const fs::path& path, std::string& contents) {
    std::ifstream file(path, std::ios::in);
    if (!file.is_open()) {
        throw std::runtime_error("could not open file: " + path.string());
    }

    std::stringstream ss;
    std::string line;
    while (std::getline(file, line)) {
        ss << line << '\n';
    }

    contents = ss.str();
}

static void read_binary(const fs::path& path, std::vector<uint8_t>& data) {
    std::ifstream file(path, std::ios::binary | std::ios::in);
    if (!file.is_open()) {
        throw std::runtime_error("could not open file: " + path.string());
    }

    file.seekg(0, file.end);
    data.resize(file.tellg());

    file.seekg(0, file.beg);
    file.read((char*)data.data(), data.size());
}

class source_template {
public:
    source_template(const fs::path& path) {
        m_path = path;
        if (s_logging_enabled) {
            std::cout << "parsing template " << path << std::endl;
        }

        read_file(path, m_source);
        parse_file();
    }

    source_template(const source_template&) = delete;
    source_template& operator=(const source_template&) = delete;

    template <typename T>
    void execute(const std::unordered_map<std::string, std::stringstream>& tokens, T& stream) const {
        std::string result = m_source;
        std::vector<token_t> token_map(m_tokens);

        for (size_t i = 0; i < token_map.size(); i++) {
            const token_t& token = token_map[i];

            auto it = tokens.find(token.name);
            if (it == tokens.end()) {
                throw std::runtime_error("no value for key " + token.name);
            }

            std::string data = it->second.str();
            result.replace(token.offset, token.length, data);

            for (size_t j = i + 1; j < token_map.size(); j++) {
                // lets be safe about this
                int64_t data_length = data.length();
                int64_t token_length = token.length;

                token_map[j].offset += data_length - token_length;
            }
        }

        stream << result;
    }

    bool has_token(const std::string& name) const {
        for (const auto& token : m_tokens) {
            if (token.name == name) {
                return true;
            }
        }

        return false;
    }

private:
    void parse_file() {
        size_t position = 0;

        static const std::string open_bracket_symbol = "${";
        static const std::string close_bracket_symbol = "}";

        while (true) {
            size_t open_bracket = m_source.find(open_bracket_symbol, position);
            if (open_bracket == std::string::npos) {
                break;
            }

            size_t close_bracket = m_source.find(close_bracket_symbol, open_bracket);
            if (close_bracket == std::string::npos) {
                throw std::runtime_error("malformed template!");
            }

            // advance the cursor, "close_bracket" is the INDEX of the close bracket
            size_t token_start = open_bracket + open_bracket_symbol.length();
            position = close_bracket + close_bracket_symbol.length();

            token_t& token = m_tokens.emplace_back();
            token.offset = open_bracket;
            token.length = position - open_bracket;
            token.name = m_source.substr(token_start, close_bracket - token_start);
        }
    }

    fs::path m_path;
    std::string m_source;
    std::vector<token_t> m_tokens;
};

struct arguments_t {
    std::vector<std::string> arguments;
    std::unordered_map<std::string, std::string> parameters;
    std::unordered_set<std::string> flags;
};

// this is spaghetti. im sorry
static void parse_args(const std::vector<std::string>& strings, arguments_t& args) {
    std::string current_param;
    for (size_t i = 1; i < strings.size(); i++) {
        const std::string& current_argument = strings[i];
        if (current_argument.length() == 0) {
            continue;
        }

        bool is_double_hypen = current_argument.length() == 2 && current_argument[1] == '-';
        if (current_argument[0] != '-' || current_argument.length() == 1 || is_double_hypen) {
            if (current_param.empty()) {
                args.arguments.push_back(current_argument);
            } else {
                args.parameters[current_param] = current_argument;
                current_param.clear();
            }

            continue;
        } else if (!current_param.empty()) {
            break;
        }

        if (current_argument[1] != '-') {
            std::string alias = current_argument.substr(1);
            for (char c : alias) {
                std::string param_name;
                if (s_aliases.find(c) != s_aliases.end()) {
                    param_name = s_aliases.at(c);
                } else {
                    param_name = std::string(&c, 1);
                }

                if (s_boolean_flags.find(param_name) != s_boolean_flags.end()) {
                    args.flags.insert(param_name);
                } else {
                    if (alias.length() > 1) {
                        throw std::runtime_error(
                            "attempted to use a non-boolean flag in flag glob!");
                    }

                    current_param = param_name;
                }
            }
        } else {
            auto param_name = current_argument.substr(2);
            if (s_boolean_flags.find(param_name) != s_boolean_flags.end()) {
                args.flags.insert(param_name);
            } else {
                current_param = param_name;
            }
        }
    }

    if (!current_param.empty()) {
        throw std::runtime_error("no value for parameter " + current_param + "!");
    }
}

template <typename K, typename V>
static const V& value_or_throw(const std::unordered_map<K, V>& map, const K& key,
                               const std::string& message) {
    auto it = map.find(key);
    if (it == map.end()) {
        throw std::runtime_error(message);
    }

    return it->second;
}

template <typename T>
static void parse_cmake_list(const std::string& data, T& list) {
    std::cout << "parsing list " << data << std::endl;
    size_t position = 0;

    while (true) {
        size_t delimiter = data.find(',', position);
        size_t length = delimiter != std::string::npos ? delimiter - position : delimiter;

        std::string substr = data.substr(position, length);
        if (substr.length() > 0) {
            list.push_back(substr);
        }

        if (delimiter == std::string::npos) {
            break;
        }

        position = delimiter + 1;
    }
}

int main(int argc, const char** argv) {
    std::vector<std::string> strings;
    for (int i = 0; i < argc; i++) {
        strings.push_back(argv[i]);
    }

    arguments_t args;
    parse_args(strings, args);

    bool stdout_output = args.flags.contains("stdout");
    if (stdout_output) {
        s_logging_enabled = false;
    }

    fs::path main_template_path = value_or_throw<std::string>(
        args.parameters, "template", "no template passed! use -t or --template");

    fs::path data_template_path = value_or_throw<std::string>(
        args.parameters, "data-template", "no data template passed! use -d or --data-template");

    std::string input_list = value_or_throw<std::string>(args.parameters, "inputs",
                                                         "no inputs passed! use -i or --inputs");

    auto main_template = std::make_unique<source_template>(main_template_path);
    auto data_template = std::make_unique<source_template>(data_template_path);

    if (!main_template->has_token("content")) {
        throw std::runtime_error("main template does not have the \"content\" token!");
    }

    if (!data_template->has_token("data")) {
        throw std::runtime_error("data template does not have the \"data\" token!");
    }

    std::vector<fs::path> inputs;
    parse_cmake_list(input_list, inputs);

    std::stringstream content;
    for (const auto& path : inputs) {
        std::vector<uint8_t> data;
        read_binary(path, data);

        fs::path relative_path;
        if (args.parameters.contains("relative")) {
            relative_path = fs::relative(path, args.parameters.at("relative"));
        } else {
            relative_path = path;
        }

        std::stringstream data_text;
        for (size_t i = 0; i < data.size(); i++) {
            if (i > 0) {
                data_text << ", ";
            }

            data_text << "0x";
            data_text << std::hex << std::setw(2) << std::setfill('0') << (int)data[i];
        }

        std::unordered_map<std::string, std::stringstream> data_tokens;
        data_tokens["data"] << "{ " << data_text.str() << " }";
        data_tokens["path"] << path;
        data_tokens["relative_path"] << relative_path;
        data_tokens["filename"] << path.filename();

        data_template->execute(data_tokens, content);
    }

    std::unordered_map<std::string, std::stringstream> tokens;
    tokens["content"] << content.str();

    std::optional<fs::path> output_file;
    if (args.parameters.contains("output")) {
        output_file = args.parameters.at("output");
    }

    if (output_file.has_value()) {
        if (stdout_output) {
            std::cout << "--stdout and --output passed; ignoring --stdout" << std::endl;
        }

        fs::path path = output_file.value();
        fs::path directory = path.parent_path();

        if (!fs::is_directory(directory)) {
            fs::create_directories(directory);
        }

        std::ofstream file(path);
        main_template->execute(tokens, file);
    } else if (stdout_output) {
        main_template->execute(tokens, std::cout);
    } else {
        throw std::runtime_error("no output method chosen! use --output <file> or --stdout");
    }

    return 0;
}