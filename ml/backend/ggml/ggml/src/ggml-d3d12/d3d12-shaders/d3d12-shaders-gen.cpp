#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct options {
    fs::path    input_dir;
    fs::path    output;
    fs::path    dxc;
    fs::path    depfile;
    std::string target = "cs_6_6";
    bool        debug = false;
    bool        depfile_set = false;
};

struct shader_source {
    fs::path    path;
    std::string logical_name;
    std::string symbol_name;
};

struct compiled_shader {
    std::string              logical_name;
    std::string              symbol_name;
    std::vector<std::uint8_t> data;
};

struct process_result {
    DWORD       exit_code = 0;
    std::string output;
};

class unique_handle {
public:
    unique_handle() = default;
    explicit unique_handle(HANDLE handle) : handle_(handle) {}

    unique_handle(const unique_handle &) = delete;
    unique_handle & operator=(const unique_handle &) = delete;

    unique_handle(unique_handle && other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    unique_handle & operator=(unique_handle && other) noexcept {
        if (this != &other) {
            reset();
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    ~unique_handle() {
        reset();
    }

    HANDLE get() const {
        return handle_;
    }

    HANDLE * put() {
        reset();
        return &handle_;
    }

    void reset(HANDLE handle = nullptr) {
        if (handle_ != nullptr && handle_ != INVALID_HANDLE_VALUE) {
            CloseHandle(handle_);
        }
        handle_ = handle;
    }

private:
    HANDLE handle_ = nullptr;
};

std::string usage() {
    return "usage: d3d12-shaders-gen.exe --input-dir <dir> --output <ggml-d3d12-shaders.hpp> "
           "--dxc <path\\to\\dxc.exe> [--target cs_6_6] [--debug] [--depfile <path>]";
}

bool starts_with_flag_prefix(const std::string & value) {
    return value.size() >= 2 && value[0] == '-' && value[1] == '-';
}

bool parse_args(int argc, char ** argv, options & opts, std::string & error) {
    bool seen_input_dir = false;
    bool seen_output = false;
    bool seen_dxc = false;
    bool seen_target = false;
    bool seen_debug = false;
    bool seen_depfile = false;

    auto require_value = [&](int & index, const std::string & flag, std::string & value) -> bool {
        if (index + 1 >= argc || starts_with_flag_prefix(argv[index + 1])) {
            error = "missing value for " + flag;
            return false;
        }
        ++index;
        value = argv[index];
        return true;
    };

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--input-dir") {
            if (seen_input_dir) {
                error = "duplicate flag --input-dir";
                return false;
            }
            std::string value;
            if (!require_value(i, arg, value)) {
                return false;
            }
            opts.input_dir = fs::path(value);
            seen_input_dir = true;
        } else if (arg == "--output") {
            if (seen_output) {
                error = "duplicate flag --output";
                return false;
            }
            std::string value;
            if (!require_value(i, arg, value)) {
                return false;
            }
            opts.output = fs::path(value);
            seen_output = true;
        } else if (arg == "--dxc") {
            if (seen_dxc) {
                error = "duplicate flag --dxc";
                return false;
            }
            std::string value;
            if (!require_value(i, arg, value)) {
                return false;
            }
            opts.dxc = fs::path(value);
            seen_dxc = true;
        } else if (arg == "--target") {
            if (seen_target) {
                error = "duplicate flag --target";
                return false;
            }
            std::string value;
            if (!require_value(i, arg, value)) {
                return false;
            }
            if (value.empty()) {
                error = "empty value for --target";
                return false;
            }
            opts.target = value;
            seen_target = true;
        } else if (arg == "--debug") {
            if (seen_debug) {
                error = "duplicate flag --debug";
                return false;
            }
            opts.debug = true;
            seen_debug = true;
        } else if (arg == "--depfile") {
            if (seen_depfile) {
                error = "duplicate flag --depfile";
                return false;
            }
            std::string value;
            if (!require_value(i, arg, value)) {
                return false;
            }
            opts.depfile = fs::path(value);
            opts.depfile_set = true;
            seen_depfile = true;
        } else if (starts_with_flag_prefix(arg)) {
            error = "unknown flag " + arg;
            return false;
        } else {
            error = "unexpected positional argument " + arg;
            return false;
        }
    }

    if (!seen_input_dir) {
        error = "missing required flag --input-dir";
        return false;
    }
    if (!seen_output) {
        error = "missing required flag --output";
        return false;
    }
    if (!seen_dxc) {
        error = "missing required flag --dxc";
        return false;
    }

    return true;
}

std::string format_windows_error(DWORD error_code) {
    LPSTR buffer = nullptr;
    const DWORD length = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr,
        error_code,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        reinterpret_cast<LPSTR>(&buffer),
        0,
        nullptr);

    std::string message;
    if (length != 0 && buffer != nullptr) {
        message.assign(buffer, length);
        while (!message.empty() && (message.back() == '\r' || message.back() == '\n')) {
            message.pop_back();
        }
    } else {
        message = "unknown error";
    }
    if (buffer != nullptr) {
        LocalFree(buffer);
    }

    std::ostringstream oss;
    oss << message << " (" << error_code << ")";
    return oss.str();
}

void normalize_and_validate_options(options & opts) {
    opts.input_dir = fs::absolute(opts.input_dir).lexically_normal();
    opts.output = fs::absolute(opts.output).lexically_normal();
    opts.dxc = fs::absolute(opts.dxc).lexically_normal();
    if (opts.depfile_set) {
        opts.depfile = fs::absolute(opts.depfile).lexically_normal();
    }

    std::error_code ec;
    if (!fs::exists(opts.input_dir, ec) || !fs::is_directory(opts.input_dir, ec)) {
        throw std::runtime_error("--input-dir is not a directory: " + opts.input_dir.string());
    }
    if (!fs::exists(opts.dxc, ec) || fs::is_directory(opts.dxc, ec)) {
        throw std::runtime_error("--dxc does not name an executable file: " + opts.dxc.string());
    }
    if (!opts.output.has_filename()) {
        throw std::runtime_error("--output must name a header file: " + opts.output.string());
    }
    if (opts.target.empty()) {
        throw std::runtime_error("--target must not be empty");
    }

    const fs::path output_parent = opts.output.parent_path();
    if (!output_parent.empty()) {
        fs::create_directories(output_parent);
    }
    if (opts.depfile_set) {
        const fs::path depfile_parent = opts.depfile.parent_path();
        if (!depfile_parent.empty()) {
            fs::create_directories(depfile_parent);
        }
    }
}

std::string to_lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

bool has_hlsl_extension(const fs::path & path) {
    return to_lower_ascii(path.extension().string()) == ".hlsl";
}

bool uses_native_fp16_types(const shader_source & source) {
    const std::string filename = to_lower_ascii(source.path.filename().string());
    const std::string suffix = "_fp16.hlsl";
    return filename.size() >= suffix.size() && filename.compare(filename.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool parse_compute_target(const std::string & target, int & major, int & minor) {
    if (target.size() < 6 || target[0] != 'c' || target[1] != 's' || target[2] != '_') {
        return false;
    }
    const std::size_t sep = target.find('_', 3);
    if (sep == std::string::npos || sep == 3 || sep + 1 >= target.size()) {
        return false;
    }
    for (std::size_t i = 3; i < sep; ++i) {
        if (std::isdigit(static_cast<unsigned char>(target[i])) == 0) return false;
    }
    for (std::size_t i = sep + 1; i < target.size(); ++i) {
        if (std::isdigit(static_cast<unsigned char>(target[i])) == 0) return false;
    }
    major = std::stoi(target.substr(3, sep - 3));
    minor = std::stoi(target.substr(sep + 1));
    return true;
}

std::string target_for_source(const shader_source & source, const options & opts) {
    if (!uses_native_fp16_types(source)) {
        return opts.target;
    }

    int major = 0;
    int minor = 0;
    if (parse_compute_target(opts.target, major, minor) && (major < 6 || (major == 6 && minor < 2))) {
        return "cs_6_2";
    }
    return opts.target;
}

std::string logical_name_from_relative_path(fs::path relative_path) {
    relative_path.replace_extension();
    const std::string raw = relative_path.generic_string();

    std::string result;
    result.reserve(raw.size());
    for (unsigned char ch : raw) {
        if (std::isalnum(ch) != 0 || ch == '_') {
            result.push_back(static_cast<char>(ch));
        } else {
            result.push_back('_');
        }
    }

    if (result.empty()) {
        result = "shader";
    }
    return result;
}

std::string symbol_name_from_logical_name(const std::string & logical_name) {
    std::string symbol = logical_name.empty() ? std::string("shader") : logical_name;
    const unsigned char first = static_cast<unsigned char>(symbol[0]);
    if (!(std::isalpha(first) != 0 || symbol[0] == '_')) {
        symbol = "shader_" + symbol;
    }
    return symbol;
}

std::vector<shader_source> collect_sources(const fs::path & input_dir) {
    std::vector<shader_source> sources;

    for (const fs::directory_entry & entry : fs::recursive_directory_iterator(input_dir)) {
        if (!entry.is_regular_file() || !has_hlsl_extension(entry.path())) {
            continue;
        }

        const fs::path absolute_path = fs::absolute(entry.path()).lexically_normal();
        const fs::path relative_path = fs::relative(absolute_path, input_dir);
        const std::string logical_name = logical_name_from_relative_path(relative_path);
        sources.push_back({ absolute_path, logical_name, symbol_name_from_logical_name(logical_name) });
    }

    std::sort(sources.begin(), sources.end(), [](const shader_source & lhs, const shader_source & rhs) {
        if (lhs.logical_name != rhs.logical_name) {
            return lhs.logical_name < rhs.logical_name;
        }
        return lhs.path.string() < rhs.path.string();
    });

    for (std::size_t i = 1; i < sources.size(); ++i) {
        if (sources[i - 1].logical_name == sources[i].logical_name) {
            throw std::runtime_error(
                "multiple HLSL files map to shader name " + sources[i].logical_name + ": " +
                sources[i - 1].path.string() + " and " + sources[i].path.string());
        }
    }

    std::vector<std::string> symbols;
    symbols.reserve(sources.size());
    for (const shader_source & source : sources) {
        symbols.push_back(source.symbol_name);
    }
    std::sort(symbols.begin(), symbols.end());
    for (std::size_t i = 1; i < symbols.size(); ++i) {
        if (symbols[i - 1] == symbols[i]) {
            throw std::runtime_error("multiple shader names map to C++ symbol " + symbols[i]);
        }
    }

    return sources;
}

std::string quote_arg(const std::string & arg) {
    if (arg.empty()) {
        return "\"\"";
    }

    bool needs_quotes = false;
    for (unsigned char ch : arg) {
        if (std::isspace(ch) != 0 || ch == '"') {
            needs_quotes = true;
            break;
        }
    }
    if (!needs_quotes) {
        return arg;
    }

    std::string result;
    result.push_back('"');
    std::size_t backslashes = 0;
    for (char ch : arg) {
        if (ch == '\\') {
            ++backslashes;
        } else if (ch == '"') {
            result.append(backslashes * 2U + 1U, '\\');
            result.push_back('"');
            backslashes = 0;
        } else {
            result.append(backslashes, '\\');
            result.push_back(ch);
            backslashes = 0;
        }
    }
    result.append(backslashes * 2U, '\\');
    result.push_back('"');
    return result;
}

process_result run_process_capture(const std::string & command_line) {
    SECURITY_ATTRIBUTES security_attributes{};
    security_attributes.nLength = sizeof(security_attributes);
    security_attributes.bInheritHandle = TRUE;

    unique_handle read_pipe;
    unique_handle write_pipe;
    if (CreatePipe(read_pipe.put(), write_pipe.put(), &security_attributes, 0) == 0) {
        throw std::runtime_error("CreatePipe failed: " + format_windows_error(GetLastError()));
    }
    if (SetHandleInformation(read_pipe.get(), HANDLE_FLAG_INHERIT, 0) == 0) {
        throw std::runtime_error("SetHandleInformation failed: " + format_windows_error(GetLastError()));
    }

    STARTUPINFOA startup_info{};
    startup_info.cb = sizeof(startup_info);
    startup_info.dwFlags = STARTF_USESTDHANDLES;
    startup_info.hStdOutput = write_pipe.get();
    startup_info.hStdError = write_pipe.get();
    startup_info.hStdInput = nullptr;

    PROCESS_INFORMATION process_info{};
    std::vector<char> mutable_command(command_line.begin(), command_line.end());
    mutable_command.push_back('\0');

    if (CreateProcessA(
            nullptr,
            mutable_command.data(),
            nullptr,
            nullptr,
            TRUE,
            CREATE_NO_WINDOW,
            nullptr,
            nullptr,
            &startup_info,
            &process_info) == 0) {
        throw std::runtime_error("CreateProcessA failed: " + format_windows_error(GetLastError()) + "\ncommand: " + command_line);
    }

    unique_handle process(process_info.hProcess);
    unique_handle thread(process_info.hThread);
    thread.reset();
    write_pipe.reset();

    std::string output;
    char buffer[4096];
    for (;;) {
        DWORD bytes_read = 0;
        const BOOL read_ok = ReadFile(read_pipe.get(), buffer, static_cast<DWORD>(sizeof(buffer)), &bytes_read, nullptr);
        if (bytes_read != 0) {
            output.append(buffer, buffer + bytes_read);
        }
        if (read_ok == 0) {
            const DWORD read_error = GetLastError();
            if (read_error == ERROR_BROKEN_PIPE) {
                break;
            }
            throw std::runtime_error("ReadFile failed: " + format_windows_error(read_error));
        }
        if (bytes_read == 0) {
            break;
        }
    }

    if (WaitForSingleObject(process.get(), INFINITE) == WAIT_FAILED) {
        throw std::runtime_error("WaitForSingleObject failed: " + format_windows_error(GetLastError()));
    }

    DWORD exit_code = 0;
    if (GetExitCodeProcess(process.get(), &exit_code) == 0) {
        throw std::runtime_error("GetExitCodeProcess failed: " + format_windows_error(GetLastError()));
    }

    return { exit_code, output };
}

std::vector<std::uint8_t> read_binary_file(const fs::path & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open binary file: " + path.string());
    }

    std::vector<std::uint8_t> data;
    char buffer[4096];
    while (input) {
        input.read(buffer, static_cast<std::streamsize>(sizeof(buffer)));
        const std::streamsize bytes_read = input.gcount();
        for (std::streamsize i = 0; i < bytes_read; ++i) {
            data.push_back(static_cast<std::uint8_t>(static_cast<unsigned char>(buffer[static_cast<std::size_t>(i)])));
        }
    }
    if (input.bad()) {
        throw std::runtime_error("failed while reading binary file: " + path.string());
    }
    return data;
}

compiled_shader compile_shader(
    const shader_source & source,
    const options &       opts,
    const fs::path &      temp_dir,
    std::size_t           index) {
    std::ostringstream temp_name;
    temp_name << "d3d12-shader-" << GetCurrentProcessId() << '-' << index << '-' << source.symbol_name << ".dxil";
    const fs::path dxil_path = temp_dir / temp_name.str();

    std::error_code ignored_ec;
    fs::remove(dxil_path, ignored_ec);

    try {
        const bool native_fp16 = uses_native_fp16_types(source);
        std::string command_line = quote_arg(opts.dxc.string());
        command_line += " -T " + quote_arg(target_for_source(source, opts));
        command_line += " -E main";
        command_line += " -Fo " + quote_arg(dxil_path.string());
        command_line += " -nologo";
        if (native_fp16) {
            command_line += " -enable-16bit-types";
        }
        command_line += opts.debug ? " -Zi -Qembed_debug -Od" : " -O3";
        command_line += " " + quote_arg(source.path.string());

        const process_result result = run_process_capture(command_line);
        if (result.exit_code != 0) {
            std::string compiler_output = result.output.empty() ? std::string("<no compiler output>\n") : result.output;
            if (!compiler_output.empty() && compiler_output.back() != '\n') {
                compiler_output.push_back('\n');
            }
            std::ostringstream message;
            message << source.path.string() << ":\n" << compiler_output << "dxc exited with code " << result.exit_code;
            throw std::runtime_error(message.str());
        }

        std::vector<std::uint8_t> data = read_binary_file(dxil_path);
        if (data.empty()) {
            throw std::runtime_error("dxc produced an empty DXIL blob for " + source.path.string());
        }

        fs::remove(dxil_path, ignored_ec);
        if (ignored_ec) {
            throw std::runtime_error("failed to delete temporary DXIL file " + dxil_path.string() + ": " + ignored_ec.message());
        }

        return { source.logical_name, source.symbol_name, std::move(data) };
    } catch (...) {
        fs::remove(dxil_path, ignored_ec);
        throw;
    }
}

void write_shader_array(std::ostream & output, const compiled_shader & shader) {
    output << "inline constexpr unsigned char " << shader.symbol_name << "_data[] = {";
    for (std::size_t i = 0; i < shader.data.size(); ++i) {
        if (i % 16U == 0U) {
            output << "\n    ";
        }
        output << "0x" << std::hex << std::nouppercase << std::setw(2) << std::setfill('0')
               << static_cast<unsigned int>(shader.data[i]) << std::dec << std::setfill(' ');
        if (i + 1U != shader.data.size()) {
            output << ", ";
        }
    }
    output << "\n};\n";
    output << "inline constexpr size_t " << shader.symbol_name << "_size = sizeof(" << shader.symbol_name << "_data);\n\n";
}

std::string temp_header_path_string(const fs::path & output) {
    std::ostringstream suffix;
    suffix << ".tmp." << GetCurrentProcessId();
    fs::path temp = output;
    temp += suffix.str();
    return temp.string();
}

void replace_file_with_temp(const fs::path & temp, const fs::path & output) {
    if (MoveFileExA(
            temp.string().c_str(),
            output.string().c_str(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) == 0) {
        std::error_code ignored_ec;
        fs::remove(temp, ignored_ec);
        throw std::runtime_error("failed to replace " + output.string() + ": " + format_windows_error(GetLastError()));
    }
}

void write_header(const fs::path & output_path, const std::vector<compiled_shader> & shaders) {
    const fs::path temp_path(temp_header_path_string(output_path));

    {
        std::ofstream output(temp_path, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw std::runtime_error("failed to open temporary output header: " + temp_path.string());
        }

        output << "// Auto-generated by d3d12-shaders-gen. Do not edit.\n";
        output << "#pragma once\n";
        output << "#include <cstddef>\n";
        output << "#include <cstdint>\n\n";
        output << "namespace ggml_d3d12_shaders {\n\n";

        for (const compiled_shader & shader : shaders) {
            write_shader_array(output, shader);
        }

        output << "struct shader_entry {\n";
        output << "    const char *    name;\n";
        output << "    const uint8_t * data;\n";
        output << "    size_t          size;\n";
        output << "};\n\n";

        if (shaders.empty()) {
            output << "inline constexpr shader_entry shader_table[1] = {};\n";
            output << "inline constexpr size_t shader_table_count = 0;\n\n";
        } else {
            output << "inline constexpr shader_entry shader_table[] = {\n";
            for (const compiled_shader & shader : shaders) {
                output << "    { \"" << shader.logical_name << "\", " << shader.symbol_name << "_data, " << shader.symbol_name << "_size },\n";
            }
            output << "};\n";
            output << "inline constexpr size_t shader_table_count = sizeof(shader_table) / sizeof(shader_table[0]);\n\n";
        }

        output << "} // namespace ggml_d3d12_shaders\n";
        output.close();
        if (!output) {
            throw std::runtime_error("failed while writing temporary output header: " + temp_path.string());
        }
    }

    replace_file_with_temp(temp_path, output_path);
}

std::string escape_depfile_path(const fs::path & path) {
    const std::string raw = path.lexically_normal().generic_string();
    std::string escaped;
    escaped.reserve(raw.size());
    for (char ch : raw) {
        if (ch == ' ' || ch == '#' || ch == '$' || ch == ':' || ch == '\\') {
            escaped.push_back('\\');
        }
        escaped.push_back(ch);
    }
    return escaped;
}

void write_depfile(const fs::path & depfile_path, const fs::path & output_path, const std::vector<shader_source> & sources) {
    std::ofstream depfile(depfile_path, std::ios::binary | std::ios::trunc);
    if (!depfile) {
        throw std::runtime_error("failed to open depfile: " + depfile_path.string());
    }

    depfile << escape_depfile_path(output_path) << ":";
    if (sources.empty()) {
        depfile << "\n";
    } else {
        depfile << " \\\n";
        for (std::size_t i = 0; i < sources.size(); ++i) {
            depfile << "  " << escape_depfile_path(sources[i].path);
            if (i + 1U != sources.size()) {
                depfile << " \\\n";
            } else {
                depfile << "\n";
            }
        }
    }

    depfile.close();
    if (!depfile) {
        throw std::runtime_error("failed while writing depfile: " + depfile_path.string());
    }
}

} // namespace

int main(int argc, char ** argv) {
    try {
        options opts;
        std::string parse_error;
        if (!parse_args(argc, argv, opts, parse_error)) {
            std::cerr << "error: " << parse_error << '\n' << usage() << '\n';
            return 2;
        }

        normalize_and_validate_options(opts);

        const std::vector<shader_source> sources = collect_sources(opts.input_dir);
        if (sources.empty()) {
            std::cerr << "d3d12-shaders-gen: no HLSL files found in " << opts.input_dir.string() << '\n';
        }

        std::vector<compiled_shader> shaders;
        shaders.reserve(sources.size());
        for (std::size_t i = 0; i < sources.size(); ++i) {
            shaders.push_back(compile_shader(sources[i], opts, opts.output.parent_path(), i));
        }

        write_header(opts.output, shaders);
        if (opts.depfile_set) {
            write_depfile(opts.depfile, opts.output, sources);
        }

        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "error: " << ex.what() << '\n';
        return 1;
    }
}
