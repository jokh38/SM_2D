#include "utils/logger.hpp"
#include <iostream>
#include <chrono>
#include <cstdio>
#include <cstdarg>

namespace {
    const char* level_strings[] = {"TRACE", "DEBUG", "INFO", "WARN", "ERROR"};
    const char* color_codes[] = {"\033[0;90m", "\033[0;36m", "\033[0;32m", "\033[0;33m", "\033[0;31m"};
    const char* reset_code = "\033[0m";
}

Logger::Logger() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
}

Logger& Logger::get() {
    static Logger instance;
    return instance;
}

void Logger::set_level(LogLevel level) {
    level_ = level;
}

LogLevel Logger::get_level() const {
    return level_;
}

void Logger::info(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log(LogLevel::INFO, fmt, args);
    va_end(args);
}

void Logger::error(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log(LogLevel::ERROR, fmt, args);
    va_end(args);
}

void Logger::log(LogLevel level, const char* fmt, va_list args) {
    if (level_ > level) return;

    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);

    std::printf("[%02d:%02d:%02d] ",
                std::localtime(&time_t)->tm_hour,
                std::localtime(&time_t)->tm_min,
                std::localtime(&time_t)->tm_sec);

    if (colors_) {
        std::printf("%s%s%s: ", color_codes[static_cast<int>(level)],
                   level_strings[static_cast<int>(level)], reset_code);
    } else {
        std::printf("%s: ", level_strings[static_cast<int>(level)]);
    }

    std::vprintf(fmt, args);
    std::printf("\n");
}

void Logger::flush() {
    std::fflush(stdout);
}