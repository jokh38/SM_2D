#pragma once
#include <string>
#include <memory>
#include <cstdio>
#include <cstdarg>

enum class LogLevel { TRACE = 0, DEBUG = 1, INFO = 2, WARN = 3, ERROR = 4 };

class Logger {
public:
    static Logger& get();
    void set_level(LogLevel level);
    LogLevel get_level() const;
    void info(const char* fmt, ...);
    void error(const char* fmt, ...);
    void flush();
private:
    Logger();
    ~Logger() = default;
    void log(LogLevel level, const char* fmt, va_list args);
    LogLevel level_ = LogLevel::INFO;
    bool colors_ = true;
};