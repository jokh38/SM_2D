#include <gtest/gtest.h>
#include "utils/logger.hpp"

TEST(LoggerTest, InfoLevelWorks) {
    Logger& log = Logger::get();
    log.set_level(LogLevel::INFO);
    log.info("Test info message: {}", 42);
}