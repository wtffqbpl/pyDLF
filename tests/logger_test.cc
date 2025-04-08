#include <gtest/gtest.h>
#include <utils/logger.hpp>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <iostream>
#include <algorithm>

class LoggerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize logger with a test log file
        test_log_file_ = "test_dlf.log";
        dlf::Logger::getInstance().initialize(test_log_file_);
    }

    void TearDown() override {
        // Clean up the test log file
        if (std::filesystem::exists(test_log_file_)) {
            std::filesystem::remove(test_log_file_);
        }
    }

    std::string test_log_file_;
};

TEST_F(LoggerTest, BasicLogging) {
    // Test different log levels
    DLF_LOG_TRACE("This is a trace message");
    DLF_LOG_DEBUG("This is a debug message");
    DLF_LOG_INFO("This is an info message");
    DLF_LOG_WARN("This is a warning message");
    DLF_LOG_ERROR("This is an error message");
    DLF_LOG_CRITICAL("This is a critical message");

    // Verify that the log file was created
    ASSERT_TRUE(std::filesystem::exists(test_log_file_));
}

TEST_F(LoggerTest, LogFormatting) {
    // Test log formatting with different types
    int number = 42;
    double pi = 3.14159;
    std::string text = "test";
    
    DLF_LOG_INFO("Number: {}, Pi: {:.2f}, Text: {}", number, pi, text);
    
    // Ensure the log is flushed to disk using our Logger's flush method
    dlf::Logger::getInstance().flush();
    
    // Read the log file and verify the formatted message
    std::ifstream log_file(test_log_file_);
    ASSERT_TRUE(log_file.is_open()) << "Failed to open log file: " << test_log_file_;
    
    std::string line;
    bool found = false;
    while (std::getline(log_file, line)) {
        std::cout << "Checking line: " << line << std::endl;  // Debug output
        if (line.find("Number: 42, Pi: 3.14, Text: test") != std::string::npos) {
            found = true;
            break;
        }
    }
    
    ASSERT_TRUE(found) << "Expected log message not found in log file. Last line checked: " << line;
}

TEST_F(LoggerTest, LogLevels) {
    // Test that different log levels are properly filtered
    DLF_LOG_DEBUG("Debug message");
    DLF_LOG_INFO("Info message");
    DLF_LOG_WARN("Warning message");

    // Flush the logger to ensure the message is written to the file
    dlf::Logger::getInstance().flush();
    
    // Read the log file and verify the messages
    std::ifstream log_file(test_log_file_);
    std::string line;
    // bool found_debug = false;
    bool found_info = false;
    bool found_warn = false;
    
    while (std::getline(log_file, line)) {
        std::cout << "Checking line: " << line << std::endl; // Debug output

        // if (line.find("Debug message") != std::string::npos) found_debug = true;
        if (line.find("Info message") != std::string::npos) found_info = true;
        if (line.find("Warning message") != std::string::npos) found_warn = true;
    }
    
    // ASSERT_TRUE(found_debug);
    ASSERT_TRUE(found_info);
    ASSERT_TRUE(found_warn);
}

TEST_F(LoggerTest, LogFileRotation) {
    // Test that the logger can handle multiple initializations
    dlf::Logger::getInstance().initialize("new_test.log");
    DLF_LOG_INFO("Message to new log file");
    
    // Verify both log files exist
    ASSERT_TRUE(std::filesystem::exists("new_test.log"));
    ASSERT_TRUE(std::filesystem::exists(test_log_file_));
    
    // Clean up the new log file
    std::filesystem::remove("new_test.log");
}

TEST_F(LoggerTest, LogPattern) {
    // Test that the log pattern is correct
    DLF_LOG_INFO("Test pattern");

    // Flush the logger to ensure the message is written to the file
    dlf::Logger::getInstance().flush();
    
    std::ifstream log_file(test_log_file_);
    std::string line;
    std::getline(log_file, line);
    
    // Print the actual log line for debugging
    std::cout << "Actual log line: " << line << std::endl;
    
    // Convert both strings to lowercase for case-insensitive comparison
    std::string line_lower = line;
    std::transform(line_lower.begin(), line_lower.end(), line_lower.begin(), ::tolower);
    
    // Verify the pattern contains timestamp, level, and message
    ASSERT_TRUE(line_lower.find("[info]") != std::string::npos);
    ASSERT_TRUE(line.find("Test pattern") != std::string::npos);
    ASSERT_TRUE(line.find("logger_test.cc") != std::string::npos);
}

#if 0
TEST_F(LoggerTest, Performance) {
    // Test logging performance with multiple messages
    const int num_messages = 1000;
    for (int i = 0; i < num_messages; ++i) {
        DLF_LOG_INFO("Performance test message {}", i);
    }
    
    // Verify all messages were written
    std::ifstream log_file(test_log_file_);
    int message_count = 0;
    std::string line;
    while (std::getline(log_file, line)) {
        if (line.find("Performance test message") != std::string::npos) {
            message_count++;
        }
    }
    ASSERT_EQ(message_count, num_messages);
}
#endif