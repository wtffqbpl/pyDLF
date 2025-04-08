#pragma once

#include <iostream>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <memory>
#include <string>

namespace dlf {

class Logger {
public:
  static Logger& getInstance() {
    static Logger instance;
    return instance;
  }

  void initialize(const std::string& log_file = "dlf.log") {
    try {
      // Create console sink
      auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
      console_sink->set_level(spdlog::level::info);
      console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%s:%#] %v");

      // Create file sink
      auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_file, true);
      file_sink->set_level(spdlog::level::debug);
      file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%s:%#] %v");

      // Create logger with both sinks
      logger_ = std::make_shared<spdlog::logger>("dlf", spdlog::sinks_init_list{console_sink, file_sink});
      logger_->set_level(spdlog::level::debug);
      logger_->flush_on(spdlog::level::err);
      logger_->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%s:%#] %v");
      logger_->enable_backtrace(32);  // Enable backtrace for source location

      // Register as default logger
      spdlog::set_default_logger(logger_);
    } catch (const spdlog::spdlog_ex& ex) {
      std::cerr << "Log initialization failed: " << ex.what() << std::endl;
    }
  }

  void flush() {
    if (logger_)
      logger_->flush();
  }

  [[nodiscard]] std::shared_ptr<spdlog::logger> getLogger() const {
    return logger_;
  }

  Logger(const Logger&) = delete;
  Logger& operator=(const Logger&) = delete;

private:
  Logger() = default;
  ~Logger() = default;
  std::shared_ptr<spdlog::logger> logger_;
};

// Convenience macros for logging
#define DLF_LOG_TRACE(...)    SPDLOG_LOGGER_TRACE(dlf::Logger::getInstance().getLogger(), __VA_ARGS__)
#define DLF_LOG_DEBUG(...)    SPDLOG_LOGGER_DEBUG(dlf::Logger::getInstance().getLogger(), __VA_ARGS__)
#define DLF_LOG_INFO(...)     SPDLOG_LOGGER_INFO(dlf::Logger::getInstance().getLogger(), __VA_ARGS__)
#define DLF_LOG_WARN(...)     SPDLOG_LOGGER_WARN(dlf::Logger::getInstance().getLogger(), __VA_ARGS__)
#define DLF_LOG_ERROR(...)    SPDLOG_LOGGER_ERROR(dlf::Logger::getInstance().getLogger(), __VA_ARGS__)
#define DLF_LOG_CRITICAL(...) SPDLOG_LOGGER_CRITICAL(dlf::Logger::getInstance().getLogger(), __VA_ARGS__)

} // namespace dlf 