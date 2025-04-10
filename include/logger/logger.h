#pragma once

#include <memory>
#include <string>
#include <spdlog/spdlog.h>

namespace dlf {

class Logger {
public:
    static void init(const std::string& logger_name = "dlf");
    static void set_level(spdlog::level::level_enum level);
    
    template<typename... Args>
    static void trace(const char* fmt, Args&&... args) {
        logger_->trace(fmt, std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    static void debug(const char* fmt, Args&&... args) {
        logger_->debug(fmt, std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    static void info(const char* fmt, Args&&... args) {
        logger_->info(fmt, std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    static void warn(const char* fmt, Args&&... args) {
        logger_->warn(fmt, std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    static void error(const char* fmt, Args&&... args) {
        logger_->error(fmt, std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    static void critical(const char* fmt, Args&&... args) {
        logger_->critical(fmt, std::forward<Args>(args)...);
    }

private:
    static std::shared_ptr<spdlog::logger> logger_;
};

} // namespace dlf 