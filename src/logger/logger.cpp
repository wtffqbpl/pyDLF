#include "logger/logger.h"
#include <spdlog/sinks/stdout_color_sinks.h>

namespace dlf
{
std::shared_ptr<spdlog::logger> Logger::logger_;

void Logger::init(const std::string& logger_name)
{
    if (!logger_)
    {
        logger_ = spdlog::stdout_color_mt(logger_name);
        logger_->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%n] %v");
        logger_->set_level(spdlog::level::info);
    }
}

void Logger::set_level(spdlog::level::level_enum level)
{
    if (!logger_)
    {
        init();
    }
    logger_->set_level(level);
}

// Template method instantiations
template void Logger::trace<const char*>(const char*, const char*&&);
template void Logger::debug<const char*>(const char*, const char*&&);
template void Logger::info<const char*>(const char*, const char*&&);
template void Logger::warn<const char*>(const char*, const char*&&);
template void Logger::error<const char*>(const char*, const char*&&);
template void Logger::critical<const char*>(const char*, const char*&&);

}  // namespace dlf