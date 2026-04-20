#pragma once

#include <string>
#include <cstdint>

namespace neurons::cli::utils {

/**
 * Format a file size in bytes to a human-readable string
 * @param bytes Size in bytes
 * @return Formatted string (e.g., "1.2 MB", "345 KB")
 */
std::string formatFileSize(int64_t bytes);

} // namespace neurons::cli::utils