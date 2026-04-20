#include "format_utils.h"
#include <vector>
#include <string>
#include <iomanip>
#include <sstream>

namespace neurons::cli::utils {

std::string formatFileSize(int64_t bytes) {
    const std::vector<std::string> units = {"B", "KB", "MB", "GB", "TB"};
    int unitIndex = 0;
    double size = static_cast<double>(bytes);

    while (size >= 1024.0 && unitIndex < static_cast<int>(units.size()) - 1) {
        size /= 1024.0;
        unitIndex++;
    }

    std::ostringstream oss;
    if (unitIndex == 0) {
        oss << bytes << " " << units[unitIndex];
    } else {
        oss << std::fixed << std::setprecision(1) << size << " " << units[unitIndex];
    }

    return oss.str();
}

} // namespace neurons::cli::utils