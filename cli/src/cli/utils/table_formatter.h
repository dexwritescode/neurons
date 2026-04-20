#pragma once

#include <tabulate/table.hpp>
#include <vector>
#include <string>

namespace neurons::cli {

class TableFormatter {
public:
    TableFormatter();
    ~TableFormatter() = default;

    void set_headers(const std::vector<std::string>& headers);
    void add_row(const std::vector<std::string>& row);
    void add_model_row(const std::string& name, const std::string& size,
                      const std::string& status, const std::string& path = "");

    std::string render() const;
    void print() const;

private:
    tabulate::Table table_;
    std::vector<std::string> headers_;
    size_t row_count_;

    void setup_table_style();
};

}