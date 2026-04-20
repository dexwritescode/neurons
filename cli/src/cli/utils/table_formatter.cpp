#include "table_formatter.h"
#include <iostream>

namespace neurons::cli {

TableFormatter::TableFormatter() : row_count_(0) {
    setup_table_style();
}

void TableFormatter::set_headers(const std::vector<std::string>& headers) {
    headers_ = headers;

    // Convert vector to initializer list compatible format
    tabulate::Table::Row_t header_row;
    for (const auto& header : headers) {
        header_row.push_back(header);
    }
    table_.add_row(header_row);
    row_count_ = 1;
}

void TableFormatter::add_row(const std::vector<std::string>& row) {
    // Convert vector to initializer list compatible format
    tabulate::Table::Row_t table_row;
    for (const auto& cell : row) {
        table_row.push_back(cell);
    }
    table_.add_row(table_row);
    row_count_++;
}

void TableFormatter::add_model_row(const std::string& name, const std::string& size,
                                 const std::string& status, const std::string& path) {
    if (path.empty()) {
        add_row({name, size, status});
    } else {
        add_row({name, size, status, path});
    }
}

std::string TableFormatter::render() const {
    // Cast away const to call non-const str() method
    return const_cast<tabulate::Table&>(table_).str();
}

void TableFormatter::print() const {
    std::cout << render() << std::endl;
}

void TableFormatter::setup_table_style() {
    table_.format()
        .font_style({tabulate::FontStyle::bold})
        .border_top(" ")
        .border_bottom(" ")
        .border_left(" ")
        .border_right(" ")
        .corner(" ");

    // Style will be applied to header row when it's added
}

}