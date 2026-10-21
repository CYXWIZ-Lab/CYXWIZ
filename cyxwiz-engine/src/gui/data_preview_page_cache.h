#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace gui::data_input {

// Bounded UI cache for virtual tabular preview rows. Pages are addressed by
// their dataset row offset and evicted by least-recent use. This keeps preview
// memory independent of the source row count.
class PreviewPageCache {
public:
    using Row = std::vector<std::string>;
    using Rows = std::vector<Row>;

    explicit PreviewPageCache(int64_t page_size = 100,
                              std::size_t max_pages = 5)
        : page_size_(std::max<int64_t>(1, page_size)),
          max_pages_(std::max<std::size_t>(1, max_pages)) {}

    void Clear() {
        pages_.clear();
        access_clock_ = 0;
    }

    int64_t PageSize() const { return page_size_; }

    int64_t AlignOffset(int64_t row_index) const {
        const int64_t safe_index = std::max<int64_t>(0, row_index);
        return (safe_index / page_size_) * page_size_;
    }

    bool ContainsPage(int64_t offset) const {
        return pages_.find(AlignOffset(offset)) != pages_.end();
    }

    void PutPage(int64_t offset, Rows rows) {
        const int64_t aligned = AlignOffset(offset);
        pages_[aligned] = Page{aligned, std::move(rows), ++access_clock_};
        while (pages_.size() > max_pages_) {
            auto oldest = pages_.end();
            uint64_t oldest_access = std::numeric_limits<uint64_t>::max();
            for (auto it = pages_.begin(); it != pages_.end(); ++it) {
                if (it->second.last_access < oldest_access) {
                    oldest = it;
                    oldest_access = it->second.last_access;
                }
            }
            if (oldest == pages_.end()) break;
            pages_.erase(oldest);
        }
    }

    const Row* FindRow(int64_t row_index) {
        if (row_index < 0 || pages_.empty()) return nullptr;
        auto it = pages_.upper_bound(row_index);
        if (it == pages_.begin()) return nullptr;
        --it;
        const int64_t local_index = row_index - it->second.offset;
        if (local_index < 0 ||
            local_index >= static_cast<int64_t>(it->second.rows.size())) {
            return nullptr;
        }
        it->second.last_access = ++access_clock_;
        return &it->second.rows[static_cast<std::size_t>(local_index)];
    }

    std::size_t PageCount() const { return pages_.size(); }

    std::size_t RowCount() const {
        std::size_t count = 0;
        for (const auto& [_, page] : pages_) count += page.rows.size();
        return count;
    }

private:
    struct Page {
        int64_t offset = 0;
        Rows rows;
        uint64_t last_access = 0;
    };

    int64_t page_size_ = 100;
    std::size_t max_pages_ = 5;
    uint64_t access_clock_ = 0;
    std::map<int64_t, Page> pages_;
};

} // namespace gui::data_input
