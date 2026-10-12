#include "../src/utils/hdf5_browser.h"

#ifdef CYXWIZ_HAS_HDF5
#include <highfive/highfive.hpp>
#endif

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

using cyxwiz::HDF5Browser;
using cyxwiz::Hdf5BrowsePage;
using cyxwiz::Hdf5BrowseRequest;
using cyxwiz::Hdf5ObjectKind;
using cyxwiz::Hdf5TableStatus;
int checks = 0;

void Check(bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

void CheckFailure(const Hdf5BrowsePage& page, Hdf5TableStatus status,
                  const std::string& context) {
    Check(page.status == status, context + ": unexpected status: " + page.error);
    Check(!page.error.empty(), context + ": error explains failure");
    Check(page.entries.empty(), context + ": no partial entries on failure");
    Check(!page.has_next && page.next_offset == page.offset,
          context + ": failed page does not advance pagination");
}

#ifdef CYXWIZ_HAS_HDF5
struct Workspace {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("cyxwiz_hdf5_browser_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    Workspace() { Check(std::filesystem::create_directory(path), "unique workspace"); }
    ~Workspace() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};

void CheckPage(const Hdf5BrowsePage& page, const Hdf5BrowseRequest& request,
               uint64_t total, size_t count) {
    Check(page.status == Hdf5TableStatus::Ok && page.error.empty(), "browse succeeds: " + page.error);
    Check(page.group_path == request.group_path && page.offset == request.offset, "page identifies its request");
    Check(page.total_entries == total && page.entries.size() == count, "page counts match group");
    Check(page.next_offset == request.offset + count && page.has_next == (page.next_offset < total),
          "pagination advances only by returned entries");
}

const cyxwiz::Hdf5BrowseEntry& Find(const Hdf5BrowsePage& page, const std::string& name) {
    const auto it = std::find_if(page.entries.begin(), page.entries.end(),
                                 [&](const auto& entry) { return entry.name == name; });
    Check(it != page.entries.end(), "entry should be listed: " + name);
    return *it;
}

void CreateFixture(const std::string& path, const std::string& missing_external) {
    HighFive::File file(path, HighFive::File::Overwrite);
    // Deliberately create entries out of name order. No dataset payload is
    // needed for numeric metadata, including the oversized full-load fixture.
    file.createDataSet<float>("/rank3", HighFive::DataSpace(std::vector<size_t>{2, 3, 4}));
    file.createDataSet<uint64_t>("/numeric", HighFive::DataSpace(std::vector<size_t>{3, 2}));
    const std::vector<std::string> strings{"one", "two"};
    file.createDataSet<std::string>("/strings", HighFive::DataSpace::From(strings)).write(strings);
    file.createDataSet<double>("/over_budget", HighFive::DataSpace(std::vector<size_t>{20000000}));
    file.createDataSet<int32_t>("/empty", HighFive::DataSpace(std::vector<size_t>{0}));
    file.createGroup("/group");
    file.createGroup("/group/subgroup");
    file.createDataSet<float>("/group/values", HighFive::DataSpace(std::vector<size_t>{4}));
    Check(H5Lcreate_hard(file.getId(), "/group", file.getId(), "/group/cycle",
                         H5P_DEFAULT, H5P_DEFAULT) >= 0, "create hard group cycle");
    Check(H5Lcreate_soft("/numeric", file.getId(), "/soft", H5P_DEFAULT, H5P_DEFAULT) >= 0,
          "create soft dataset link");
    Check(H5Lcreate_soft("/missing", file.getId(), "/dangling", H5P_DEFAULT, H5P_DEFAULT) >= 0,
          "create dangling soft link");
    Check(H5Lcreate_soft("/group", file.getId(), "/soft_group", H5P_DEFAULT, H5P_DEFAULT) >= 0,
          "create soft group link");
    Check(H5Lcreate_external(missing_external.c_str(), "/numeric", file.getId(), "/external",
                             H5P_DEFAULT, H5P_DEFAULT) >= 0, "create external link to absent file");
    const hid_t named_type = H5Tcopy(H5T_NATIVE_INT32);
    Check(named_type >= 0, "copy named datatype");
    const auto committed = H5Tcommit2(file.getId(), "/type", named_type,
                                     H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    const auto closed = H5Tclose(named_type);
    Check(committed >= 0 && closed >= 0, "create named datatype");
    auto pages = file.createGroup("/pages");
    for (int index = 128; index >= 0; --index) {
        const auto digits = std::to_string(index);
        pages.createGroup("entry_" + std::string(3 - digits.size(), '0') + digits);
    }
    file.createGroup("/long");
    file.createGroup("/long/" + std::string(4090, 'x'));
    file.createGroup("/names");
    file.createGroup("/names/a");
    file.createGroup("/names/" + std::string(4090, 'z'));
    file.createGroup("/oversized_names");
    file.createGroup("/oversized_names/" + std::string(4097, 'x'));
}
#endif

} // namespace

int main() try {
    Hdf5BrowseRequest request;
#ifdef CYXWIZ_HAS_HDF5
    Workspace workspace;
    const auto path = (workspace.path / "input.h5").string();
    const auto absent = (workspace.path / "absent.h5").string();
    CreateFixture(path, absent);
    const std::vector<std::string> expected{
        "dangling", "empty", "external", "group", "long", "names", "numeric",
        "over_budget", "oversized_names", "pages", "rank3", "soft", "soft_group", "strings", "type"};
    auto full = HDF5Browser::BrowsePage(path, request);
    CheckPage(full, request, expected.size(), expected.size());
    for (size_t index = 0; index < expected.size(); ++index) {
        Check(full.entries[index].name == expected[index] && full.entries[index].path == "/" + expected[index],
              "root entries are immediate children in stable name order");
    }
    const auto& numeric = Find(full, "numeric");
    Check(numeric.kind == Hdf5ObjectKind::Dataset && numeric.status == Hdf5TableStatus::Ok &&
              numeric.reason.empty(), "numeric dataset is compatible");
    Check(numeric.dataset.path == "/numeric" && numeric.dataset.shape == std::vector<uint64_t>({3, 2}) &&
              numeric.dataset.source_type == "uint64", "numeric metadata preserves full shape and native type");
    cyxwiz::Hdf5TableReadOptions probe_options;
    probe_options.selection.data_path = "/numeric";
    const auto probe = cyxwiz::ProbeHdf5Table(path, probe_options);
    Check(probe.status == Hdf5TableStatus::Ok && !probe.table && probe.estimated_materialized_bytes > 0 &&
              numeric.estimated_materialized_bytes == probe.estimated_materialized_bytes,
          "browse reuses the metadata-only full-load estimate");
    const auto& rank3 = Find(full, "rank3");
    Check(rank3.kind == Hdf5ObjectKind::Dataset && rank3.status == Hdf5TableStatus::UnsupportedRank &&
              !rank3.reason.empty() && rank3.dataset.shape == std::vector<uint64_t>({2, 3, 4}) &&
              rank3.dataset.source_type == "float", "unsupported rank retains shape and type");
    const auto& strings = Find(full, "strings");
    Check(strings.kind == Hdf5ObjectKind::Dataset && strings.status == Hdf5TableStatus::UnsupportedType &&
              !strings.reason.empty() && strings.dataset.shape == std::vector<uint64_t>({2}) &&
              strings.dataset.source_type == "string", "unsupported strings retain metadata");
    const auto& empty = Find(full, "empty");
    Check(empty.status == Hdf5TableStatus::EmptyDataset && empty.dataset.shape == std::vector<uint64_t>({0}),
          "empty dataset remains inspectable");
    const auto& large = Find(full, "over_budget");
    Check(large.status == Hdf5TableStatus::ResourceLimit && !large.reason.empty() &&
              large.dataset.shape == std::vector<uint64_t>({20000000}) &&
              large.estimated_materialized_bytes > 256ULL * 1024 * 1024,
          "full-load memory incompatibility is an entry result, not a page failure");
    for (const auto* name : {"soft", "dangling", "external", "soft_group"}) {
        const auto& entry = Find(full, name);
        Check(entry.kind == Hdf5ObjectKind::IndirectLink && entry.status == Hdf5TableStatus::UnsupportedLayout &&
                  !entry.reason.empty() && entry.dataset.path.empty() && entry.dataset.shape.empty() &&
                  entry.dataset.source_type.empty() && entry.estimated_materialized_bytes == 0,
              "indirect links are listed without target inspection");
    }
    const auto& other = Find(full, "type");
    Check(other.kind == Hdf5ObjectKind::Other && other.status == Hdf5TableStatus::UnsupportedLayout &&
              !other.reason.empty(), "named datatype is an unsupported object");

    request.limit = 4;
    std::vector<std::string> paged_names;
    do {
        const auto page = HDF5Browser::BrowsePage(path, request);
        CheckPage(page, request, expected.size(), std::min<size_t>(request.limit, expected.size() - request.offset));
        const auto repeated = HDF5Browser::BrowsePage(path, request);
        CheckPage(repeated, request, expected.size(), page.entries.size());
        for (size_t index = 0; index < page.entries.size(); ++index) {
            paged_names.push_back(page.entries[index].name);
            Check(repeated.entries[index].name == page.entries[index].name &&
                      repeated.entries[index].status == page.entries[index].status, "repeated page is stable");
        }
        request.offset = page.next_offset;
        if (!page.has_next) break;
    } while (true);
    Check(paged_names == expected, "pagination has no missing or duplicate links");
    CheckPage(HDF5Browser::BrowsePage(path, request), request, expected.size(), 0);
    ++request.offset;
    CheckFailure(HDF5Browser::BrowsePage(path, request), Hdf5TableStatus::InvalidSelection, "offset beyond end");

    request = {};
    request.group_path = "/group";
    const auto nested = HDF5Browser::BrowsePage(path, request);
    CheckPage(nested, request, 3, 3);
    Check(Find(nested, "values").path == "/group/values" &&
              Find(nested, "values").dataset.shape == std::vector<uint64_t>({4}), "nonroot dataset metadata");
    Check(Find(nested, "cycle").kind == Hdf5ObjectKind::Group &&
              Find(nested, "cycle").status == Hdf5TableStatus::Ok, "hard cycle is a browsable group");
    request.group_path = "/group/cycle";
    const auto cycle = HDF5Browser::BrowsePage(path, request);
    CheckPage(cycle, request, 3, 3);
    Check(Find(cycle, "cycle").path == "/group/cycle/cycle", "hard group cycle is not recursively expanded");
    request.group_path = "/group/subgroup";
    CheckPage(HDF5Browser::BrowsePage(path, request), request, 0, 0);

    request.group_path = "/pages";
    request.limit = 128;
    const auto maximum = HDF5Browser::BrowsePage(path, request);
    CheckPage(maximum, request, 129, 128);
    Check(maximum.entries.front().name == "entry_000" && maximum.entries.back().name == "entry_127",
          "maximum page is bounded and name ordered");
    request.offset = 128;
    const auto last = HDF5Browser::BrowsePage(path, request);
    CheckPage(last, request, 129, 1);
    Check(last.entries.front().name == "entry_128", "last partial page");
    request.offset = 129;
    CheckPage(HDF5Browser::BrowsePage(path, request), request, 129, 0);
    request.offset = std::numeric_limits<uint64_t>::max();
    CheckFailure(HDF5Browser::BrowsePage(path, request), Hdf5TableStatus::InvalidSelection, "huge offset");

    request = {};
    for (uint32_t limit : {uint32_t{0}, uint32_t{129}, std::numeric_limits<uint32_t>::max()}) {
        request.limit = limit;
        CheckFailure(HDF5Browser::BrowsePage(path, request), Hdf5TableStatus::InvalidSelection, "invalid limit");
    }
    request = {};
    for (const auto& invalid : std::vector<std::string>{"", "group", "/group/", "/group//subgroup", "/group/.",
             "/group/../numeric", "/missing/../group", std::string("/group") + '\0' + "extra",
             "/" + std::string(4096, 'x')}) {
        request.group_path = invalid;
        CheckFailure(HDF5Browser::BrowsePage(path, request), Hdf5TableStatus::InvalidSelection, "invalid group path");
    }
    for (const auto* indirect : {"/soft", "/external", "/soft_group", "/soft_group/subgroup"}) {
        request.group_path = indirect;
        CheckFailure(HDF5Browser::BrowsePage(path, request), Hdf5TableStatus::UnsupportedLayout, "indirect group path");
    }
    request.group_path = "/numeric";
    CheckFailure(HDF5Browser::BrowsePage(path, request), Hdf5TableStatus::InvalidSelection, "dataset is not a group");
    request.group_path = "/missing";
    CheckFailure(HDF5Browser::BrowsePage(path, request), Hdf5TableStatus::MissingDataset, "missing group");
    request = {};
    CheckFailure(HDF5Browser::BrowsePage(absent, request), Hdf5TableStatus::InvalidFile, "missing file");
    const auto fake = (workspace.path / "fake.h5").string();
    { std::ofstream out(fake, std::ios::binary); out << "not HDF5"; }
    CheckFailure(HDF5Browser::BrowsePage(fake, request), Hdf5TableStatus::InvalidFile, "fake signature");

    request.group_path = "/long";
    const auto long_name = HDF5Browser::BrowsePage(path, request);
    CheckPage(long_name, request, 1, 1);
    Check(long_name.entries.front().path.size() == 4096, "4096-byte full path is permitted");
    request.group_path = long_name.entries.front().path;
    CheckPage(HDF5Browser::BrowsePage(path, request), request, 0, 0);
    request.group_path = "/names";
    request.limit = 1;
    CheckPage(HDF5Browser::BrowsePage(path, request), request, 2, 1);
    request.offset = 1;
    CheckFailure(HDF5Browser::BrowsePage(path, request), Hdf5TableStatus::ResourceLimit, "oversized combined path");
    request.offset = 0;
    request.limit = 2;
    CheckFailure(HDF5Browser::BrowsePage(path, request), Hdf5TableStatus::ResourceLimit, "name failure clears earlier entry");
    request.group_path = "/oversized_names";
    CheckFailure(HDF5Browser::BrowsePage(path, request), Hdf5TableStatus::ResourceLimit, "oversized link name");

    request = {};
    request.cancel_requested = [] { return true; };
    CheckFailure(HDF5Browser::BrowsePage(path, request), Hdf5TableStatus::Cancelled, "cancel before browsing");
    request.group_path = "/pages";
    request.limit = 4;
    int polls = 0;
    request.cancel_requested = [&] { return ++polls >= 4; };
    CheckFailure(HDF5Browser::BrowsePage(path, request), Hdf5TableStatus::Cancelled, "cancel mid-page clears entries");
    Check(polls >= 4, "cancellation is checked during enumeration");
    request = {};
    request.offset = 5; // A group followed by the numeric dataset probe.
    request.limit = 2;
    polls = 0;
    request.cancel_requested = [&] { return ++polls >= 4; };
    CheckFailure(HDF5Browser::BrowsePage(path, request), Hdf5TableStatus::Cancelled, "probe cancellation clears prior entry");
    request.cancel_requested = [] { return false; };
    CheckPage(HDF5Browser::BrowsePage(path, request), request, expected.size(), 2);
#else
    Check(request.group_path == "/" && request.offset == 0 && request.limit == 64, "default browse request");
    CheckFailure(HDF5Browser::BrowsePage("unavailable.h5", request),
                 Hdf5TableStatus::DependencyUnavailable, "disabled browser");
#endif
    std::cout << "HDF5 browser page: " << checks << " checks passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}
