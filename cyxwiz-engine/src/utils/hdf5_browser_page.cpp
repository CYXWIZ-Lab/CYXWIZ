#include "hdf5_browser.h"
#include "../core/hdf5_table_internal.h"

#ifdef CYXWIZ_HAS_HDF5
#include <highfive/highfive.hpp>

#include <algorithm>
#include <new>
#include <utility>
#endif

namespace cyxwiz {

#ifdef CYXWIZ_HAS_HDF5
namespace {

using hdf5_detail::ReadFailure;
constexpr size_t kMaxPathBytes = 4096;

void CheckCancellation(const Hdf5BrowseRequest& request) {
    if (request.cancel_requested && request.cancel_requested())
        throw ReadFailure(Hdf5TableStatus::Cancelled, "HDF5 browsing was cancelled");
}

void ValidateRequest(const Hdf5BrowseRequest& request) {
    if (request.limit == 0 || request.limit > 128)
        throw ReadFailure(Hdf5TableStatus::InvalidSelection, "HDF5 browse limit must be 1-128 entries");
    hdf5_detail::ValidatePathSyntax(request.group_path, true);
}

std::string ReadEntryName(hid_t group, hsize_t index, size_t path_prefix_bytes) {
    const auto length = H5Lget_name_by_idx(
        group, ".", H5_INDEX_NAME, H5_ITER_INC, index, nullptr, 0, H5P_DEFAULT);
    if (length <= 0)
        throw ReadFailure(Hdf5TableStatus::ReadFailed, "Cannot read HDF5 link name length");
    if (static_cast<uint64_t>(length) > kMaxPathBytes ||
        path_prefix_bytes > kMaxPathBytes ||
        static_cast<size_t>(length) > kMaxPathBytes - path_prefix_bytes)
        throw ReadFailure(Hdf5TableStatus::ResourceLimit, "HDF5 entry path exceeds 4096 bytes");

    std::string name(static_cast<size_t>(length) + 1, '\0');
    const auto copied = H5Lget_name_by_idx(
        group, ".", H5_INDEX_NAME, H5_ITER_INC, index, name.data(), name.size(), H5P_DEFAULT);
    if (copied != length)
        throw ReadFailure(Hdf5TableStatus::ReadFailed, "HDF5 link changed while reading its name");
    name.resize(static_cast<size_t>(length));
    return name;
}

bool IsDatasetCompatibilityStatus(Hdf5TableStatus status) {
    switch (status) {
        case Hdf5TableStatus::Ok:
        case Hdf5TableStatus::UnsupportedLayout:
        case Hdf5TableStatus::UnsupportedRank:
        case Hdf5TableStatus::UnsupportedType:
        case Hdf5TableStatus::EmptyDataset:
        case Hdf5TableStatus::ResourceLimit:
            return true;
        default:
            return false;
    }
}

} // namespace
#endif

Hdf5BrowsePage HDF5Browser::BrowsePage(const std::string& path,
                                     const Hdf5BrowseRequest& request) {
    Hdf5BrowsePage page;
    page.offset = request.offset;
    page.next_offset = request.offset;
#ifdef CYXWIZ_HAS_HDF5
    try {
        CheckCancellation(request);
        ValidateRequest(request);
        page.group_path = request.group_path;
        const auto file = hdf5_detail::OpenFile(path);
        hdf5_detail::ValidateLocalPath(file, request.group_path, true);
        if (file.getObjectType(request.group_path) != HighFive::ObjectType::Group)
            throw ReadFailure(Hdf5TableStatus::InvalidSelection, "Selected HDF5 object is not a group");
        const auto group = file.getGroup(request.group_path);
        H5G_info_t info{};
        if (H5Gget_info(group.getId(), &info) < 0)
            throw ReadFailure(Hdf5TableStatus::ReadFailed, "Cannot inspect HDF5 group");
        page.total_entries = info.nlinks;
        if (request.offset > page.total_entries)
            throw ReadFailure(Hdf5TableStatus::InvalidSelection, "HDF5 browse offset exceeds group entry count");

        const auto count = std::min<uint64_t>(request.limit, page.total_entries - request.offset);
        page.entries.reserve(static_cast<size_t>(count));
        const size_t prefix_bytes = request.group_path == "/" ? 1 : request.group_path.size() + 1;
        for (uint64_t index = 0; index < count; ++index) {
            CheckCancellation(request);
            Hdf5BrowseEntry entry;
            entry.name = ReadEntryName(group.getId(), static_cast<hsize_t>(request.offset + index), prefix_bytes);
            entry.path = request.group_path == "/"
                ? "/" + entry.name : request.group_path + "/" + entry.name;

            // Inspect the link itself before querying anything about its target.
            H5L_info2_t link{};
            if (H5Lget_info2(group.getId(), entry.name.c_str(), &link, H5P_DEFAULT) < 0)
                throw ReadFailure(Hdf5TableStatus::ReadFailed, "Cannot inspect HDF5 link: " + entry.path);
            if (link.type != H5L_TYPE_HARD) {
                entry.kind = Hdf5ObjectKind::IndirectLink;
                entry.status = Hdf5TableStatus::UnsupportedLayout;
                entry.reason = "Indirect HDF5 links are not followed";
            } else {
                const auto kind = group.getObjectType(entry.name);
                if (kind == HighFive::ObjectType::Group) {
                    entry.kind = Hdf5ObjectKind::Group;
                    entry.status = Hdf5TableStatus::Ok;
                } else if (kind == HighFive::ObjectType::Dataset) {
                    entry.kind = Hdf5ObjectKind::Dataset;
                    Hdf5TableReadOptions options;
                    options.selection.data_path = entry.path;
                    options.cancel_requested = request.cancel_requested;
                    auto probe = ProbeHdf5Table(path, options);
                    if (!IsDatasetCompatibilityStatus(probe.status))
                        throw ReadFailure(probe.status, probe.error);
                    entry.dataset = std::move(probe.data);
                    entry.status = probe.status;
                    entry.reason = std::move(probe.error);
                    entry.estimated_materialized_bytes = probe.estimated_materialized_bytes;
                } else {
                    entry.kind = Hdf5ObjectKind::Other;
                    entry.status = Hdf5TableStatus::UnsupportedLayout;
                    entry.reason = "HDF5 object is neither a group nor a dataset";
                }
            }
            page.entries.push_back(std::move(entry));
        }
        CheckCancellation(request);
        page.next_offset = request.offset + count;
        page.has_next = page.next_offset < page.total_entries;
        page.status = Hdf5TableStatus::Ok;
        return page;
    } catch (const ReadFailure& error) {
        page.status = error.status;
        page.error = error.what();
    } catch (const std::bad_alloc&) {
        page.status = Hdf5TableStatus::ResourceLimit;
        page.error = "HDF5 browsing could not allocate the required memory";
    } catch (const std::exception& error) {
        page.status = Hdf5TableStatus::ReadFailed;
        page.error = "HDF5 browsing failed: " + std::string(error.what());
    }
#else
    (void)path;
    page.status = Hdf5TableStatus::DependencyUnavailable;
    page.error = "HDF5 support is not compiled into this build";
#endif
    page.entries.clear();
    page.has_next = false;
    page.next_offset = request.offset;
    return page;
}

} // namespace cyxwiz
