#include "python_node_provider.h"

#include "python_node_factory.h"
#include "../../core/extension_node_registry.h"

#include <cyxwiz/sequential.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <sstream>
#include <system_error>

namespace cyxwiz {

namespace fs = std::filesystem;

namespace {

// The factory of a bundle whose code is not approved: the node is listed and
// saved, compiling it reports why it does not run.
class NotApprovedFactory final : public ITensorNodeFactory {
public:
    explicit NotApprovedFactory(std::string message) : message_(std::move(message)) {}

    ExtensionShapeResult InferOutputShape(const std::vector<size_t>&,
                                          const std::map<std::string, std::string>&) const override {
        return {false, {}, message_};
    }
    std::unique_ptr<Module> CreateModule(const std::vector<size_t>&, const std::map<std::string, std::string>&,
                                         std::string& error) const override {
        error = message_;
        return nullptr;
    }

private:
    std::string message_;
};

void AppendFileStamp(std::ostringstream& out, const fs::path& path) {
    std::error_code ec;
    const auto size = fs::file_size(path, ec);
    const auto time = fs::last_write_time(path, ec);
    out << path.string() << '|' << (ec ? 0 : size) << '|'
        << (ec ? 0 : time.time_since_epoch().count()) << '\n';
}

}  // namespace

std::string PythonBundleNotApprovedMessage(const std::string& bundle_id) {
    return "the code of Python node bundle " + bundle_id +
           " is not approved yet; approve it in the Plugin Manager to run it";
}

PythonNodeProvider::PythonNodeProvider(PythonNodeApprovals& approvals) : approvals_(approvals) {}

PythonNodeProvider::~PythonNodeProvider() { Clear(); }

fs::path PythonNodeProvider::ProjectNodesFolder(const fs::path& project_root) { return project_root / "nodes"; }

std::string PythonNodeProvider::Stamp(const std::vector<fs::path>& roots) const {
    std::ostringstream out;
    for (const auto& root : roots) out << "root " << root.string() << '\n';
    for (const auto& folder : FindPythonBundles(roots)) {
        AppendFileStamp(out, folder / "node.json");
        AppendFileStamp(out, folder / "node.py");
    }
    // An approval changes what is registered too.
    for (const auto& a : approvals_.All()) out << "approved " << a.bundle_id << ' ' << a.content_hash << '\n';
    return out.str();
}

bool PythonNodeProvider::Refresh(const std::vector<fs::path>& roots) {
    const std::string stamp = Stamp(roots);
    if (scanned_ && roots == roots_ && stamp == stamp_) return false;
    roots_ = roots;
    stamp_ = stamp;
    scanned_ = true;
    Register();
    return true;
}

void PythonNodeProvider::Rescan() {
    stamp_ = Stamp(roots_);
    scanned_ = true;
    Register();
}

void PythonNodeProvider::Clear() {
    auto& registry = ExtensionNodeRegistry::Instance();
    for (const auto& id : registered_providers_) registry.RemoveByProvider(id);
    registered_providers_.clear();
    bundles_.clear();
    issues_.clear();
}

void PythonNodeProvider::Register() {
    Clear();
    auto& registry = ExtensionNodeRegistry::Instance();
    std::set<std::string> seen_ids;
    for (const auto& folder : FindPythonBundles(roots_)) {
        PythonBundleState state;
        std::string error;
        if (!ReadPythonBundle(folder, state.bundle, error)) {
            issues_.push_back({folder, error});
            spdlog::warn("Python node bundle {} not loaded: {}", folder.string(), error);
            continue;
        }
        // The first root wins (project before user before plugins).
        if (!seen_ids.insert(state.bundle.id).second) {
            issues_.push_back({folder, "another bundle with the id " + state.bundle.id + " is already loaded"});
            continue;
        }
        // A plugin may already own this id.
        const bool taken = std::any_of(state.bundle.nodes.begin(), state.bundle.nodes.end(), [&](const auto& node) {
            return registry.Has(MakeExtensionTypeId(state.bundle.id, node.descriptor.type_name));
        });
        if (taken) {
            issues_.push_back({folder, "the id " + state.bundle.id + " is already used by a loaded plugin"});
            continue;
        }
        state.approved = approvals_.IsApproved(state.bundle.id, state.bundle.content_hash);
        size_t registered = 0;
        for (const auto& node : state.bundle.nodes) {
            ExtensionNodeRegistry::Registration registration;
            registration.descriptor = node.descriptor;
            if (state.approved) {
                registration.tensor_factory = MakePythonTensorNodeFactory(state.bundle, node);
            } else {
                registration.tensor_factory =
                    std::make_shared<NotApprovedFactory>(PythonBundleNotApprovedMessage(state.bundle.id));
            }
            if (registry.Register(std::move(registration), error)) {
                ++registered;
            } else {
                issues_.push_back({folder, error});
            }
        }
        if (registered > 0) registered_providers_.insert(state.bundle.id);
        spdlog::info("Python node bundle {} ({}): {} node(s), {}", state.bundle.id, folder.string(), registered,
                     state.approved ? "approved" : "not approved yet");
        bundles_.push_back(std::move(state));
    }
}

}  // namespace cyxwiz
