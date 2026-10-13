// Column roles and the dataset contract (TOFIX134 P3.4): inference on the
// real Spotify columns (board 14), contract > user > inferred, conflicts,
// unmatched roles, and the project store.

#include "../src/core/column_role_store.h"
#include "../src/core/dataset_contract.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

using namespace cyxwiz;
namespace fs = std::filesystem;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

ColumnFacts F(const std::string& name, ColumnFacts::Type type, size_t distinct, double avg_len = 0, double date_share = 0) {
    ColumnFacts f;
    f.name = name;
    f.type = type;
    f.rows = f.non_null = 8582;
    f.distinct = distinct;
    f.avg_length = avg_len;
    f.date_share = date_share;
    return f;
}
}  // namespace

int main() {
    using T = ColumnFacts::Type;
    // Spotify, 8,582 rows (profile of 2026-10-03).
    const std::vector<ColumnFacts> spotify = {
        F("track_id", T::Text, 8582, 22.0),
        F("track_name", T::Text, 7462, 17.0),
        F("track_number", T::Integer, 54),
        F("track_popularity", T::Integer, 98),
        F("explicit", T::Boolean, 2),
        F("artist_name", T::Text, 2547, 10.5),
        F("artist_popularity", T::Integer, 96),
        F("artist_followers", T::Integer, 3740),
        F("artist_genres", T::Text, 662, 13.1),
        F("album_id", T::Text, 5205, 22.0),
        F("album_name", T::Text, 4870, 20.9),
        F("album_release_date", T::Text, 2384, 10.0, 1.0),
        F("album_total_tracks", T::Integer, 82),
        F("album_type", T::Text, 3, 5.6),
        F("track_duration_min", T::Float, 647),
    };
    // The Data Input's label column is the target (contract).
    DatasetContract c = BuildContract("D:/data/spotify.csv", spotify, {{"track_popularity", ColumnRole::Target}}, {});
    const auto role = [&](const std::string& n) { return c.Find(n)->role; };
    Check(role("track_id") == ColumnRole::Id && c.Find("track_id")->reason == "unique per row", "track_id: ID, unique per row");
    Check(role("track_name") == ColumnRole::Text, "track_name: text (7,462 of 8,582 different)");
    Check(role("track_number") == ColumnRole::Numeric, "track_number: numeric (54 values)");
    Check(role("track_popularity") == ColumnRole::Target && c.Find("track_popularity")->source == RoleSource::Contract &&
              c.Find("track_popularity")->reason == "Data Input label", "target from the Data Input label");
    Check(role("explicit") == ColumnRole::Category && c.Find("explicit")->reason == "2 values", "explicit: category, 2 values");
    Check(role("artist_name") == ColumnRole::Category && c.Find("artist_name")->reason == "2,547 values", "artist_name: category");
    Check(role("artist_genres") == ColumnRole::Category, "artist_genres: category");
    Check(role("album_id") == ColumnRole::Id && c.Find("album_id")->reason == "name ends in _id", "album_id: ID by name");
    Check(role("album_name") == ColumnRole::Text, "album_name: text");
    Check(role("album_release_date") == ColumnRole::DateTime, "album_release_date: date");
    Check(role("album_type") == ColumnRole::Category && c.Find("album_type")->reason == "3 values", "album_type: category");
    Check(role("track_duration_min") == ColumnRole::Numeric && role("artist_followers") == ColumnRole::Numeric, "numbers");
    Check(c.Target() == std::string("track_popularity"), "the target");
    Check(!c.schema_fingerprint.empty() && c.schema_fingerprint == SchemaFingerprint(spotify), "schema fingerprint");

    // User roles: applied where the graph says nothing; a contradiction is a warning, not applied.
    c = BuildContract("D:/data/spotify.csv", spotify, {{"track_popularity", ColumnRole::Target}},
                      {{"artist_name", ColumnRole::Text}, {"track_popularity", ColumnRole::Numeric}, {"gone_column", ColumnRole::Ignore}});
    Check(role("artist_name") == ColumnRole::Text && c.Find("artist_name")->source == RoleSource::User, "user role applied");
    Check(role("track_popularity") == ColumnRole::Target && c.Find("track_popularity")->user_conflict == ColumnRole::Numeric,
          "changing the target is kept as a warning; training keeps the label");
    Check(c.unmatched == std::vector<std::string>({"gone_column"}), "a saved role for a column that is gone is kept as unmatched");
    c = BuildContract("k", spotify, {{"track_popularity", ColumnRole::Target}}, {{"artist_popularity", ColumnRole::Target}});
    Check(c.Find("artist_popularity")->user_conflict == ColumnRole::Target, "a second target from the user is a warning");

    // More inference.
    std::string why;
    Check(InferRole(F("id", T::Integer, 8582), &why) == ColumnRole::Id && why == "named id", "integer id");
    Check(InferRole(F("rating", T::Integer, 5), &why) == ColumnRole::Category && why == "5 values", "few integers: category");
    Check(InferRole(F("review", T::Text, 4000, 180.0)) == ColumnRole::Text, "long text");
    ColumnFacts img = F("file", T::Text, 2000, 30);
    img.path_share = 1.0;
    Check(InferRole(img) == ColumnRole::FilePath, "image paths");
    Check(InferRole(F("when", T::Temporal, 100)) == ColumnRole::DateTime, "date type");
    Check(LooksLikeDate("2024-05-01") && LooksLikeDate("2024/05/01") && LooksLikeDate("2024-05-01T10:00:00") && LooksLikeDate("01/05/2024") &&
              LooksLikeDate("2024-05") && !LooksLikeDate("hello") && !LooksLikeDate("12345") && !LooksLikeDate("2024-05-01x"),
          "date shapes");
    Check(LooksLikeMediaPath("cats/001.JPG") && LooksLikeMediaPath("a.wav") && !LooksLikeMediaPath("a.csv"), "media paths");
    Check(RoleFromId("file_path") == ColumnRole::FilePath && !RoleFromId("x") && std::string(RoleLabel(ColumnRole::DateTime)) == "Date" &&
              std::string(RoleSourceLabel(RoleSource::User)) == "you" && !IsFeatureRole(ColumnRole::Id) && IsFeatureRole(ColumnRole::Category),
          "role names");

    // The project store: round trip, missing texts, clear, a broken file is not overwritten.
    const fs::path root = fs::temp_directory_path() / "cyxwiz_test_roles";
    std::error_code ec;
    fs::remove_all(root, ec);
    const std::string file = ColumnRoleStore::ProjectFile(root.string());
    Check(fs::path(file).filename() == "column_roles.json" && fs::path(file).parent_path().filename() == "datasets", "project file place");
    {
        ColumnRoleStore store(file);
        Check(store.Load(), "no file: empty store");
        store.SetRole("D:/data/spotify.csv", "artist_name", ColumnRole::Text);
        store.SetRole("D:/data/spotify.csv", "album_id", ColumnRole::Ignore);
        store.SetMissingText("D:/data/spotify.csv", "artist_genres", {"N/A"});
        store.SetSchema("D:/data/spotify.csv", c.schema_fingerprint);
        std::string error;
        Check(store.Save(&error) && fs::exists(file), "saved: " + error);
    }
    {
        ColumnRoleStore store(file);
        Check(store.Load(), "loads");
        const auto* s = store.Find("D:/data/spotify.csv");
        Check(s && s->roles.at("artist_name") == ColumnRole::Text && s->roles.at("album_id") == ColumnRole::Ignore &&
                  s->missing_text.at("artist_genres") == std::vector<std::string>({"N/A"}) && s->schema_fingerprint == c.schema_fingerprint,
              "round trip");
        store.ClearRole("D:/data/spotify.csv", "album_id");
        Check(store.RolesFor("D:/data/spotify.csv").size() == 1 && store.RolesFor("other").empty(), "clear a role; other datasets empty");
    }
    std::ofstream(file, std::ios::trunc) << "{ not json";
    {
        ColumnRoleStore store(file);
        std::string error;
        Check(!store.Load(&error) && error.find("could not be read") != std::string::npos, "broken file reported");
        store.SetRole("x", "y", ColumnRole::Text);
        Check(!store.Save(&error), "a broken file is not overwritten");
        std::ifstream in(file);
        std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        Check(text == "{ not json", "the broken file is left as it is");
    }
    fs::remove_all(root, ec);

    std::cout << "dataset contract: Spotify roles as on board 14, contract > user > inferred, target warnings, unmatched roles, "
                 "inference cases, date and path shapes, project store round trip and broken-file safety. OK\n";
    return 0;
}
