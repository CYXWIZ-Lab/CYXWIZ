// DataInputDialog database and cloud source rendering.

#include "node_config_dialog.h"

#include <cstring>
#include <string>

namespace gui {

void DataInputDialog::RenderDatabaseSource() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::TextColored(accent, "DATABASE CONNECTION");
    ImGui::Spacing();
    ImGui::TextWrapped("%s", UnsupportedApplyMessage());
    ImGui::Spacing();

    int db_idx = static_cast<int>(database_type_);
    const char* db_types[] = {"SQLite", "PostgreSQL", "MySQL", "DuckDB"};

    ImGui::Text("Type:");
    ImGui::SameLine(80);
    ImGui::SetNextItemWidth(120);
    if (ImGui::Combo("##dbtype", &db_idx, db_types, 4)) {
        database_type_ = static_cast<DatabaseType>(db_idx);
        has_changes_ = true;
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    RenderDatabaseConnection();

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    RenderSQLQuery();
}

void DataInputDialog::RenderDatabaseConnection() {
    if (database_type_ == DatabaseType::SQLite || database_type_ == DatabaseType::DuckDB) {
        ImGui::Text("Database file:");
        ImGui::SameLine(100);
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 110);
        ImGui::InputText("##dbfile", db_file_, sizeof(db_file_));
        ImGui::SameLine();
        if (ImGui::Button("...##dbbrowse", ImVec2(100, 0))) {
            std::string path;
            const char* filter = (database_type_ == DatabaseType::SQLite)
                ? "SQLite\0*.db;*.sqlite;*.sqlite3\0All Files\0*.*\0"
                : "DuckDB\0*.duckdb;*.db\0All Files\0*.*\0";
            if (FileSelector("Select Database:", path, filter)) {
                strncpy(db_file_, path.c_str(), sizeof(db_file_) - 1);
                has_changes_ = true;
            }
        }
    } else {
        ImGui::Text("Host:");
        ImGui::SameLine(80);
        ImGui::SetNextItemWidth(150);
        ImGui::InputText("##dbhost", db_host_, sizeof(db_host_));
        ImGui::SameLine();
        ImGui::Text("Port:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(60);
        ImGui::InputInt("##dbport", &db_port_);

        ImGui::Text("Database:");
        ImGui::SameLine(80);
        ImGui::SetNextItemWidth(150);
        ImGui::InputText("##dbname", db_name_, sizeof(db_name_));

        ImGui::Text("User:");
        ImGui::SameLine(80);
        ImGui::SetNextItemWidth(120);
        ImGui::InputText("##dbuser", db_user_, sizeof(db_user_));
        ImGui::SameLine();
        ImGui::Text("Password:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(100);
        ImGui::InputText("##dbpass", db_password_, sizeof(db_password_), ImGuiInputTextFlags_Password);
    }

    ImGui::Spacing();

    ImGui::BeginDisabled();
    ImGui::Button("Connection test unavailable", ImVec2(220, 0));
    ImGui::EndDisabled();
}

void DataInputDialog::RenderSQLQuery() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::TextColored(accent, "SQL Query");
    ImGui::Spacing();

    ImGui::SetNextItemWidth(-1);
    if (ImGui::InputTextMultiline("##sqlquery", sql_query_, sizeof(sql_query_), ImVec2(0, 100))) {
        has_changes_ = true;
    }

    ImGui::Spacing();

    if (ImGui::Button("SELECT *")) {
        strncpy(sql_query_, "SELECT * FROM table_name LIMIT 1000", sizeof(sql_query_) - 1);
    }
    ImGui::SameLine();
    if (ImGui::Button("SHOW TABLES")) {
        if (database_type_ == DatabaseType::SQLite || database_type_ == DatabaseType::DuckDB) {
            strncpy(sql_query_, "SELECT name FROM sqlite_master WHERE type='table'", sizeof(sql_query_) - 1);
        } else if (database_type_ == DatabaseType::PostgreSQL) {
            strncpy(sql_query_, "SELECT tablename FROM pg_tables WHERE schemaname='public'", sizeof(sql_query_) - 1);
        } else {
            strncpy(sql_query_, "SHOW TABLES", sizeof(sql_query_) - 1);
        }
    }

    ImGui::Spacing();

    ImGui::BeginDisabled();
    ImGui::Button("Execute Query", ImVec2(-1, 25));
    ImGui::EndDisabled();
    ImGui::TextDisabled("Database query preview is planned but not wired yet.");
}

void DataInputDialog::RenderCloudSource() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::TextColored(accent, "CLOUD STORAGE");
    ImGui::Spacing();
    ImGui::TextWrapped("%s", UnsupportedApplyMessage());
    ImGui::Spacing();

    static int cloud_provider = 0;
    ImGui::RadioButton("AWS S3", &cloud_provider, 0);
    ImGui::SameLine();
    ImGui::RadioButton("Google Cloud", &cloud_provider, 1);
    ImGui::SameLine();
    ImGui::RadioButton("Azure Blob", &cloud_provider, 2);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Text("Bucket:");
    ImGui::SameLine(100);
    ImGui::SetNextItemWidth(200);
    ImGui::InputText("##bucket", cloud_bucket_, sizeof(cloud_bucket_));

    ImGui::Text("Path:");
    ImGui::SameLine(100);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 110);
    ImGui::InputText("##cloudpath", cloud_path_, sizeof(cloud_path_));

    ImGui::Spacing();

    ImGui::Text("Credentials:");
    ImGui::SameLine(100);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 110);
    ImGui::InputText("##cloudcreds", cloud_credentials_, sizeof(cloud_credentials_));
    ImGui::SameLine();
    if (ImGui::Button("...##credbrowse", ImVec2(100, 0))) {
        std::string path;
        if (FileSelector("Select Credentials:", path, "JSON\0*.json\0All Files\0*.*\0")) {
            strncpy(cloud_credentials_, path.c_str(), sizeof(cloud_credentials_) - 1);
        }
    }

    ImGui::Spacing();
    ImGui::TextDisabled("AWS: ~/.aws/credentials or JSON key");
    ImGui::TextDisabled("GCS: service-account.json");
    ImGui::TextDisabled("Azure: connection string or SAS token");

    ImGui::Spacing();

    ImGui::BeginDisabled();
    ImGui::Button("Connect & List Files", ImVec2(-1, 30));
    ImGui::EndDisabled();
}

} // namespace gui
