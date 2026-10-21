// The sign-in screen's words and states (TOFIX139), checked against the
// approved mockup: one message at a time, each saying what happened and what
// to do next; the API's raw text never shows.
#include "core/sign_in_presentation.h"

#include <catch2/catch_test_macros.hpp>

using namespace cyxwiz::servernode::core;

namespace {

SignInInputs LocalStack() {
    SignInInputs in;
    in.service_url = "http://127.0.0.1:3002/api";
    in.daemon_connected = true;
    in.version = "0.3";
    return in;
}

bool Has(const std::vector<SignInAction>& actions, SignInAction action) {
    for (const auto a : actions) {
        if (a == action) return true;
    }
    return false;
}

}  // namespace

TEST_CASE("sign-in card: default state", "[sign-in]") {
    const auto card = BuildSignInCard(LocalStack());
    CHECK(card.title == "CyxWiz Server Node");
    CHECK(card.subtitle == "Sign in to lend this machine to the network");
    CHECK(card.service_prefix == "Signing in to");
    CHECK(card.service_host == "127.0.0.1:3002");
    CHECK(card.change_enabled);
    CHECK(card.fields_enabled);
    CHECK(card.button_label == "Sign in");
    CHECK_FALSE(card.button_enabled);  // both fields empty
    CHECK_FALSE(card.button_busy);
    CHECK(card.busy_note.empty());
    CHECK(card.create_prefix == "New to CyxWiz?");
    CHECK(card.create_link == "Create an account");
    CHECK(card.offline_link == "Work offline");
    CHECK(card.offline_note == "Local training only. Nothing is lent and no one can reserve this machine.");
    CHECK(card.footer == "CyxWiz Server Node 0.3 \xC2\xB7 the daemon is connected");
    CHECK_FALSE(card.message.shown);
    CHECK(card.focus == SignInFocus::Email);
    CHECK(card.log_line.empty());

    SECTION("the button turns on when both fields have text") {
        auto in = LocalStack();
        in.email = "e2e-user@local.test";
        CHECK_FALSE(BuildSignInCard(in).button_enabled);
        in.password_filled = true;
        CHECK(BuildSignInCard(in).button_enabled);
        in.email = "   ";
        CHECK_FALSE(BuildSignInCard(in).button_enabled);
    }
    SECTION("a remembered email puts the cursor in the password field") {
        auto in = LocalStack();
        in.email = "e2e-user@local.test";
        CHECK(BuildSignInCard(in).focus == SignInFocus::Password);
    }
    SECTION("the footer says when the daemon is not running and skips an unknown version") {
        auto in = LocalStack();
        in.daemon_connected = false;
        in.version.clear();
        CHECK(BuildSignInCard(in).footer == "CyxWiz Server Node \xC2\xB7 the daemon is not running");
    }
}

TEST_CASE("sign-in card: signing in locks everything but shows what is happening", "[sign-in]") {
    auto in = LocalStack();
    in.phase = SignInPhase::SigningIn;
    in.email = "e2e-user@local.test";
    in.password_filled = true;
    in.problem = SignInProblem::WrongPassword;  // a stale message never shows while busy
    const auto card = BuildSignInCard(in);
    CHECK(card.button_busy);
    CHECK(card.button_label == "Signing in...");
    CHECK_FALSE(card.button_enabled);
    CHECK_FALSE(card.fields_enabled);
    CHECK_FALSE(card.change_enabled);
    CHECK_FALSE(card.create_enabled);
    CHECK_FALSE(card.offline_enabled);
    CHECK(card.busy_note == "Checking your password with the CyxWiz service. This takes a few seconds.");
    CHECK_FALSE(card.message.shown);
    CHECK(card.focus == SignInFocus::None);
}

TEST_CASE("sign-in errors: one message, what happened and what to do", "[sign-in]") {
    auto in = LocalStack();
    in.email = "e2e-user@local.test";

    SECTION("service unreachable") {
        in.problem = SignInProblem::Unreachable;
        in.problem_detail = "Network error: Read error";
        const auto card = BuildSignInCard(in);
        REQUIRE(card.message.shown);
        CHECK(card.message.tone == MessageTone::Error);
        CHECK(card.message.title == "Can't reach the CyxWiz service at 127.0.0.1:3002");
        CHECK(card.message.body == "Check that the service is running and that this address is right.");
        CHECK(Has(card.message.actions, SignInAction::TryAgain));
        CHECK(Has(card.message.actions, SignInAction::ChangeService));
        CHECK(card.message.title.find("Read error") == std::string::npos);  // raw text never shows
        CHECK(card.log_line == "Sign-in: service unreachable: Network error: Read error");
    }
    SECTION("wrong password, with the Caps Lock hint only when it is on") {
        in.problem = SignInProblem::WrongPassword;
        in.http_status = 401;
        auto card = BuildSignInCard(in);
        CHECK(card.message.title == "That password doesn't match this email");
        CHECK(card.message.body == "Try again, or reset it on the website.");
        CHECK(card.message.actions.empty());
        CHECK_FALSE(card.message.caps_lock_hint);
        in.caps_lock = true;
        card = BuildSignInCard(in);
        CHECK(card.message.caps_lock_hint);
        CHECK(card.log_line == "Sign-in: wrong password (HTTP 401)");
    }
    SECTION("no account offers to create one") {
        in.problem = SignInProblem::NoAccount;
        const auto card = BuildSignInCard(in);
        CHECK(card.message.title == "No account with that email");
        CHECK(card.message.body == "Check the spelling, or create an account on the website.");
        CHECK(card.message.link_word == "create an account");
        CHECK(Has(card.message.actions, SignInAction::CreateAccount));
    }
    SECTION("expired session is information, not an error") {
        in.problem = SignInProblem::Expired;
        const auto card = BuildSignInCard(in);
        CHECK(card.message.tone == MessageTone::Info);
        CHECK(card.message.title == "Your sign-in expired");
        CHECK(card.message.body ==
              "Sign in again to keep lending this machine. Jobs already running were not affected.");
    }
    SECTION("service error names the HTTP status") {
        in.problem = SignInProblem::ServiceError;
        in.http_status = 500;
        const auto card = BuildSignInCard(in);
        CHECK(card.message.title == "The CyxWiz service had a problem (HTTP 500)");
        CHECK(card.message.body == "Try again in a moment. If it keeps happening, the node's log has the details.");
        CHECK(Has(card.message.actions, SignInAction::TryAgain));
    }
    SECTION("the rest") {
        in.problem = SignInProblem::TooManyAttempts;
        CHECK(BuildSignInCard(in).message.title == "Too many sign-in attempts");
        in.problem = SignInProblem::Suspended;
        CHECK(BuildSignInCard(in).message.title == "This account is suspended");
        in.problem = SignInProblem::Other;
        CHECK(BuildSignInCard(in).message.title == "Couldn't sign in");
    }
}

TEST_CASE("classifying the API's answers", "[sign-in]") {
    CHECK(ClassifySignInError(0, "Network error: Connection failed") == SignInProblem::Unreachable);
    CHECK(ClassifySignInError(0, "Connection error: timed out") == SignInProblem::Unreachable);
    CHECK(ClassifySignInError(0, "") == SignInProblem::None);
    CHECK(ClassifySignInError(401, "Invalid email or password") == SignInProblem::WrongPassword);
    CHECK(ClassifySignInError(401, "No account with that email") == SignInProblem::NoAccount);
    CHECK(ClassifySignInError(404, "User not found") == SignInProblem::NoAccount);
    CHECK(ClassifySignInError(403, "Account suspended") == SignInProblem::Suspended);
    CHECK(ClassifySignInError(429, "Too many attempts. Please wait.") == SignInProblem::TooManyAttempts);
    CHECK(ClassifySignInError(500, "Database error: pool timed out") == SignInProblem::ServiceError);
    CHECK(ClassifySignInError(503, "Server error (503)") == SignInProblem::ServiceError);
    CHECK(ClassifySignInError(418, "I'm a teapot") == SignInProblem::Other);
    CHECK(ClassifySignInError(0, "Session expired. Please log in again.") == SignInProblem::Expired);
}

TEST_CASE("service address: host shown, website derived", "[sign-in]") {
    CHECK(ServiceHost("http://127.0.0.1:3002/api") == "127.0.0.1:3002");
    CHECK(ServiceHost("https://cyxwiz.com/api") == "cyxwiz.com");
    CHECK(ServiceHost("https://cyxwiz.com:443/api") == "cyxwiz.com");
    CHECK(ServiceHost("http://192.168.1.116:3002/api") == "192.168.1.116:3002");
    CHECK(ServiceHost("  http://192.168.1.116:3002/api \n") == "192.168.1.116:3002");
    CHECK(WebsiteOf("https://cyxwiz.com/api") == "https://cyxwiz.com");
    CHECK(WebsiteOf("http://127.0.0.1:3002/api") == "http://127.0.0.1:3002");
    CHECK(WebsiteOf("") == "");
}

TEST_CASE("change the sign-in service", "[sign-in]") {
    SECTION("addresses are checked and normalized") {
        auto check = CheckServiceAddress("http://192.168.1.116:3002/api");
        CHECK(check.ok);
        CHECK(check.normalized == "http://192.168.1.116:3002/api");
        check = CheckServiceAddress("  https://cyxwiz.com  ");
        CHECK(check.ok);
        CHECK(check.normalized == "https://cyxwiz.com/api");  // "/api" added
        check = CheckServiceAddress("http://127.0.0.1:3002/api/");
        CHECK(check.normalized == "http://127.0.0.1:3002/api");
        check = CheckServiceAddress("cyxwiz.com/api");
        CHECK_FALSE(check.ok);
        CHECK(check.message.rfind("The address must start with http:// or https://", 0) == 0);
        check = CheckServiceAddress("http://:3002/api");
        CHECK_FALSE(check.ok);
        CHECK(check.message.rfind("The address has no host", 0) == 0);
        check = CheckServiceAddress("http://192.168.1.116:abc/api");
        CHECK_FALSE(check.ok);
        CHECK(check.message.rfind("The port must be a number", 0) == 0);
        check = CheckServiceAddress("http://192.168.1.116 :3002/api");
        CHECK_FALSE(check.ok);
        check = CheckServiceAddress("");
        CHECK_FALSE(check.ok);
        CHECK(check.message == "Enter an address like http://192.168.1.116:3002/api");
    }
    SECTION("the dialog") {
        const auto dialog = BuildServiceDialog("http://127.0.0.1:3002/api", ServiceReach::Reachable);
        CHECK(dialog.title == "Sign-in service");
        CHECK(dialog.explanation ==
              "The CyxWiz website this node signs in with. The central server it reports to is set separately, in "
              "Settings.");
        REQUIRE(dialog.examples.size() == 3);
        CHECK(dialog.examples[0].first == "CyxWiz (production)");
        CHECK(dialog.examples[0].second == "https://cyxwiz.com/api");
        CHECK(dialog.examples[1].second == "http://127.0.0.1:3002/api");
        CHECK(dialog.examples[2].first == "Another PC on the network");
        CHECK(dialog.reach_line == "Reachable \xC2\xB7 CyxWiz API");
        CHECK(dialog.use_enabled);
        CHECK(dialog.use_label == "Use this service");
        CHECK(dialog.cancel_label == "Cancel");
        CHECK(dialog.address_problem.empty());
        CHECK(dialog.note.rfind("\"Create an account\" opens the website of this address.", 0) == 0);

        const auto bad = BuildServiceDialog("nope", ServiceReach::Reachable);
        CHECK_FALSE(bad.use_enabled);
        CHECK(bad.reach == ServiceReach::Unknown);  // no reach claim for a malformed address
        CHECK(bad.reach_line.empty());
        CHECK_FALSE(bad.address_problem.empty());

        CHECK(BuildServiceDialog("http://x:1/api", ServiceReach::Unreachable).reach_line ==
              "Not reachable \xC2\xB7 check the address and that the service is running");
        CHECK(BuildServiceDialog("http://x:1/api", ServiceReach::Checking).reach_line == "Checking...");
    }
}

TEST_CASE("work offline confirm", "[sign-in]") {
    const auto confirm = BuildOfflineConfirm();
    CHECK(confirm.title == "Work offline?");
    REQUIRE(confirm.lines.size() == 3);
    CHECK(confirm.lines[0] == "You can train on this machine from the local Engine and manage its models.");
    CHECK(confirm.lines[1].rfind("Nothing is lent: no one can reserve it, it earns nothing", 0) == 0);
    CHECK(confirm.lines[2] == "Sign in later from Account to go online.");
    CHECK(confirm.back_label == "Back to sign-in");
    CHECK(confirm.confirm_label == "Work offline");
}

TEST_CASE("after sign-in progress card", "[sign-in]") {
    ProgressInputs in;
    in.user_name = "e2e_user";
    in.user_email = "e2e-user@local.test";

    SECTION("registering") {
        const auto card = BuildProgressCard(in);
        CHECK(card.title == "Welcome, e2e_user");
        CHECK(card.subtitle == "Getting this machine onto the network");
        REQUIRE(card.steps.size() == 3);
        CHECK(card.steps[0].state == StepState::Done);
        CHECK(card.steps[0].title == "Signed in");
        CHECK(card.steps[0].detail == "e2e-user@local.test \xC2\xB7 stays signed in; renewed in the background");
        CHECK(card.steps[1].state == StepState::Running);
        CHECK(card.steps[1].title == "Registering this machine");
        CHECK(card.steps[1].detail == "Listing its devices with your account");
        CHECK(card.steps[2].state == StepState::Pending);
        CHECK(card.steps[2].title == "Choose what to lend");
        CHECK(card.steps[2].detail ==
              "Opens the Allocation page: pick devices, memory and cores, then Apply to go live");
        CHECK(card.footnote == "This card shows for a few seconds, then the dashboard opens.");
        CHECK_FALSE(card.failed);
    }
    SECTION("registered") {
        in.registering = StepState::Done;
        in.node_name = "CyxWiz-DESKTOP";
        const auto card = BuildProgressCard(in);
        CHECK(card.steps[1].state == StepState::Done);
        CHECK(card.steps[1].detail == "Listed with your account as CyxWiz-DESKTOP");
    }
    SECTION("registration refused") {
        in.registering = StepState::Failed;
        in.register_error = "machine name already in use";
        const auto card = BuildProgressCard(in);
        CHECK(card.failed);
        CHECK(card.failure_title == "Couldn't register this machine");
        CHECK(card.failure_detail == "The CyxWiz service didn't accept it: machine name already in use.");
        CHECK(card.retry_label == "Try again");
        CHECK(card.continue_label == "Continue to the dashboard");
        CHECK(card.failure_note == "You stay signed in; the node tries again every minute while it runs.");
        CHECK(card.footnote.empty());
    }
    SECTION("registration could not reach the service") {
        in.registering = StepState::Failed;
        in.register_unreachable = true;
        CHECK(BuildProgressCard(in).failure_detail == "The CyxWiz service couldn't be reached.");
    }
    SECTION("a user without a display name is greeted by email") {
        in.user_name.clear();
        CHECK(BuildProgressCard(in).title == "Welcome, e2e-user@local.test");
    }
}
