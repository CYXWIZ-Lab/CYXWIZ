#pragma once

// FontAwesome 6 Free Solid Icon Definitions
// These are Unicode code points for FontAwesome 6 Free Solid icons
// To use: ImGui::Text(ICON_FA_FILE); or with text: ImGui::Text(ICON_FA_FILE " New File");

// Icon font range for FontAwesome 6
#define ICON_MIN_FA 0xe005
#define ICON_MAX_FA 0xf8ff

// FontAwesome 6 Free Solid Icons
// Using explicit UTF-8 byte sequences for MSVC C++20 compatibility

// File operations
#define ICON_FA_FILE                "\xef\x85\x9b"  // U+F15B
#define ICON_FA_FILE_LINES          "\xef\x85\x9c"  // U+F15C
#define ICON_FA_FILE_CODE           "\xef\x87\x89"  // U+F1C9
#define ICON_FA_FOLDER              "\xef\x81\xbb"  // U+F07B
#define ICON_FA_FOLDER_OPEN         "\xef\x81\xbc"  // U+F07C
#define ICON_FA_FOLDER_PLUS         "\xef\x99\x9e"  // U+F65E
#define ICON_FA_FLOPPY_DISK         "\xef\x83\x87"  // U+F0C7
#define ICON_FA_DOWNLOAD            "\xef\x80\x99"  // U+F019

// Edit operations
#define ICON_FA_COPY                "\xef\x83\x85"  // U+F0C5
#define ICON_FA_PASTE               "\xef\x83\xaa"  // U+F0EA
#define ICON_FA_SCISSORS            "\xef\x83\x84"  // U+F0C4
#define ICON_FA_TRASH               "\xef\x87\xb8"  // U+F1F8
#define ICON_FA_TRASH_CAN           "\xef\x8b\xad"  // U+F2ED
#define ICON_FA_PEN                 "\xef\x8c\x84"  // U+F304
#define ICON_FA_PENCIL              "\xef\x8c\x83"  // U+F303
#define ICON_FA_ROTATE_LEFT         "\xef\x8b\xaa"  // U+F2EA
#define ICON_FA_ROTATE_RIGHT        "\xef\x8b\xb9"  // U+F2F9

// View/Layout
#define ICON_FA_TABLE_COLUMNS       "\xef\x83\x9b"  // U+F0DB
#define ICON_FA_EXPAND              "\xef\x81\xa5"  // U+F065
#define ICON_FA_COMPRESS            "\xef\x81\xa6"  // U+F066
#define ICON_FA_MAXIMIZE            "\xef\x8c\x9e"  // U+F31E
#define ICON_FA_MINIMIZE            "\xef\x8b\x91"  // U+F2D1
#define ICON_FA_WINDOW_RESTORE      "\xef\x8b\x92"  // U+F2D2
#define ICON_FA_EYE                 "\xef\x81\xae"  // U+F06E
#define ICON_FA_EYE_SLASH           "\xef\x81\xb0"  // U+F070
#define ICON_FA_BARS                "\xef\x83\x89"  // U+F0C9
#define ICON_FA_GRIP_VERTICAL       "\xef\x96\x8e"  // U+F58E
#define ICON_FA_GRIP                "\xef\x96\x8d"  // U+F58D

// Navigation
#define ICON_FA_ARROW_LEFT          "\xef\x81\xa0"  // U+F060
#define ICON_FA_ARROW_RIGHT         "\xef\x81\xa1"  // U+F061
#define ICON_FA_ARROW_UP            "\xef\x81\xa2"  // U+F062
#define ICON_FA_ARROW_DOWN          "\xef\x81\xa3"  // U+F063
#define ICON_FA_CHEVRON_LEFT        "\xef\x81\x93"  // U+F053
#define ICON_FA_CHEVRON_RIGHT       "\xef\x81\x94"  // U+F054
#define ICON_FA_CHEVRON_UP          "\xef\x81\xb7"  // U+F077
#define ICON_FA_CHEVRON_DOWN        "\xef\x81\xb8"  // U+F078
#define ICON_FA_ANGLES_LEFT         "\xef\x84\x80"  // U+F100
#define ICON_FA_ANGLES_RIGHT        "\xef\x84\x81"  // U+F101
#define ICON_FA_ANGLES_UP           "\xef\x84\x82"  // U+F102
#define ICON_FA_ANGLES_DOWN         "\xef\x84\x83"  // U+F103
#define ICON_FA_ANGLE_LEFT          "\xef\x84\x84"  // U+F104
#define ICON_FA_ANGLE_RIGHT         "\xef\x84\x85"  // U+F105

// Actions
#define ICON_FA_PLAY                "\xef\x81\x8b"  // U+F04B
#define ICON_FA_PAUSE               "\xef\x81\x8c"  // U+F04C
#define ICON_FA_STOP                "\xef\x81\x8d"  // U+F04D
#define ICON_FA_FORWARD             "\xef\x81\x8e"  // U+F04E
#define ICON_FA_BACKWARD            "\xef\x81\x8a"  // U+F04A
#define ICON_FA_FORWARD_STEP        "\xef\x81\x91"  // U+F051
#define ICON_FA_BACKWARD_STEP       "\xef\x81\x88"  // U+F048
#define ICON_FA_ROTATE              "\xef\x8b\xb1"  // U+F2F1
#define ICON_FA_ARROWS_ROTATE       "\xef\x80\xa1"  // U+F021
#define ICON_FA_PLUS                "\xef\x81\xa7"  // U+F067
#define ICON_FA_MINUS               "\xef\x81\xa8"  // U+F068
#define ICON_FA_XMARK               "\xef\x80\x8d"  // U+F00D
#define ICON_FA_CHECK               "\xef\x80\x8c"  // U+F00C
#define ICON_FA_MAGNIFYING_GLASS    "\xef\x80\x82"  // U+F002

// Status/Info
#define ICON_FA_CIRCLE_INFO         "\xef\x81\x9a"  // U+F05A
#define ICON_FA_CIRCLE_CHECK        "\xef\x81\x98"  // U+F058
#define ICON_FA_CIRCLE_XMARK        "\xef\x81\x97"  // U+F057
#define ICON_FA_TRIANGLE_EXCLAMATION "\xef\x81\xb1" // U+F071
#define ICON_FA_CIRCLE_EXCLAMATION  "\xef\x81\xaa"  // U+F06A
#define ICON_FA_BUG                 "\xef\x86\x88"  // U+F188
#define ICON_FA_SPINNER             "\xef\x84\x90"  // U+F110

// Panels/Windows - Using verified FA6 icons
#define ICON_FA_DIAGRAM_PROJECT     "\xef\x95\x82"  // U+F542 - sitemap/network diagram
#define ICON_FA_CODE                "\xef\x84\xa1"  // U+F121 - code brackets
#define ICON_FA_TERMINAL            "\xef\x84\xa0"  // U+F120 - terminal
#define ICON_FA_CHART_LINE          "\xef\x88\x81"  // U+F201 - line chart
#define ICON_FA_GAUGE               "\xef\x98\xa4"  // U+F624 - gauge/speedometer
#define ICON_FA_GAUGE_HIGH          "\xef\x98\xa5"  // U+F625 - gauge high
#define ICON_FA_CUBE                "\xef\x86\xb2"  // U+F1B2 - single cube
#define ICON_FA_CUBES               "\xef\x86\xb3"  // U+F1B3 - cubes
#define ICON_FA_SLIDERS             "\xef\x87\x9e"  // U+F1DE - sliders
#define ICON_FA_IMAGE               "\xef\x80\xbe"  // U+F03E - image
#define ICON_FA_IMAGES              "\xef\x8c\x82"  // U+F302 - multiple images
#define ICON_FA_DATABASE            "\xef\x87\x80"  // U+F1C0 - database
#define ICON_FA_LIST_CHECK          "\xef\x82\xae"  // U+F0AE - task list
#define ICON_FA_WALLET              "\xef\x95\x95"  // U+F555 - wallet

// Network/Connection
#define ICON_FA_NETWORK_WIRED       "\xef\x9b\xbf"  // U+F6FF
#define ICON_FA_WIFI                "\xef\x87\xab"  // U+F1EB
#define ICON_FA_GLOBE               "\xef\x82\xac"  // U+F0AC
#define ICON_FA_SERVER              "\xef\x88\xb3"  // U+F233
#define ICON_FA_PLUG                "\xef\x87\xa6"  // U+F1E6
#define ICON_FA_LINK                "\xef\x83\x81"  // U+F0C1
#define ICON_FA_LINK_SLASH          "\xef\x84\xa7"  // U+F127

// Settings/Tools
#define ICON_FA_GEAR                "\xef\x80\x93"  // U+F013
#define ICON_FA_GEARS               "\xef\x82\x85"  // U+F085
#define ICON_FA_WRENCH              "\xef\x82\xad"  // U+F0AD
#define ICON_FA_SCREWDRIVER_WRENCH  "\xef\x9f\x99"  // U+F7D9
#define ICON_FA_PALETTE             "\xef\x94\xbf"  // U+F53F
#define ICON_FA_BRUSH               "\xef\x95\x9d"  // U+F55D
#define ICON_FA_WAND_MAGIC_SPARKLES "\xee\x8b\x8a"  // U+E2CA

// ML/Training specific
#define ICON_FA_BRAIN               "\xef\x97\x9c"  // U+F5DC
#define ICON_FA_MICROCHIP           "\xef\x8b\x9b"  // U+F2DB
#define ICON_FA_LAYER_GROUP         "\xef\x97\xbd"  // U+F5FD
#define ICON_FA_SITEMAP             "\xef\x83\xa8"  // U+F0E8
#define ICON_FA_SQUARE_ROOT_VARIABLE "\xef\x9a\x98" // U+F698
#define ICON_FA_CHART_AREA          "\xef\x87\xbe"  // U+F1FE
#define ICON_FA_CHART_BAR           "\xef\x82\x80"  // U+F080
#define ICON_FA_CHART_PIE           "\xef\x88\x80"  // U+F200
#define ICON_FA_CHART_SIMPLE        "\xee\x91\xb3"  // U+E473

// Misc UI
#define ICON_FA_CIRCLE              "\xef\x84\x91"  // U+F111
#define ICON_FA_SQUARE              "\xef\x83\x88"  // U+F0C8
#define ICON_FA_STAR                "\xef\x80\x85"  // U+F005
#define ICON_FA_HEART               "\xef\x80\x84"  // U+F004
#define ICON_FA_BOOKMARK            "\xef\x80\xae"  // U+F02E
#define ICON_FA_TAG                 "\xef\x80\xab"  // U+F02B
#define ICON_FA_TAGS                "\xef\x80\xac"  // U+F02C
#define ICON_FA_CLOCK               "\xef\x80\x97"  // U+F017
#define ICON_FA_CALENDAR            "\xef\x84\xb3"  // U+F133
#define ICON_FA_USER                "\xef\x80\x87"  // U+F007
#define ICON_FA_USERS               "\xef\x83\x80"  // U+F0C0
#define ICON_FA_LOCK                "\xef\x80\xa3"  // U+F023
#define ICON_FA_LOCK_OPEN           "\xef\x8f\x81"  // U+F3C1
#define ICON_FA_KEY                 "\xef\x82\x84"  // U+F084
#define ICON_FA_QUESTION            "\xef\x84\xa8"  // U+F128
#define ICON_FA_LIGHTBULB           "\xef\x83\xab"  // U+F0EB
#define ICON_FA_HOUSE               "\xef\x80\x95"  // U+F015
#define ICON_FA_HOME                "\xef\x80\x95"  // U+F015

// Direction indicators
#define ICON_FA_CARET_UP            "\xef\x83\x98"  // U+F0D8
#define ICON_FA_CARET_DOWN          "\xef\x83\x97"  // U+F0D7
#define ICON_FA_CARET_LEFT          "\xef\x83\x99"  // U+F0D9
#define ICON_FA_CARET_RIGHT         "\xef\x83\x9a"  // U+F0DA

// Sorting
#define ICON_FA_SORT                "\xef\x83\x9c"  // U+F0DC
#define ICON_FA_SORT_UP             "\xef\x83\x9e"  // U+F0DE
#define ICON_FA_SORT_DOWN           "\xef\x83\x9d"  // U+F0DD

// Media
#define ICON_FA_MUSIC               "\xef\x80\x81"  // U+F001
#define ICON_FA_VIDEO               "\xef\x80\xbd"  // U+F03D
#define ICON_FA_FILM                "\xef\x80\x88"  // U+F008
#define ICON_FA_VOLUME_HIGH         "\xef\x80\xa8"  // U+F028
#define ICON_FA_VOLUME_LOW          "\xef\x80\xa7"  // U+F027
#define ICON_FA_VOLUME_OFF          "\xef\x80\xa6"  // U+F026
#define ICON_FA_VOLUME_XMARK        "\xef\x9a\xa9"  // U+F6A9

// Blockchain/Crypto
#define ICON_FA_COINS               "\xef\x94\x9e"  // U+F51E
#define ICON_FA_MONEY_BILL          "\xef\x83\x96"  // U+F0D6
#define ICON_FA_CREDIT_CARD         "\xef\x82\x9d"  // U+F09D
#define ICON_FA_BITCOIN_SIGN        "\xee\x82\xb4"  // U+E0B4
#define ICON_FA_ETHEREUM            "\xef\x90\xae"  // U+F42E

// Common/Simple shapes
#define ICON_FA_CIRCLE_DOT          "\xef\x86\x92"  // U+F192
#define ICON_FA_SQUARE_CHECK        "\xef\x85\x8a"  // U+F14A
#define ICON_FA_SQUARE_MINUS        "\xef\x85\x86"  // U+F146
#define ICON_FA_SQUARE_PLUS         "\xef\x83\xbe"  // U+F0FE

// Menu bar icons
#define ICON_FA_BOOK                ""  // U+F02D
#define ICON_FA_ARROW_UP_FROM_BRACKET ""  // U+E09A
#define ICON_FA_ARROW_RIGHT_FROM_BRACKET ""  // U+F08B

