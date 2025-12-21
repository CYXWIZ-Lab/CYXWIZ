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
#define ICON_FA_FILE_EXPORT         "\xef\x95\xae"  // U+F56E - file export
#define ICON_FA_FILE_IMPORT         "\xef\x95\xaf"  // U+F56F - file import
#define ICON_FA_FOLDER              "\xef\x81\xbb"  // U+F07B
#define ICON_FA_FOLDER_OPEN         "\xef\x81\xbc"  // U+F07C
#define ICON_FA_FOLDER_PLUS         "\xef\x99\x9e"  // U+F65E
#define ICON_FA_FLOPPY_DISK         "\xef\x83\x87"  // U+F0C7
#define ICON_FA_DOWNLOAD            "\xef\x80\x99"  // U+F019

// Edit operations
#define ICON_FA_COPY                "\xef\x83\x85"  // U+F0C5
#define ICON_FA_PASTE               "\xef\x83\xaa"  // U+F0EA
#define ICON_FA_CLIPBOARD           "\xef\x8c\xa8"  // U+F328 - clipboard
#define ICON_FA_SCISSORS            "\xef\x83\x84"  // U+F0C4
#define ICON_FA_TRASH               "\xef\x87\xb8"  // U+F1F8
#define ICON_FA_TRASH_CAN           "\xef\x8b\xad"  // U+F2ED
#define ICON_FA_PEN                 "\xef\x8c\x84"  // U+F304
#define ICON_FA_PENCIL              "\xef\x8c\x83"  // U+F303
#define ICON_FA_ERASER              "\xef\x84\xad"  // U+F12D
#define ICON_FA_PARAGRAPH           "\xef\x87\x9d"  // U+F1DD
#define ICON_FA_ROTATE_LEFT         "\xef\x8b\xaa"  // U+F2EA
#define ICON_FA_ROTATE_RIGHT        "\xef\x8b\xb9"  // U+F2F9
#define ICON_FA_OBJECT_GROUP        "\xef\x89\x87"  // U+F247 - group selection
#define ICON_FA_OBJECT_UNGROUP      "\xef\x89\x88"  // U+F248 - ungroup
#define ICON_FA_RIGHT_LEFT          "\xef\x8d\xa2"  // U+F362 - exchange/swap
#define ICON_FA_COMMENT             "\xef\x81\xb5"  // U+F075 - comment
#define ICON_FA_COMMENTS            "\xef\x82\x86"  // U+F086 - comments/block

// View/Layout
#define ICON_FA_TABLE               "\xef\x83\x8e"  // U+F0CE
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

// Model Analysis icons
#define ICON_FA_LIST                "\xef\x80\xba"  // U+F03A - list
#define ICON_FA_CALCULATOR          "\xef\x87\xac"  // U+F1EC - calculator
#define ICON_FA_MAGNIFYING_GLASS_CHART "\xee\x94\xa2"  // U+E522 - magnifying glass with chart
#define ICON_FA_MAGNIFYING_GLASS_PLUS "\xef\x80\x8e"  // U+F00E - zoom in
#define ICON_FA_MAGNIFYING_GLASS_MINUS "\xef\x80\x90"  // U+F010 - zoom out
#define ICON_FA_ARROWS_LEFT_RIGHT   "\xef\x8c\xbe"  // U+F33E - left-right arrows
#define ICON_FA_HASHTAG             "\x23"          // # - hashtag
#define ICON_FA_RIGHT_TO_BRACKET    "\xef\x8b\xb6"  // U+F2F6 - right to bracket
#define ICON_FA_RIGHT_FROM_BRACKET  "\xef\x8b\xb5"  // U+F2F5 - right from bracket
#define ICON_FA_HARD_DRIVE          "\xef\x82\xa0"  // U+F0A0 - hard drive
#define ICON_FA_WEIGHT_SCALE        "\xef\x92\x96"  // U+F496 - weight scale
#define ICON_FA_GRADUATION_CAP      "\xef\x86\x9d"  // U+F19D - graduation cap
#define ICON_FA_SCALE_BALANCED      "\xef\x89\x8e"  // U+F24E - scale balanced
#define ICON_FA_LIGHTBULB           "\xef\x83\xab"  // U+F0EB - lightbulb
#define ICON_FA_STETHOSCOPE         "\xef\x83\xb1"  // U+F0F1 - stethoscope
#define ICON_FA_ARROW_TREND_UP      "\xee\x82\x98"  // U+E098 - arrow trend up
#define ICON_FA_LIST_CHECK          "\xef\x82\xae"  // U+F0AE - list check
#define ICON_FA_TABLE_CELLS         "\xef\x83\x8e"  // U+F0CE - table cells (same as TABLE)
#define ICON_FA_PERCENT             "\x25"          // '%' character
#define ICON_FA_REPEAT              "\xef\x8d\xa3"  // U+F363 - repeat/loop
#define ICON_FA_COG                 "\xef\x80\x93"  // U+F013 - cog (same as GEAR)

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
#define ICON_FA_LIST_UL             "\xef\x83\x8a"  // U+F0CA - unordered list
#define ICON_FA_WALLET              "\xef\x95\x95"  // U+F555 - wallet

// Network/Connection
#define ICON_FA_NETWORK_WIRED       "\xef\x9b\xbf"  // U+F6FF
#define ICON_FA_WIFI                "\xef\x87\xab"  // U+F1EB
#define ICON_FA_GLOBE               "\xef\x82\xac"  // U+F0AC
#define ICON_FA_SERVER              "\xef\x88\xb3"  // U+F233
#define ICON_FA_PLUG                "\xef\x87\xa6"  // U+F1E6
#define ICON_FA_LINK                "\xef\x83\x81"  // U+F0C1
#define ICON_FA_LINK_SLASH          "\xef\x84\xa7"  // U+F127
#define ICON_FA_ROCKET              "\xef\x84\xb5"  // U+F135 - rocket/deploy

// Search/Filter
#define ICON_FA_FILTER              "\xef\x82\xb0"  // U+F0B0 - filter

// Settings/Tools
#define ICON_FA_GEAR                "\xef\x80\x93"  // U+F013
#define ICON_FA_GEARS               "\xef\x82\x85"  // U+F085
#define ICON_FA_CODE                "\xef\x84\xa1"  // U+F121 - code/scripting
#define ICON_FA_KEYBOARD            "\xef\x84\x9c"  // U+F11C - keyboard
#define ICON_FA_WRENCH              "\xef\x82\xad"  // U+F0AD
#define ICON_FA_SCREWDRIVER_WRENCH  "\xef\x9f\x99"  // U+F7D9
#define ICON_FA_PALETTE             "\xef\x94\xbf"  // U+F53F
#define ICON_FA_BRUSH               "\xef\x95\x9d"  // U+F55D
#define ICON_FA_WAND_MAGIC_SPARKLES "\xee\x8b\x8a"  // U+E2CA

// Memory & Performance
#define ICON_FA_MEMORY              "\xef\x94\xb8"  // U+F538
#define ICON_FA_DESKTOP             "\xef\x8e\x90"  // U+F390
#define ICON_FA_RECYCLE             "\xef\x86\xb8"  // U+F1B8
#define ICON_FA_CLOCK_ROTATE_LEFT   "\xef\x87\x9a"  // U+F1DA - history/restore
#define ICON_FA_FLASK               "\xef\x83\x83"  // U+F0C3 - flask/test

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
#define ICON_FA_CHART_COLUMN        "\xee\x83\xa3"  // U+E0E3 - column chart

// Clustering icons
#define ICON_FA_BULLSEYE            "\xef\x85\x80"  // U+F140 - bullseye/target
#define ICON_FA_CIRCLE_NODES        "\xee\x93\xa8"  // U+E4E8 - network nodes
#define ICON_FA_CHART_SCATTER       "\xee\x94\xa4"  // U+E524 - scatter chart
#define ICON_FA_FIRE                "\xef\x81\xad"  // U+F06D - fire
#define ICON_FA_DNA                 "\xef\x91\xb1"  // U+F471 - DNA
#define ICON_FA_RANKING_STAR        "\xee\x95\xb0"  // U+E570 - ranking star (FA6 Pro alternative: use STAR)
#define ICON_FA_TROPHY              "\xef\x82\x91"  // U+F091 - trophy
#define ICON_FA_CROWN               "\xef\x94\xa1"  // U+F521 - crown
#define ICON_FA_TIMES               "\xef\x80\x8d"  // U+F00D - times (same as XMARK)
#define ICON_FA_BOLT                "\xef\x83\xa7"  // U+F0E7 - lightning bolt
#define ICON_FA_SHUFFLE             "\xef\x81\xb4"  // U+F074 - shuffle
#define ICON_FA_UPLOAD              "\xef\x82\x93"  // U+F093 - upload

// Signal Processing icons (Phase 8)
#define ICON_FA_WAVE_SQUARE         "\xef\xa0\xbe"  // U+F83E - wave square
#define ICON_FA_SIGNAL              "\xef\x80\x92"  // U+F012 - signal bars
#define ICON_FA_ASTERISK            "\x2a"          // U+002A - asterisk (basic)
#define ICON_FA_WATER               "\xef\x9d\xb3"  // U+F773 - water
#define ICON_FA_MOUNTAIN            "\xef\x9b\xbc"  // U+F6FC - mountain
#define ICON_FA_BORDER_ALL          "\xef\xa1\x8c"  // U+F84C - border all

// Utilities icons (Phase 12)
#define ICON_FA_FINGERPRINT         "\xef\x95\xb7"  // U+F577 - fingerprint (hash)
#define ICON_FA_TOOLBOX             "\xef\x95\x92"  // U+F552 - toolbox
#define ICON_FA_ALIGN_LEFT          "\xef\x80\xb6"  // U+F036 - align left
#define ICON_FA_ALIGN_RIGHT         "\xef\x80\xb8"  // U+F038 - align right
#define ICON_FA_PI                  "\xef\xa1\x87"  // U+F87F - pi symbol (FA6 Pro) - approximation
#define ICON_FA_CHECK_DOUBLE        "\xef\x95\xa0"  // U+F560 - double check mark
#define ICON_FA_BRACKETS_CURLY      "\xef\x9f\xa6"  // U+F7A6 - curly brackets for JSON
#define ICON_FA_CODE_COMPARE        "\xee\x84\xbf"  // U+E13F - code compare
#define ICON_FA_INFO                "\xef\x84\xa9"  // U+F129 - info
#define ICON_FA_LIST_OL             "\xef\x83\x8b"  // U+F0CB - ordered list

// Optimization & Calculus icons (Phase 9)
#define ICON_FA_HOURGLASS           "\xef\x89\x94"  // U+F254 - hourglass
#define ICON_FA_ARROW_TREND_DOWN    "\xee\x82\x97"  // U+E097 - arrow trend down
#define ICON_FA_ARROW_DOWN_LONG     "\xef\x85\xb5"  // U+F175 - long arrow down
#define ICON_FA_HILL_ROCKSLIDE      "\xee\x94\xa8"  // U+E508 - hill rockslide (for convexity)
#define ICON_FA_ROUTE               "\xef\x93\x97"  // U+F4D7 - route/path
#define ICON_FA_INFINITY            "\xef\x94\xb4"  // U+F534 - infinity
#define ICON_FA_BALANCE_SCALE_LEFT  "\xef\x94\x95"  // U+F515 - unbalanced scale
#define ICON_FA_DIVIDE              "\xef\x94\xa9"  // U+F529 - divide
#define ICON_FA_SUPERSCRIPT         "\xef\x84\xab"  // U+F12B - superscript (for derivatives)
#define ICON_FA_SUBSCRIPT           "\xef\x84\xac"  // U+F12C - subscript
#define ICON_FA_INTEGRAL            "\xef\x99\x87"  // U+F667 - integral (custom, use sum for now)
#define ICON_FA_SIGMA               "\xef\x9a\xa8"  // U+F6A8 - sigma/summation
#define ICON_FA_LESS_THAN_EQUAL     "\xef\x94\xb7"  // U+F537 - less than or equal
#define ICON_FA_GREATER_THAN_EQUAL  "\xef\x94\xb2"  // U+F532 - greater than or equal
#define ICON_FA_NOT_EQUAL           "\xef\x94\xbe"  // U+F53E - not equal
#define ICON_FA_EQUALS              "\x3d"          // = - equals sign
#define ICON_FA_CROSSHAIRS          "\xef\x81\x9b"  // U+F05B - crosshairs/target
#define ICON_FA_DICE                "\xef\x94\xa2"  // U+F522 - dice (for random sampling)
#define ICON_FA_SQUARE_POLL_VERTICAL "\xef\x9d\x81" // U+F681 - vertical bar chart
#define ICON_FA_DRAW_POLYGON        "\xef\x97\xae"  // U+F5EE - draw polygon
#define ICON_FA_ARROWS_UP_DOWN_LEFT_RIGHT "\xef\x82\xb2" // U+F0B2 - arrows four directions
#define ICON_FA_WAVE_SINE           "\xef\xa0\xbd"  // U+F83D - wave sine (using wave-square similar)
#define ICON_FA_EXCLAMATION         "\x21"          // U+0021 - exclamation mark

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
#define ICON_FA_FONT                "\xef\x80\xb1"  // U+F031 - font
#define ICON_FA_TOGGLE_ON           "\xef\x88\x85"  // U+F205 - toggle on

// Face icons (for sentiment analysis)
#define ICON_FA_FACE_SMILE          "\xef\x84\x98"  // U+F118 - smile
#define ICON_FA_FACE_FROWN          "\xef\x84\x99"  // U+F119 - frown
#define ICON_FA_FACE_MEH            "\xef\x84\x9a"  // U+F11A - meh/neutral

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
#define ICON_FA_DOLLAR_SIGN         "\x24"          // U+0024 (ASCII $)
#define ICON_FA_GIFT                "\xef\x81\xab"  // U+F06B

// Cloud/Network
#define ICON_FA_CLOUD               "\xef\x83\x82"  // U+F0C2
#define ICON_FA_GEM                 "ï¥"  // U+F3A5 - gem/diamond for SOL
#define ICON_FA_PAPER_PLANE         "ï"  // U+F1D8 - paper plane for sending
#define ICON_FA_UNLOCK              "ï"  // U+F09C - unlock
#define ICON_FA_ARROW_UP_RIGHT_FROM_SQUARE "ï"  // U+F08E - external link
#define ICON_FA_CIRCLE_QUESTION     "ï"  // U+F059 - question mark in circle

// Common/Simple shapes
#define ICON_FA_CIRCLE_DOT          "\xef\x86\x92"  // U+F192
#define ICON_FA_SQUARE_CHECK        "\xef\x85\x8a"  // U+F14A
#define ICON_FA_SQUARE_MINUS        "\xef\x85\x86"  // U+F146
#define ICON_FA_SQUARE_PLUS         "\xef\x83\xbe"  // U+F0FE

// Menu bar icons
#define ICON_FA_BOOK                ""  // U+F02D
#define ICON_FA_ARROW_UP_FROM_BRACKET ""  // U+E09A
#define ICON_FA_ARROW_RIGHT_FROM_BRACKET ""  // U+F08B

