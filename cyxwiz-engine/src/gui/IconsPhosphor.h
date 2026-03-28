// Phosphor Icons - Icon Font Definitions
// https://phosphoricons.com/
// MIT License
//
// Node Editor Icon Pack - Flexible icon family with multiple weights

#pragma once

#define FONT_ICON_FILE_NAME_PH "phosphor.ttf"

// Icon range for Phosphor (Private Use Area)
#define ICON_MIN_PH 0xE000
#define ICON_MAX_PH 0xF8FF

// UTF-8 encoding: For U+XXXX where XXXX is 0x0800-0xFFFF
// Byte1 = 0xE0 | ((cp >> 12) & 0x0F)
// Byte2 = 0x80 | ((cp >> 6) & 0x3F)
// Byte3 = 0x80 | (cp & 0x3F)

// ============================================
// Data & Database
// ============================================
#define ICON_PH_DATABASE        "\xee\xa4\x88"  // database
#define ICON_PH_HARD_DRIVES     "\xee\xa8\x90"  // hard-drives
#define ICON_PH_CLOUD           "\xee\xa3\x98"  // cloud
#define ICON_PH_SERVER          "\xee\xac\x80"  // server

// ============================================
// Files & Documents
// ============================================
#define ICON_PH_FILE            "\xee\xa6\x80"  // file
#define ICON_PH_FILE_TEXT       "\xee\xa6\x90"  // file-text
#define ICON_PH_FILE_CODE       "\xee\xa6\x84"  // file-code
#define ICON_PH_FILE_XLS        "\xee\xa6\x9c"  // file-xls
#define ICON_PH_FILE_CSV        "\xee\xa6\x88"  // file-csv
#define ICON_PH_FOLDER          "\xee\xa7\x80"  // folder
#define ICON_PH_DOWNLOAD        "\xee\xa5\x80"  // download
#define ICON_PH_UPLOAD          "\xee\xae\x80"  // upload
#define ICON_PH_EXPORT          "\xee\xa5\x88"  // export

// ============================================
// Neural Network & AI
// ============================================
#define ICON_PH_BRAIN           "\xee\xa3\x80"  // brain
#define ICON_PH_CPU             "\xee\xa4\x80"  // cpu
#define ICON_PH_ROBOT           "\xee\xab\x88"  // robot
#define ICON_PH_SPARKLE         "\xee\xac\x88"  // sparkle
#define ICON_PH_MAGIC_WAND      "\xee\xa9\x80"  // magic-wand

// ============================================
// Layers & Structure
// ============================================
#define ICON_PH_STACK           "\xee\xac\x90"  // stack
#define ICON_PH_CUBE            "\xee\xa4\x90"  // cube
#define ICON_PH_CUBES           "\xee\xa4\x94"  // cubes-four
#define ICON_PH_SQUARES         "\xee\xac\x98"  // squares-four
#define ICON_PH_GRID_FOUR       "\xee\xa8\x84"  // grid-four

// ============================================
// Math & Functions
// ============================================
#define ICON_PH_FUNCTION        "\xee\xa7\x88"  // function
#define ICON_PH_CALCULATOR      "\xee\xa3\x84"  // calculator
#define ICON_PH_PLUS            "\xee\xaa\x80"  // plus
#define ICON_PH_MINUS           "\xee\xa9\x88"  // minus
#define ICON_PH_X               "\xee\xae\x88"  // x
#define ICON_PH_DIVIDE          "\xee\xa4\x98"  // divide
#define ICON_PH_PERCENT         "\xee\xa9\xa0"  // percent
#define ICON_PH_SIGMA           "\xee\xac\x84"  // sigma
#define ICON_PH_EQUALS          "\xee\xa5\x90"  // equals

// ============================================
// Charts & Analytics
// ============================================
#define ICON_PH_CHART_BAR       "\xee\xa3\x90"  // chart-bar
#define ICON_PH_CHART_LINE      "\xee\xa3\x94"  // chart-line
#define ICON_PH_CHART_PIE       "\xee\xa3\x98"  // chart-pie
#define ICON_PH_CHART_SCATTER   "\xee\xa3\x9c"  // chart-scatter
#define ICON_PH_TREND_UP        "\xee\xad\x90"  // trend-up
#define ICON_PH_TREND_DOWN      "\xee\xad\x88"  // trend-down
#define ICON_PH_PULSE           "\xee\xaa\x90"  // pulse

// ============================================
// Arrows & Navigation
// ============================================
#define ICON_PH_ARROW_UP        "\xee\xa2\x90"  // arrow-up
#define ICON_PH_ARROW_DOWN      "\xee\xa2\x88"  // arrow-down
#define ICON_PH_ARROW_LEFT      "\xee\xa2\x8c"  // arrow-left
#define ICON_PH_ARROW_RIGHT     "\xee\xa2\x94"  // arrow-right
#define ICON_PH_SHUFFLE         "\xee\xac\x80"  // shuffle
#define ICON_PH_REPEAT          "\xee\xab\x80"  // repeat
#define ICON_PH_ARROWS_MERGE    "\xee\xa2\xa0"  // arrows-merge
#define ICON_PH_ARROWS_SPLIT    "\xee\xa2\xa4"  // arrows-split
#define ICON_PH_SWAP            "\xee\xac\xa0"  // swap

// ============================================
// Actions & Operations
// ============================================
#define ICON_PH_FUNNEL          "\xee\xa7\x90"  // funnel
#define ICON_PH_SORT_ASCENDING  "\xee\xac\x94"  // sort-ascending
#define ICON_PH_SORT_DESCENDING "\xee\xac\x98"  // sort-descending
#define ICON_PH_MAGNIFYING_GLASS "\xee\xa9\x84" // magnifying-glass
#define ICON_PH_ARROWS_CLOCKWISE "\xee\xa2\x98" // arrows-clockwise
#define ICON_PH_TRASH           "\xee\xad\x80"  // trash
#define ICON_PH_PENCIL          "\xee\xa9\xa8"  // pencil
#define ICON_PH_COPY            "\xee\xa4\x84"  // copy
#define ICON_PH_SCISSORS        "\xee\xab\x90"  // scissors
#define ICON_PH_CROP            "\xee\xa4\x88"  // crop

// ============================================
// Machine Learning
// ============================================
#define ICON_PH_TREE_STRUCTURE  "\xee\xad\x84"  // tree-structure
#define ICON_PH_FLOW_ARROW      "\xee\xa6\xa0"  // flow-arrow
#define ICON_PH_GIT_BRANCH      "\xee\xa7\x94"  // git-branch
#define ICON_PH_GIT_MERGE       "\xee\xa7\x98"  // git-merge
#define ICON_PH_GRAPH           "\xee\xa8\x80"  // graph

// ============================================
// Shapes & UI
// ============================================
#define ICON_PH_CIRCLE          "\xee\xa3\xa0"  // circle
#define ICON_PH_SQUARE          "\xee\xac\x9c"  // square
#define ICON_PH_TRIANGLE        "\xee\xad\x98"  // triangle
#define ICON_PH_CHECK           "\xee\xa3\x88"  // check
#define ICON_PH_CHECK_CIRCLE    "\xee\xa3\x8c"  // check-circle

// ============================================
// Signals & Waves
// ============================================
#define ICON_PH_WAVEFORM        "\xee\xad\xa0"  // waveform
#define ICON_PH_WAVE_SINE       "\xee\xad\xa4"  // wave-sine
#define ICON_PH_WAVE_SAWTOOTH   "\xee\xad\xa8"  // wave-sawtooth
#define ICON_PH_FADERS          "\xee\xa5\xa0"  // faders
#define ICON_PH_SLIDERS         "\xee\xac\x8c"  // sliders

// ============================================
// Text & Typography
// ============================================
#define ICON_PH_TEXT_T          "\xee\xad\x8c"  // text-t
#define ICON_PH_TEXT_AA         "\xee\xad\x88"  // text-aa
#define ICON_PH_TEXT_ALIGN_LEFT "\xee\xad\x94"  // text-align-left
#define ICON_PH_LIST            "\xee\xa9\x90"  // list
#define ICON_PH_HASH            "\xee\xa8\x88"  // hash
#define ICON_PH_TAG             "\xee\xac\xa4"  // tag

// ============================================
// Media & Images
// ============================================
#define ICON_PH_IMAGE           "\xee\xa8\x94"  // image
#define ICON_PH_MUSIC_NOTE      "\xee\xa9\x98"  // music-note
#define ICON_PH_VIDEO           "\xee\xad\xac"  // video-camera

// ============================================
// Settings & Config
// ============================================
#define ICON_PH_GEAR            "\xee\xa7\x9c"  // gear
#define ICON_PH_WRENCH          "\xee\xad\xb0"  // wrench
#define ICON_PH_CODE            "\xee\xa3\xb0"  // code
#define ICON_PH_TERMINAL        "\xee\xac\xa8"  // terminal
#define ICON_PH_BUG             "\xee\xa3\x9c"  // bug
#define ICON_PH_PLUG            "\xee\xaa\x88"  // plug

// ============================================
// Table Operations
// ============================================
#define ICON_PH_TABLE           "\xee\xac\xac"  // table
#define ICON_PH_ROWS            "\xee\xab\x84"  // rows
#define ICON_PH_COLUMNS         "\xee\xa3\xb8"  // columns

// ============================================
// Misc & Default
// ============================================
#define ICON_PH_LINK            "\xee\xa9\x94"  // link
#define ICON_PH_LINK_BREAK      "\xee\xa9\x9c"  // link-break
#define ICON_PH_CROSSHAIR       "\xee\xa4\x9c"  // crosshair
#define ICON_PH_TARGET          "\xee\xac\xb0"  // target
#define ICON_PH_LIGHTBULB       "\xee\xa8\x9c"  // lightbulb
#define ICON_PH_LIGHTNING       "\xee\xa8\xa0"  // lightning
#define ICON_PH_ATOM            "\xee\xa2\x84"  // atom
#define ICON_PH_DOTS_NINE       "\xee\xa5\x84"  // dots-nine
