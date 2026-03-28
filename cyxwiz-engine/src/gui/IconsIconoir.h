// Iconoir Icons - Icon Font Definitions
// https://iconoir.com/
// MIT License
//
// Node Editor Icon Pack - Simple, consistent stroke icons

#pragma once

#define FONT_ICON_FILE_NAME_IO "iconoir.ttf"

// Icon range for Iconoir (Private Use Area)
#define ICON_MIN_IO 0xE900
#define ICON_MAX_IO 0xEFFF

// UTF-8 encoding: For U+XXXX where XXXX is 0x0800-0xFFFF
// Byte1 = 0xE0 | ((cp >> 12) & 0x0F)
// Byte2 = 0x80 | ((cp >> 6) & 0x3F)
// Byte3 = 0x80 | (cp & 0x3F)

// ============================================
// Data & Database
// ============================================
#define ICON_IO_DATABASE        "\xee\xa4\x80"  // database
#define ICON_IO_DATABASE_EXPORT "\xee\xa4\x84"  // database-export
#define ICON_IO_SERVER          "\xee\xab\x80"  // server
#define ICON_IO_CLOUD           "\xee\xa3\x90"  // cloud-upload

// ============================================
// Files & Documents
// ============================================
#define ICON_IO_FILE            "\xee\xa5\x80"  // page
#define ICON_IO_FILE_EDIT       "\xee\xa5\x88"  // page-edit
#define ICON_IO_FILE_SEARCH     "\xee\xa5\x8c"  // page-search
#define ICON_IO_FOLDER          "\xee\xa5\x90"  // folder
#define ICON_IO_IMPORT          "\xee\xa6\x98"  // import
#define ICON_IO_DOWNLOAD        "\xee\xa4\x90"  // download

// ============================================
// Neural Network & AI
// ============================================
#define ICON_IO_BRAIN           "\xee\xa3\x80"  // brain
#define ICON_IO_CPU             "\xee\xa4\x88"  // cpu
#define ICON_IO_SPARKS          "\xee\xab\x88"  // sparks
#define ICON_IO_MAGIC_WAND      "\xee\xa8\x80"  // magic-wand

// ============================================
// Layers & Structure
// ============================================
#define ICON_IO_LAYERS          "\xee\xa7\x80"  // layers
#define ICON_IO_BOX             "\xee\xa3\x84"  // box
#define ICON_IO_CUBE            "\xee\xa4\x84"  // cube
#define ICON_IO_VIEW_GRID       "\xee\xac\x80"  // view-grid
#define ICON_IO_GRID            "\xee\xa6\x80"  // grid

// ============================================
// Math & Functions
// ============================================
#define ICON_IO_CALCULATOR      "\xee\xa3\x88"  // calculator
#define ICON_IO_PLUS            "\xee\xa9\x80"  // plus
#define ICON_IO_MINUS           "\xee\xa8\x88"  // minus
#define ICON_IO_DIVIDE          "\xee\xa4\x94"  // divide
#define ICON_IO_SIGMA           "\xee\xab\x84"  // sigma-function
#define ICON_IO_PERCENTAGE      "\xee\xa9\x84"  // percentage

// ============================================
// Charts & Analytics
// ============================================
#define ICON_IO_STATS_UP        "\xee\xab\x90"  // stats-up-square
#define ICON_IO_STATS_DOWN      "\xee\xab\x94"  // stats-down-square
#define ICON_IO_GRAPH_UP        "\xee\xa6\x84"  // graph-up
#define ICON_IO_GRAPH_DOWN      "\xee\xa6\x88"  // graph-down
#define ICON_IO_PIE_CHART       "\xee\xa9\x88"  // pie-chart
#define ICON_IO_HISTOGRAM       "\xee\xa6\x90"  // histogram

// ============================================
// Arrows & Navigation
// ============================================
#define ICON_IO_ARROW_UP        "\xee\xa2\x90"  // arrow-up
#define ICON_IO_ARROW_DOWN      "\xee\xa2\x88"  // arrow-down
#define ICON_IO_ARROW_LEFT      "\xee\xa2\x8c"  // arrow-left
#define ICON_IO_ARROW_RIGHT     "\xee\xa2\x94"  // arrow-right
#define ICON_IO_SHUFFLE         "\xee\xab\x8c"  // shuffle
#define ICON_IO_REFRESH         "\xee\xaa\x80"  // refresh
#define ICON_IO_UNDO            "\xee\xac\x84"  // undo
#define ICON_IO_REDO            "\xee\xaa\x84"  // redo

// ============================================
// Actions & Operations
// ============================================
#define ICON_IO_FILTER          "\xee\xa5\x94"  // filter
#define ICON_IO_SORT            "\xee\xab\x98"  // sort
#define ICON_IO_SEARCH          "\xee\xaa\x88"  // search
#define ICON_IO_TRASH           "\xee\xac\x88"  // trash
#define ICON_IO_EDIT            "\xee\xa4\xa0"  // edit
#define ICON_IO_COPY            "\xee\xa4\x80"  // copy
#define ICON_IO_CUT             "\xee\xa4\x8c"  // cut
#define ICON_IO_CROP            "\xee\xa4\x88"  // crop

// ============================================
// Machine Learning
// ============================================
#define ICON_IO_NETWORK         "\xee\xa8\x90"  // network
#define ICON_IO_GIT_BRANCH      "\xee\xa5\xb8"  // git-branch
#define ICON_IO_GIT_MERGE       "\xee\xa5\xbc"  // git-merge
#define ICON_IO_TREE            "\xee\xac\x8c"  // tree

// ============================================
// Shapes & UI
// ============================================
#define ICON_IO_CIRCLE          "\xee\xa3\x94"  // circle
#define ICON_IO_SQUARE          "\xee\xab\x9c"  // square
#define ICON_IO_TRIANGLE        "\xee\xac\x90"  // triangle
#define ICON_IO_CHECK           "\xee\xa3\x9c"  // check
#define ICON_IO_CHECK_CIRCLE    "\xee\xa3\xa0"  // check-circle

// ============================================
// Signals & Waves
// ============================================
#define ICON_IO_WAVE            "\xee\xac\x94"  // sine-wave
#define ICON_IO_SOUND_HIGH      "\xee\xab\xa0"  // sound-high
#define ICON_IO_TUNNING         "\xee\xac\x98"  // tuning

// ============================================
// Text & Typography
// ============================================
#define ICON_IO_TEXT            "\xee\xab\xb8"  // text
#define ICON_IO_ALIGN_LEFT      "\xee\xa2\x80"  // align-left
#define ICON_IO_LIST            "\xee\xa7\x88"  // list
#define ICON_IO_HASHTAG         "\xee\xa6\x8c"  // hashtag
#define ICON_IO_TAG             "\xee\xab\xbc"  // tag

// ============================================
// Media & Images
// ============================================
#define ICON_IO_IMAGE           "\xee\xa6\x94"  // media-image
#define ICON_IO_MUSIC           "\xee\xa8\x94"  // music
#define ICON_IO_VIDEO           "\xee\xac\x9c"  // video-camera

// ============================================
// Settings & Config
// ============================================
#define ICON_IO_SETTINGS        "\xee\xab\xa4"  // settings
#define ICON_IO_TOOLS           "\xee\xac\xa0"  // tools
#define ICON_IO_CODE            "\xee\xa3\xb0"  // code
#define ICON_IO_TERMINAL        "\xee\xac\xa4"  // terminal
#define ICON_IO_BUG             "\xee\xa3\x8c"  // bug
#define ICON_IO_PLUG            "\xee\xa9\x8c"  // plug

// ============================================
// Table Operations
// ============================================
#define ICON_IO_TABLE           "\xee\xab\xac"  // table
#define ICON_IO_TABLE_ROWS      "\xee\xab\xb0"  // table-rows
#define ICON_IO_VIEW_COLUMNS    "\xee\xac\xa8"  // view-columns
#define ICON_IO_COMBINE         "\xee\xa3\xb8"  // combine

// ============================================
// Misc & Default
// ============================================
#define ICON_IO_LINK            "\xee\xa7\x8c"  // link
#define ICON_IO_UNLINK          "\xee\xac\xac"  // link-off
#define ICON_IO_TARGET          "\xee\xab\xb4"  // target
#define ICON_IO_FOCUS           "\xee\xa5\xa0"  // focus
#define ICON_IO_FLASH           "\xee\xa5\x98"  // flash
#define ICON_IO_LIGHTBULB       "\xee\xa7\x90"  // light-bulb
#define ICON_IO_APPS            "\xee\xa2\x84"  // app-window
