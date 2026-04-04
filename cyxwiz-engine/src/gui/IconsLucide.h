// Lucide Icons - Icon Font Definitions
// https://lucide.dev/
// ISC License (fork of Feather Icons)
//
// Node Editor Icon Pack - Clean minimal stroke style

#pragma once

#define FONT_ICON_FILE_NAME_LU "lucide.ttf"

// Icon range for Lucide
#define ICON_MIN_LU 0xE900
#define ICON_MAX_LU 0xEFFF

// UTF-8 encoding: For U+XXXX where XXXX is 0x0800-0xFFFF
// Byte1 = 0xE0 | ((cp >> 12) & 0x0F)
// Byte2 = 0x80 | ((cp >> 6) & 0x3F)
// Byte3 = 0x80 | (cp & 0x3F)

// ============================================
// Data & Database
// ============================================
#define ICON_LU_DATABASE        "\xee\xa4\x80"  // U+E900 - database
#define ICON_LU_HARD_DRIVE      "\xee\xa6\x90"  // U+E990 - hard-drive
#define ICON_LU_SERVER          "\xee\xab\x80"  // U+EAC0 - server
#define ICON_LU_CLOUD           "\xee\xa3\x98"  // U+E8D8 - cloud

// ============================================
// Files & Documents
// ============================================
#define ICON_LU_FILE            "\xee\xa5\x88"  // U+E948 - file
#define ICON_LU_FILE_TEXT       "\xee\xa5\x98"  // U+E958 - file-text
#define ICON_LU_FILE_CODE       "\xee\xa5\x8c"  // U+E94C - file-code
#define ICON_LU_FILE_SPREADSHEET "\xee\xa5\x94" // U+E954 - file-spreadsheet
#define ICON_LU_FILE_JSON       "\xee\xa5\x90"  // U+E950 - file-json
#define ICON_LU_FOLDER          "\xee\xa5\xa0"  // U+E960 - folder
#define ICON_LU_IMPORT          "\xee\xa6\xa0"  // U+E9A0 - import
#define ICON_LU_DOWNLOAD        "\xee\xa4\x98"  // U+E918 - download

// ============================================
// Neural Network & AI
// ============================================
#define ICON_LU_BRAIN           "\xee\xa3\x80"  // U+E8C0 - brain
#define ICON_LU_CPU             "\xee\xa4\x88"  // U+E908 - cpu
#define ICON_LU_CIRCUIT_BOARD   "\xee\xa3\x90"  // U+E8D0 - circuit-board
#define ICON_LU_SPARKLES        "\xee\xab\x90"  // U+EAD0 - sparkles
#define ICON_LU_BOT             "\xee\xa3\x88"  // U+E8C8 - bot

// ============================================
// Layers & Structure
// ============================================
#define ICON_LU_LAYERS          "\xee\xa7\x88"  // U+E9C8 - layers
#define ICON_LU_BOX             "\xee\xa3\x84"  // U+E8C4 - box
#define ICON_LU_BOXES           "\xee\xa3\x85"  // U+E8C5 - boxes
#define ICON_LU_LAYOUT_GRID     "\xee\xa7\x90"  // U+E9D0 - layout-grid
#define ICON_LU_GRID_3X3        "\xee\xa6\x88"  // U+E988 - grid-3x3

// ============================================
// Math & Functions
// ============================================
#define ICON_LU_FUNCTION_SQUARE "\xee\xa5\xb8"  // U+E978 - function-square
#define ICON_LU_CALCULATOR      "\xee\xa3\x94"  // U+E8D4 - calculator
#define ICON_LU_PLUS            "\xee\xa9\x80"  // U+EA40 - plus
#define ICON_LU_MINUS           "\xee\xa8\x80"  // U+EA00 - minus
#define ICON_LU_X               "\xee\xad\x88"  // U+EB48 - x
#define ICON_LU_DIVIDE          "\xee\xa4\x90"  // U+E910 - divide
#define ICON_LU_PERCENT         "\xee\xa8\xb0"  // U+EA30 - percent
#define ICON_LU_SIGMA           "\xee\xab\x88"  // U+EAC8 - sigma
#define ICON_LU_EQUAL           "\xee\xa4\xb0"  // U+E930 - equal

// ============================================
// Charts & Analytics
// ============================================
#define ICON_LU_BAR_CHART       "\xee\xa3\x80"  // U+E8C0 - bar-chart
#define ICON_LU_LINE_CHART      "\xee\xa7\x98"  // U+E9D8 - line-chart
#define ICON_LU_PIE_CHART       "\xee\xa8\xc0"  // U+EA40 - pie-chart
#define ICON_LU_SCATTER_CHART   "\xee\xaa\x98"  // U+EAA8 - scatter-chart
#define ICON_LU_TRENDING_UP     "\xee\xac\x90"  // U+EB10 - trending-up
#define ICON_LU_TRENDING_DOWN   "\xee\xac\x88"  // U+EB08 - trending-down
#define ICON_LU_ACTIVITY        "\xee\xa2\x80"  // U+E880 - activity

// ============================================
// Arrows & Navigation
// ============================================
#define ICON_LU_ARROW_UP        "\xee\xa2\x98"  // U+E898 - arrow-up
#define ICON_LU_ARROW_DOWN      "\xee\xa2\x90"  // U+E890 - arrow-down
#define ICON_LU_ARROW_LEFT      "\xee\xa2\x94"  // U+E894 - arrow-left
#define ICON_LU_ARROW_RIGHT     "\xee\xa2\x9c"  // U+E89C - arrow-right
#define ICON_LU_SHUFFLE         "\xee\xab\x84"  // U+EAC4 - shuffle
#define ICON_LU_REPEAT          "\xee\xaa\x88"  // U+EA88 - repeat
#define ICON_LU_ROTATE_CW       "\xee\xaa\x90"  // U+EA90 - rotate-cw
#define ICON_LU_MOVE            "\xee\xa8\x88"  // U+EA08 - move

// ============================================
// Actions & Operations
// ============================================
#define ICON_LU_FILTER          "\xee\xa5\x80"  // U+E940 - filter
#define ICON_LU_SORT_ASC        "\xee\xab\x94"  // U+EAD4 - sort-asc
#define ICON_LU_SORT_DESC       "\xee\xab\x98"  // U+EAD8 - sort-desc
#define ICON_LU_SEARCH          "\xee\xaa\x9c"  // U+EA9C - search
#define ICON_LU_REFRESH         "\xee\xaa\x84"  // U+EA84 - refresh-cw
#define ICON_LU_TRASH           "\xee\xac\x80"  // U+EB00 - trash-2
#define ICON_LU_EDIT            "\xee\xa4\xa0"  // U+E920 - edit
#define ICON_LU_COPY            "\xee\xa4\x80"  // U+E900 - copy
#define ICON_LU_SCISSORS        "\xee\xaa\xa0"  // U+EAA0 - scissors
#define ICON_LU_CROP            "\xee\xa4\x84"  // U+E904 - crop

// ============================================
// Machine Learning
// ============================================
#define ICON_LU_NETWORK         "\xee\xa8\x90"  // U+EA10 - network
#define ICON_LU_GIT_BRANCH      "\xee\xa5\xc0"  // U+E970 - git-branch
#define ICON_LU_GIT_MERGE       "\xee\xa5\xc8"  // U+E978 - git-merge
#define ICON_LU_WORKFLOW        "\xee\xad\x80"  // U+EB40 - workflow
#define ICON_LU_SITEMAP         "\xee\xab\x8c"  // U+EACC - sitemap

// ============================================
// Shapes & UI
// ============================================
#define ICON_LU_CIRCLE          "\xee\xa3\xa0"  // U+E8E0 - circle
#define ICON_LU_SQUARE          "\xee\xab\xa0"  // U+EAE0 - square
#define ICON_LU_TRIANGLE        "\xee\xac\x84"  // U+EB04 - triangle
#define ICON_LU_CHECK           "\xee\xa3\x9c"  // U+E8DC - check
#define ICON_LU_CHECK_CIRCLE    "\xee\xa3\x98"  // U+E8D8 - check-circle

// ============================================
// Signals & Waves
// ============================================
#define ICON_LU_WAVE            "\xee\xac\xb0"  // U+EB30 - waves
#define ICON_LU_AUDIO_LINES     "\xee\xa2\xb0"  // U+E8B0 - audio-lines
#define ICON_LU_SLIDERS         "\xee\xab\x9c"  // U+EADC - sliders

// ============================================
// Text & Typography
// ============================================
#define ICON_LU_TYPE            "\xee\xac\x98"  // U+EB18 - type
#define ICON_LU_ALIGN_LEFT      "\xee\xa2\x88"  // U+E888 - align-left
#define ICON_LU_LIST            "\xee\xa7\xa0"  // U+E9E0 - list
#define ICON_LU_HASH            "\xee\xa6\x80"  // U+E980 - hash
#define ICON_LU_TAGS            "\xee\xac\x00"  // U+EB00 - tags

// ============================================
// Media & Images
// ============================================
#define ICON_LU_IMAGE           "\xee\xa6\x98"  // U+E998 - image
#define ICON_LU_MUSIC           "\xee\xa8\x84"  // U+EA04 - music
#define ICON_LU_VIDEO           "\xee\xac\xa8"  // U+EBA8 - video

// ============================================
// Settings & Config
// ============================================
#define ICON_LU_SETTINGS        "\xee\xab\x80"  // U+EAC0 - settings
#define ICON_LU_WRENCH          "\xee\xad\x84"  // U+EB44 - wrench
#define ICON_LU_CODE            "\xee\xa3\xb0"  // U+E8F0 - code
#define ICON_LU_TERMINAL        "\xee\xac\x00"  // U+EB00 - terminal
#define ICON_LU_BUG             "\xee\xa3\x8c"  // U+E8CC - bug
#define ICON_LU_PLUG            "\xee\xa9\x88"  // U+EA48 - plug

// ============================================
// Table Operations
// ============================================
#define ICON_LU_TABLE           "\xee\xab\xb0"  // U+EAF0 - table
#define ICON_LU_TABLE_2         "\xee\xab\xb4"  // U+EAF4 - table-2
#define ICON_LU_COLUMNS         "\xee\xa3\xb8"  // U+E8F8 - columns
#define ICON_LU_ROWS            "\xee\xaa\x94"  // U+EA94 - rows
#define ICON_LU_COMBINE         "\xee\xa3\xbc"  // U+E8FC - combine

// ============================================
// Misc & Default
// ============================================
#define ICON_LU_LINK            "\xee\xa7\x9c"  // U+E9DC - link
#define ICON_LU_UNLINK          "\xee\xac\x9c"  // U+EB1C - unlink
#define ICON_LU_TARGET          "\xee\xac\x00"  // U+EB00 - target
#define ICON_LU_FOCUS           "\xee\xa5\xb0"  // U+E970 - focus
#define ICON_LU_ZAP             "\xee\xad\x90"  // U+EB50 - zap
#define ICON_LU_LIGHTBULB       "\xee\xa7\x94"  // U+E9D4 - lightbulb
#define ICON_LU_GAUGE            "\xee\xa5\xbc"  // U+E97C - gauge
