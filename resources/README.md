# CyxWiz Resources

This directory contains application resources such as icons, images, and assets.

## Required Files

- `cyxwiz.png` - Application window icon (recommended: 256x256 or 512x512 PNG)
- `cyxwiz.ico` - Windows application icon (for .exe embedding)

## Icon Setup

To add the CyxWiz icon:

1. Place your `cyxwiz.png` file in this directory
2. The application will automatically load it as the window icon

For Windows .exe icon:
1. Convert your PNG to ICO format (use online converter or ImageMagick)
2. Place `cyxwiz.ico` in this directory
3. It will be embedded during build via CMake
