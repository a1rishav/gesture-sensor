common = {
    "Thumbs Up": "pageup",
    "Thumbs Down": "pagedown"
}
chrome = {
    "Thumbs Up": "up",
    "Thumbs Down": "down",
    "Thumbs Left": ["ctrl", "tab"],
    "Thumbs Right": ["ctrl", "shift", "tab"]
}
common.update(chrome)
print(common)