Set WshShell = CreateObject("WScript.Shell")
' Lấy đường dẫn hiện tại của file script
strPath = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)

' Lệnh chạy: python src/teacher_tool/main_app_modern.py
' Chr(34) là dấu ngoặc kép " để xử lý đường dẫn có khoảng trắng
strCommand = "python " & Chr(34) & strPath & "\src\teacher_tool\main_app_modern.py" & Chr(34)

' Chạy lệnh ẩn (số 0 = ẩn cửa sổ console)
WshShell.Run strCommand, 0, False
