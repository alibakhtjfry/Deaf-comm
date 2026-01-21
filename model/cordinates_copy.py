file_path = r"C:\Users\ctrend\Deaf-comm\model\Data\tmp\test.skels"
line_number = 3             # line number to copy (1-based index)

# Read all lines
with open(file_path, "r", encoding="utf-8") as file:
    lines = file.readlines()

# Check if line exists
if 1 <= line_number <= len(lines):
    line_to_copy = lines[line_number - 1]

    # Append the line to the same file
    with open(file_path, "a", encoding="utf-8") as file:
        file.write(line_to_copy)
else:
    print("Line number out of range")
