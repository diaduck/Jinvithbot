import re

input_file = "jinvithoughts_cleaned.txt"
output_file = "jinvithoughts_flat.txt"

with open(input_file, "r", encoding="utf-8") as f:
    content = f.read()

collapsed = re.sub(r'\n{2,}', '\n\n', content)

with open(output_file, "w", encoding="utf-8") as f:
    f.write(collapsed)

print("Collapsed multiple line breaks. Saved as:", output_file)
