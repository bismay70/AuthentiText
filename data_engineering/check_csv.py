import os

files_to_check = ['manual_bank.csv', 'gemini.csv']

for file_name in files_to_check:
    if not os.path.exists(file_name):
        continue
        
    print(f"🔍 Scanning {file_name} for formatting issues...")
    with open(file_name, 'r', encoding='utf-8') as f:
        header = f.readline()
        expected_commas = header.count(',')
        
        for line_num, line in enumerate(f, start=2): # Start at 2 because header is line 1
            # Simple check: if a line has physical line breaks or unquoted commas, comma count changes
            actual_commas = line.count(',')
            
            # Line 68 is our primary suspect!
            if line_num == 68 or actual_commas != expected_commas:
                print(f"\n🚨 Issue detected on Line {line_num} of {file_name}:")
                print(f"   Expected {expected_commas} commas, but found {actual_commas}")
                print(f"   Raw Line Content:\n   {line.strip()[:150]}...")