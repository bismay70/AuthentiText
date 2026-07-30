import os
import re

def repair_file(filename):
    if not os.path.exists(filename):
        print(f"⚠️ {filename} not found, skipping.")
        return
    
    print(f"🛠️ Rebuilding internal text bounds for {filename}...")
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    header = lines[0]
    repaired_lines = [header]
    
    current_row = ""
    
    for line in lines[1:]:
        # Match lines that start with a standard CSV index structure (e.g., "1,Category,model,")
        if re.match(r'^\d+,[^,]+,[^,]+,', line):
            if current_row:
                # Save the completed previous row block before opening a new one
                repaired_lines.append(current_row.strip() + "\n")
            current_row = line.strip()
        else:
            # If line doesn't start with structural tags, it belongs to the text block above it
            # Append it to the current row with an inline newline character instead of a physical break
            if current_row:
                current_row += " " + line.strip()
            else:
                # Fallback for unexpected trailing text blocks at the top
                current_row = line.strip()
                
    if current_row:
        repaired_lines.append(current_row.strip() + "\n")
        
    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(repaired_lines)
    print(f"✅ {filename} structure fully restored!")

# Run repairs over both your manual collection files
repair_file('manual_bank.csv')
repair_file('gemini.csv')