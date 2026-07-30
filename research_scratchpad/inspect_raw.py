import os

csv_path = "D:/Aiml/data/colab/train.csv"

print("🔍 Verification Check...")
if not os.path.exists(csv_path):
    print(f"❌ Target missing! Cannot find the file at: {csv_path}")
    import sys; sys.exit()

print(f"📁 Target verified size: {os.path.getsize(csv_path)} bytes")

print("\n📋 Printing the raw first 5 lines of the CSV file directly:")
print("=" * 60)
with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
    for i in range(5):
        line = f.readline()
        if not line:
            print(f"[Line {i+1} is completely empty/End of file reached]")
            break
        print(f"Line {i+1}: {line.strip()}")
print("=" * 60)