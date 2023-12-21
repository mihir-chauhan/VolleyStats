import re

def replace_substring(test_str, s1, s2):
    # Replacing all occurrences of substring s1 with s2
    test_str = re.sub(s1, s2, test_str)
    return test_str


# Using readlines()
file1 = open('piplists.txt', 'r')
Lines = file1.readlines()
 
count = 0
# Strips the newline character
for line in Lines:
    count += 1
    data = str(line.strip())
    firstSpace = data.index(' ')
    print("pip3 install " + data[:firstSpace] + "==" + replace_substring(data[firstSpace:], " ", ""))
