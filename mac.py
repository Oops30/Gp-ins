#MD5
import hashlib

# Create MD5 hash of a message
result = hashlib.md5(b'Ismile')
result1 = hashlib.md5(b'Esmile')

print("MD5 Hash of 'Ismile':", result.digest())
print("MD5 Hash of 'Esmile':", result1.digest())

#SHA1
import hashlib

# Input string to encode
str = input("Enter the value to encode: ")
result = hashlib.sha1(str.encode())

print("SHA1 Hash:", result.hexdigest())
