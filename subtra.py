'''
Substitution Cipher (Caesar Cipher):
Definition: In a substitution cipher, each letter in the plaintext 
is replaced by another letter. The Caesar Cipher
 is a type of substitution cipher where each letter 
 is shifted by a certain number of positions in the alphabet (key).
'''
# Caesar Cipher Encryption
def encrypt(text, s):
    result = ""
    # Traverse the text
    for i in range(len(text)):
        char = text[i]
        # Encrypt uppercase characters
        if char.isupper():
            result += chr((ord(char) + s - 65) % 26 + 65)
        # Encrypt lowercase characters
        else:
            result += chr((ord(char) + s - 97) % 26 + 97)
    return result

# Input and test the encryption function
text = input("Enter the text to encrypt: ")
s = 3  # Shift value
print("Text:", text)
print("Cipher:", encrypt(text, s))

'''
Transposition Cipher (Railfence Cipher):
Definition: In a transposition cipher, the letters of the
 plaintext are rearranged according to a specific system.
The Railfence Cipher arranges the letters in a zigzag pattern
 across multiple lines (rails) and reads them row by row to form 
 the ciphertext.
'''

# Railfence Cipher Encryption
def RailFence(txt):
    result = ""
    # First, pick characters at even indexes
    for i in range(len(txt)):
        if i % 2 == 0:
            result += txt[i]
    # Then, pick characters at odd indexes
    for i in range(len(txt)):
        if i % 2 != 0:
            result += txt[i]
    return result

# Input and test the Railfence encryption
string = input("Enter a string: ")
print("Cipher:", RailFence(string))
