
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
import binascii

# Generate RSA key pair
keyPair = RSA.generate(1024)  # 1024-bit key
pubKey = keyPair.publickey()   # Public key

# Display public key
print(f"Public key: (n={hex(pubKey.n)}, e={hex(pubKey.e)})")
pubKeyPEM = pubKey.exportKey()
print("Public Key in PEM format:")
print(pubKeyPEM.decode('ascii'))

# Display private key
print(f"Private key: (n={hex(pubKey.n)}, d={hex(keyPair.d)})")
privKeyPEM = keyPair.exportKey()
print("Private Key in PEM format:")
print(privKeyPEM.decode('ascii'))

# Encryption process
msg = 'Ismile Academy'  # Plaintext message
encryptor = PKCS1_OAEP.new(pubKey)  # Initialize the encryptor with the public key
encrypted = encryptor.encrypt(msg.encode('utf-8'))  # Encrypt the message
print("Encrypted:", binascii.hexlify(encrypted))  # Display encrypted message in hexadecimal

# Decryption process
decryptor = PKCS1_OAEP.new(keyPair)  # Initialize the decryptor with the private key
decrypted = decryptor.decrypt(encrypted)  # Decrypt the message
print("Decrypted:", decrypted.decode('utf-8'))  # Display the original message
