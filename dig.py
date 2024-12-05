from Crypto.Signature import pkcs1_15
from Crypto.Hash import SHA256
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes

def generate_signature(private_key, message):
    key = RSA.import_key(private_key)
    hashed_message = SHA256.new(message.encode())
    return pkcs1_15.new(key).sign(hashed_message)

def verify_signature(public_key, message, signature):
    key = RSA.import_key(public_key)
    hashed_message = SHA256.new(message.encode())
    try:
        pkcs1_15.new(key).verify(hashed_message, signature)
        return True
    except (ValueError, TypeError):
        return False

# Generate RSA key pair
key_pair = RSA.generate(2048)
public_key = key_pair.publickey().export_key()
private_key = key_pair.export_key()

# Example usage
message = "Hello, World!"
signature = generate_signature(private_key, message)
print("Generated Signature:", signature)

is_valid = verify_signature(public_key, message, signature)
print("Signature Verification Result:", is_valid)

