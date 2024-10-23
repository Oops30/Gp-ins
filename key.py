from random import randint

def diffie_hellman():
    # Public parameters
    P = 23  # A prime number
    G = 9   # A primitive root modulo P
    print('The Value of P is : %d' % P)
    print('The Value of G is : %d' % G)

    # Private keys
    a = 4  # Alice's private key
    print('Secret Number for Alice is : %d' % a)
    x = pow(G, a, P)  # Alice's public key

    b = 6  # Bob's private key
    print('Secret Number for Bob is : %d' % b)
    y = pow(G, b, P)  # Bob's public key

    # Shared secret key calculations
    ka = pow(y, a, P)  # Alice's shared secret key
    kb = pow(x, b, P)  # Bob's shared secret key

    print('Secret key for Alice is : %d' % ka)
    print('Secret key for Bob is : %d' % kb)

if __name__ == '__main__':
    diffie_hellman()
