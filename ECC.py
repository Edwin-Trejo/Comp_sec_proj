import kagglehub
from tinyec import registry
import secrets
import pickle
import os
import numpy as np
import time

""" Make sure to install both kagglehub and tinyec using 'pip install name' """
# kagglehub documentation taken from "https://github.com/Kaggle/kagglehub"
cifar = kagglehub.dataset_download('fedesoriano/cifar100')
print("Path: ",cifar)

def load_cifar100(dataset_path):
    with open(os.path.join(dataset_path, "train"), 'rb') as f:
        train_data = pickle.load(f, encoding='latin1')
    with open(os.path.join(dataset_path, "test"), 'rb') as f:
        test_data = pickle.load(f, encoding='latin1')
    return train_data, test_data

  # ECC documentation take from "https://www.geeksforgeeks.org/blockchain-elliptic-curve-cryptography/"      
train_data, test_data = load_cifar100(cifar)
x_train = np.array(train_data['data'])
x_test = np.array(test_data['data'])

# Normalize data (important for ECC encryption since we are working with numbers)
x_train = x_train / 255.0
x_test = x_test / 255.0


curve = registry.get_curve('brainpoolP256r1')

def compress_point(publicKey):
    return hex(publicKey.x) + hex(publicKey.y % 2)[2:]

def encryption_key(data, pubKey):
    # Generate private key
    cipherPrivKey = secrets.randbelow(curve.field.n)
    # Public key
    cipherPubKey = cipherPrivKey * curve.g
    sharedKey = pubKey * cipherPrivKey
    
    # Scale and convert data to integers
    data_scaled = (data * 255).astype(int)  # Scale data to range [0, 255]
    encrypted_data = np.bitwise_xor(data_scaled, int(sharedKey.x))  # XOR operation
    return encrypted_data, cipherPubKey


def decryption_key(encrypted_data, cipherPubKey, privKey):
    # Compute shared key
    sharedKey = cipherPubKey * privKey
    
    # Perform XOR to retrieve scaled data
    decrypted_data_scaled = np.bitwise_xor(encrypted_data, int(sharedKey.x))
    
    # Convert back to float and normalize
    decrypted_data = decrypted_data_scaled.astype(float) / 255
    return decrypted_data

privKey = secrets.randbelow(curve.field.n)
pubKey = privKey * curve.g

print("Private key: ", hex(privKey))
print("Public Key: ",compress_point(pubKey))

# reduced numbers of sample to prevent crashing
num_samples = 5000
x_train_sample = x_train[:num_samples]

# Encryption Process
#Timing encryption
start_encryption = time.time()
encrypted_data=  []
cipherPubKeys = []

for sample in x_train_sample:
    enc, cipherPubKey = encryption_key(sample.flatten(), pubKey)
    encrypted_data.append(enc)
    cipherPubKeys.append(cipherPubKey)
    
#Decryption Process
encryption_time = time.time() - start_encryption
print(f"Encryption Time: {encryption_time:.2f} seconds for {num_samples} samples.")

start_decryption = time.time()
decrypted_data = []

for enc, cipherPubKey in zip(encrypted_data, cipherPubKeys):
    dec = decryption_key(enc, cipherPubKey, privKey)
    decrypted_data.append(dec)

decryption_time = time.time() - start_decryption
print(f"Decryption Time: {decryption_time:.2f} seconds for {num_samples} samples.")


# Validate data integrity to find any loss or corruption
decrypted_data = np.array(decrypted_data).reshape(x_train_sample.shape)
loss = np.sum(np.abs(decrypted_data - x_train_sample)) / np.prod(x_train_sample.shape)
print(f"Data Loss: {loss * 100:.2f}%")

if np.allclose(decrypted_data, x_train_sample, atol=1e-5):
    print("Decryption successful: No data loss detected.")
else:
    print("Decryption failed: Data mismatch.")