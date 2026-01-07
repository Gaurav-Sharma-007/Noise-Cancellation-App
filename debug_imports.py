
import time
print("Starting imports...")

start = time.time()
print("Importing numpy...")
import numpy
print(f"numpy imported in {time.time() - start:.2f}s")

start = time.time()
print("Importing soundfile...")
import soundfile
print(f"soundfile imported in {time.time() - start:.2f}s")

start = time.time()
print("Importing matplotlib...")
import matplotlib.pyplot
print(f"matplotlib imported in {time.time() - start:.2f}s")

start = time.time()
print("Importing librosa...")
import librosa
print(f"librosa imported in {time.time() - start:.2f}s")

start = time.time()
print("Importing tensorflow...")
import tensorflow
print(f"tensorflow imported in {time.time() - start:.2f}s")

print("All imports successful.")
