"""
Verifica que TF detecta la RTX 3050 Ti via CUDA en el entorno condor-tf210-gpu.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs found: {len(gpus)}")
for g in gpus:
    print(f"  {g.name}")

if not gpus:
    print("\nERROR: No GPU detectada.")
    print("Posibles causas:")
    print("  - cuDNN DLLs no en PATH del env (verifica Library/bin)")
    print("  - CUDA toolkit no cargado correctamente")
    sys.exit(1)

# Enable memory growth
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

# Quick GPU compute test
import numpy as np
with tf.device('/GPU:0'):
    a = tf.constant(np.random.randn(1000, 1000).astype('float32'))
    b = tf.constant(np.random.randn(1000, 1000).astype('float32'))
    c = tf.matmul(a, b)
    _ = c.numpy()

print("\n✓ GPU compute test passed")
print("✓ RTX 3050 Ti operativa via CUDA nativo")
print("\nListo para entrenar. Selecciona kernel 'CONDOR GPU (CUDA 11.2)' en VSCode.")
