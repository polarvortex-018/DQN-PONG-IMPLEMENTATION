import os
import sys
import tensorflow as tf

def verify_gpu():
    print(f"Python version: {sys.version}")
    print(f"TensorFlow version: {tf.__version__}")
    
    # Try to find NVIDIA pip package DLLs and add them
    venv_base = os.path.dirname(os.path.dirname(sys.executable))
    nvidia_base = os.path.join(venv_base, "Lib", "site-packages", "nvidia")
    
    if os.path.exists(nvidia_base):
        print(f"Found NVIDIA base at: {nvidia_base}")
        for folder in os.listdir(nvidia_base):
            bin_dir = os.path.join(nvidia_base, folder, "bin")
            if os.path.isdir(bin_dir):
                print(f"Adding DLL directory: {bin_dir}")
                os.add_dll_directory(bin_dir)
    
    # Set verbose logging to see why DLLs fail to load
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Ensure at least one is "visible"
    
    # Refresh/check devices
    print("\nAttempting to list GPUs...")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPU check result: {gpus}")
    
    if gpus:
        print("✅ SUCCESS: GPU detected!")
        for gpu in gpus:
            print(f"  - {gpu}")
        
        # Try a simple operation to confirm it actually works
        try:
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                c = tf.matmul(a, b)
                print("✅ Test calculation on GPU successful!")
        except Exception as e:
            print(f"❌ Operation on GPU failed: {e}")
    else:
        print("❌ FAILURE: GPU still not detected.")
        print("\nChecking for missing DLLs:")
        if os.path.exists(nvidia_base):
            for folder in os.listdir(nvidia_base):
                 bin_path = os.path.join(nvidia_base, folder, "bin")
                 if os.path.isdir(bin_path):
                     print(f"Contents of {folder}/bin: {os.listdir(bin_path)}")
        
        print("\nChecking system PATH for other CUDA installations...")
        print(f"PATH contains 'CUDA': {'CUDA' in os.environ.get('PATH', '')}")

if __name__ == "__main__":
    verify_gpu()
