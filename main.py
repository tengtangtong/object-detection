from scene_recog_cnn import test

# Sample code for how to run model on testing images
if __name__ == '__main__':
    print("Starting test evaluation...")
    accuracy = test(test_data_dir='./data/test/', model_dir='.')
    print(f"Testing complete. Accuracy: {accuracy*100:.2f}%")