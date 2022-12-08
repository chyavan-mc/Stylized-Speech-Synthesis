from configparser import ConfigParser
import os 

current_dir = os.path.dirname(os.path.realpath(__file__))
CONFIG = ConfigParser()

if __name__ == "__main__":
    CONFIG.add_section("files")
    CONFIG.set("files", "google_filepath", "../data/vox1_indian/id10000/GoogleSamples/")
    CONFIG.set("files", "audio_root", "../data/vox1_indian")
    CONFIG.set("files", "base_data", "../data/voice_data.csv")
    CONFIG.set("files", "base_data_spectro", "../data/voice_data_image.pkl")
    CONFIG.set("files", "train_test_data", "../data/train_test.npy")
    CONFIG.set("files", "reconst_train_test_data", "../data/reconstruction_train_test.npy")
    CONFIG.set("files", "meta_data", "../data/vox1_meta.csv")
    CONFIG.set("files", "speaker_embedding", "../data/speaker_embeddings.csv")
    
    CONFIG.set("files", "recog_model", "../data/models/recognition_model")
    CONFIG.set("files", "recog_history", "../data/models/recognition_history.csv")
    
    CONFIG.set("files", "reconst_models", "../data/models/reconstruction_models/")

    with open(os.path.join(current_dir, 'config.ini'), 'w') as configfile:
        CONFIG.write(configfile)
else:
    CONFIG.read(os.path.join(current_dir, 'config.ini'))