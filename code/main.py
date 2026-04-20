from voice_model import VoiceModel
from spiral_model import SpiralModel

def main():
    print("Initializing Parkinson's Detection System...")

    # Voice Model
    voice = VoiceModel("dataset/parkinsons.csv")
    voice.train()

    # Spiral Model
    spiral = SpiralModel("dataset/spiral")
    spiral.train()

    print("\nSystem Training Completed!")

if __name__ == "__main__":
    main()
