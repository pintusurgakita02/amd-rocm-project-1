import torch, whisper, argparse
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", required=True)
    p.add_argument("--model", default="large-v3")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    model = whisper.load_model(args.model, device=device)
    result = model.transcribe(args.audio, fp16=True)
    for seg in result["segments"]:
        print(f"[{seg['start']:.1f}s] {seg['text']}")
if __name__ == "__main__": main()
