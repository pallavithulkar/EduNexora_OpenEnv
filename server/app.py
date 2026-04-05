from app import app

def main():
    print("EduNexora Server Ready")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
