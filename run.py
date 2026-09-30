from app import app


if __name__ == "__main__":
    print("\n========================================")
    print("AI FITNESS PROJECT")
    print("========================================")
    print("\nStarting application...")
    print("Open: http://127.0.0.1:5000")
    print("\nPress CTRL+C to stop the server.\n")

    app.run(
        debug=True
    )