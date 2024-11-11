from summary import response_generator
from save import file_path
from work_on_text import answer_question


def main():
    while True:
        print("1. Summary of the audio")
        print("2. query from the audio.")
        choice = input("Enter your choice (1/2): ")

        if choice == "1":
            response = response_generator(file_path)
            print("Summary of the audio file: ", response)

        elif choice == "2":
            query = input("Enter your query: ")
            response = answer_question(query)
            print("Query : " ,query , "\nResponse: ", response)
        else:
            print("Invalid choice!")
            return

if __name__ == "__main__":
    main()