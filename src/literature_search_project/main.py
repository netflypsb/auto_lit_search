from crew import kickoff

def main():
    research_title = input("Enter Research Title: ")
    result = kickoff(research_title)
    for task_result in result:
        print(task_result['task']['description'])
        print(task_result['output'])
    print("Literature search completed successfully!")

if __name__ == "__main__":
    main()
