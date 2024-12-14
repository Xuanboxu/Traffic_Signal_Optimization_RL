def calculate_average(file_path):
    try:
        # Open the file and read all lines
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Convert lines to a list of numbers
        numbers = [float(line.strip()) for line in lines if line.strip().isdigit()]
        
        # Calculate the average if there are numbers
        if numbers:
            average = sum(numbers) / len(numbers)
        else:
            return "The file contains no valid numbers to calculate an average."
        
        return f"The average is: {average:.2f}"
    
    except FileNotFoundError:
        return "The specified file was not found."
    except ValueError:
        return "The file contains non-numeric values."

# Specify the path to your text file
file_path = './model/test/plot_queue_data.txt'

# Call the function and print the result
print(calculate_average(file_path))
